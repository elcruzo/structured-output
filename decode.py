"""Token-level CFG mask + logits processor. Multi-char tokens are accepted only if every char stays in-language."""

from __future__ import annotations

import math

import numpy as np
import torch

from grammar import JSON_EBNF, Earley, Grammar, parse_ebnf

# Toy BPE-like pieces, including multi-char tokens that span lexer terminals.
JSON_VOCAB: list[str] = [
    "{",
    "}",
    "[",
    "]",
    '"',
    ":",
    ",",
    "true",
    "false",
    "null",
    "-",
    ".",
    " ",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "a",
    "b",
    "c",
    ": ",
    ":true",
]


class TokenMasker:
    def __init__(self, grammar: Grammar | None = None, vocab: list[str] | None = None) -> None:
        self.grammar = grammar or parse_ebnf(JSON_EBNF)
        self.vocab = vocab if vocab is not None else list(JSON_VOCAB)
        self._cache: dict[str, list[int]] = {}

    def prefix_of(self, emitted: list[int]) -> str:
        return "".join(self.vocab[i] for i in emitted)

    def legal_ids(self, emitted: list[int]) -> list[int]:
        prefix = self.prefix_of(emitted)
        hit = self._cache.get(prefix)
        if hit is not None:
            return hit
        base = Earley(self.grammar)
        if prefix and not base.feed(prefix):
            self._cache[prefix] = []
            return []
        legal: list[int] = []
        mark = base.n_cols
        for i, tok in enumerate(self.vocab):
            if not tok:
                continue
            if base.feed(tok):
                legal.append(i)
            base.rollback(mark)
        self._cache[prefix] = legal
        return legal

    def is_complete(self, emitted: list[int] | str) -> bool:
        s = emitted if isinstance(emitted, str) else self.prefix_of(emitted)
        p = Earley(self.grammar)
        return p.feed(s) and p.is_complete()


class LogitsProcessor:
    def __init__(self, masker: TokenMasker) -> None:
        self.masker = masker

    def __call__(self, logits: torch.Tensor, emitted: list[int]) -> torch.Tensor:
        legal = set(self.masker.legal_ids(emitted))
        out = logits.clone()
        v = out.shape[-1]
        neg = torch.tensor(-math.inf, dtype=out.dtype, device=out.device)
        for i in range(v):
            if i not in legal:
                out[..., i] = neg
        return out


_CLOSERS = frozenset({"}", "]", '"'})


def generate(
    masker: TokenMasker,
    rng: np.random.Generator | None = None,
    max_tokens: int = 48,
    logits: torch.Tensor | None = None,
) -> str:
    """Sample a completion in the grammar. Prefers closers as length grows so JSON finishes."""
    rng = rng or np.random.default_rng()
    proc = LogitsProcessor(masker) if logits is not None else None
    emitted: list[int] = []
    for step in range(max_tokens):
        legal = masker.legal_ids(emitted)
        done = masker.is_complete(emitted)
        if done and (not legal or rng.random() < min(0.35 + 0.04 * step, 0.92)):
            break
        if not legal:
            break
        if step > 8:
            closing = [i for i in legal if masker.vocab[i] in _CLOSERS]
            if closing and rng.random() < 0.65:
                legal = closing
        if proc is not None:
            assert logits is not None
            masked = proc(logits, emitted)
            # Restrict to the (possibly closer-biased) legal subset.
            row = masked.detach().cpu().float().flatten()
            scores = np.array([float(row[i]) if i < row.numel() else -1e9 for i in legal], dtype=np.float64)
            scores = np.where(np.isfinite(scores), scores, -1e9)
            scores -= scores.max()
            p = np.exp(scores)
            z = p.sum()
            if z <= 0:
                break
            pick = int(legal[int(rng.choice(len(legal), p=p / z))])
        else:
            pick = int(legal[int(rng.integers(0, len(legal)))])
        emitted.append(pick)
    text = masker.prefix_of(emitted)
    if not masker.is_complete(text):
        raise RuntimeError(f"generation did not complete: {text!r}")
    return text


def generate_n(n: int, seed: int = 0, max_tries: int = 8) -> list[str]:
    masker = TokenMasker()
    rng = np.random.default_rng(seed)
    out: list[str] = []
    for i in range(n):
        last_err: Exception | None = None
        for _ in range(max_tries):
            try:
                out.append(generate(masker, rng=np.random.default_rng(int(rng.integers(0, 1_000_000_000)))))
                last_err = None
                break
            except RuntimeError as e:
                last_err = e
        if last_err:
            raise last_err
    return out
