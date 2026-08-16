"""CFG masker invariants: samples parse, nesting, multi-char context, json.loads."""

import json

import numpy as np
import torch

from decode import JSON_VOCAB, LogitsProcessor, TokenMasker, generate, generate_n
from grammar import JSON_EBNF, accepts, parse_ebnf, valid_prefix


def test_hundred_random_samples_parse_and_json_loads():
    samples = generate_n(100, seed=0)
    g = parse_ebnf(JSON_EBNF)
    assert len(samples) == 100
    assert len(set(samples)) > 1
    parsed = []
    for s in samples:
        assert accepts(g, s), s
        parsed.append(json.loads(s))
    assert any(isinstance(v, (dict, list)) and v for v in parsed)


def test_nested_arrays_objects():
    g = parse_ebnf(JSON_EBNF)
    s = '{"a":[1,{"b":true}]}'
    for i in range(len(s) + 1):
        assert valid_prefix(g, s[:i]), s[:i]
    assert accepts(g, s)
    assert json.loads(s) == {"a": [1, {"b": True}]}
    deep = '[{"a":{"b":[false,null,2]}}]'
    assert accepts(g, deep)
    json.loads(deep)


def test_multichar_token_masked_by_context():
    vocab = ["{", "}", '"', "a", ":", "true", "1", ":true", " "]
    masker = TokenMasker(parse_ebnf(JSON_EBNF), vocab)
    tid = {t: i for i, t in enumerate(vocab)}

    def legal_after(text: str) -> set[str]:
        # Reconstruct emitted token ids by a greedy left-to-right match on this tiny vocab.
        emitted: list[int] = []
        i = 0
        while i < len(text):
            hit = None
            for tok in sorted(vocab, key=len, reverse=True):
                if text.startswith(tok, i):
                    hit = tok
                    break
            assert hit is not None, text[i:]
            emitted.append(tid[hit])
            i += len(hit)
        return {vocab[j] for j in masker.legal_ids(emitted)}

    after_brace = legal_after("{")
    assert "true" not in after_brace
    assert ":true" not in after_brace
    assert '"' in after_brace or "}" in after_brace

    after_colon = legal_after('{"a":')
    assert "true" in after_colon

    after_key = legal_after('{"a"')
    assert ":true" in after_key


def test_logits_processor_sets_illegal_to_neginf():
    masker = TokenMasker()
    proc = LogitsProcessor(masker)
    logits = torch.zeros(len(masker.vocab))
    # After '{', multi-char 'true' is illegal.
    emitted = [masker.vocab.index("{")]
    masked = proc(logits, emitted)
    legal = set(masker.legal_ids(emitted))
    for i, tok in enumerate(masker.vocab):
        if i in legal:
            assert torch.isfinite(masked[i]), tok
        else:
            assert masked[i].item() == float("-inf"), tok
    assert masked[masker.vocab.index("true")].item() == float("-inf")


def test_completed_json_is_loadable_single_generate():
    s = generate(TokenMasker(), rng=np.random.default_rng(1))
    json.loads(s)
