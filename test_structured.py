"""CFG masker invariants: PDA default ≡ Earley baseline, cache, nesting, json.loads."""

import json

import numpy as np
import torch

from decode import JSON_VOCAB, EarleyTokenMasker, LogitsProcessor, TokenMasker, generate, generate_n
from grammar import JSON_EBNF, Earley, accepts, parse_ebnf, valid_prefix
from pda import PDA, TokenMaskCache, accepts_pda, classify_token, valid_prefix_pda


def test_hundred_random_samples_parse_and_json_loads():
    samples = generate_n(100, seed=0)
    g = parse_ebnf(JSON_EBNF)
    assert len(samples) == 100
    assert len(set(samples)) > 1
    parsed = []
    for s in samples:
        assert accepts(g, s), s
        assert accepts_pda(g, s), s
        parsed.append(json.loads(s))
    assert any(isinstance(v, (dict, list)) and v for v in parsed)


def test_nested_arrays_objects():
    g = parse_ebnf(JSON_EBNF)
    s = '{"a":[1,{"b":true}]}'
    for i in range(len(s) + 1):
        assert valid_prefix(g, s[:i]), s[:i]
        assert valid_prefix_pda(g, s[:i]), s[:i]
    assert accepts(g, s)
    assert accepts_pda(g, s)
    assert json.loads(s) == {"a": [1, {"b": True}]}
    deep = '[{"a":{"b":[false,null,2]}}]'
    assert accepts(g, deep)
    assert accepts_pda(g, deep)
    json.loads(deep)


def test_pda_agrees_with_earley_on_prefixes_and_reject():
    g = parse_ebnf(JSON_EBNF)
    good = ['', '{', '{"a"', '{"a":', '{"a":true}', '[1,2]', 'null', 'false']
    bad = ['}', '{:', 'tru', '{"a":tru}', '[1,]', '01', '{a}']
    for s in good:
        assert valid_prefix(g, s) == valid_prefix_pda(g, s), s
        assert accepts(g, s) == accepts_pda(g, s), s
    for s in bad:
        assert valid_prefix(g, s) == valid_prefix_pda(g, s), s
        assert not accepts(g, s) and not accepts_pda(g, s), s
    raw_blob = '{"title":"Attention Is All You Need","x":' + "y" * 200 + "}"
    assert not accepts(g, raw_blob)
    assert not accepts_pda(g, "{")
    assert not accepts(g, "not json")


def test_pda_mask_matches_earley_baseline():
    g = parse_ebnf(JSON_EBNF)
    vocab = list(JSON_VOCAB)
    pda_m = TokenMasker(g, vocab)
    ear_m = EarleyTokenMasker(g, vocab)
    # Only prefixes that greedy-match the toy vocab (no bare ``t`` — use full ``true``).
    prefixes = ["", "{", '{"', '{"a', '{"a"', '{"a":', '{"a":true', "[", "[1", "[1,"]
    for pref in prefixes:
        emitted: list[int] = []
        i = 0
        while i < len(pref):
            hit = None
            for tok in sorted(vocab, key=len, reverse=True):
                if pref.startswith(tok, i):
                    hit = tok
                    break
            assert hit is not None, pref[i:]
            emitted.append(vocab.index(hit))
            i += len(hit)
        assert set(pda_m.legal_ids(emitted)) == set(ear_m.legal_ids(emitted)), pref


def test_multichar_token_masked_by_context():
    vocab = ["{", "}", '"', "a", ":", "true", "1", ":true", " "]
    masker = TokenMasker(parse_ebnf(JSON_EBNF), vocab)
    tid = {t: i for i, t in enumerate(vocab)}

    def legal_after(text: str) -> set[str]:
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


def test_token_mask_cache_partitions_and_jit():
    g = parse_ebnf(JSON_EBNF)
    pda = PDA(g)
    cache = TokenMaskCache(pda, list(JSON_VOCAB))
    assert cache._nodes == {}
    start = g.start
    mask = cache.compile_node(start, pda.fsas[start].start)
    assert mask.kind in ("accept_heavy", "reject_heavy", "bitset")
    # Ground-truth partition via classify_token (pairwise disjoint ∪ covers vocab).
    parts = {"accepted": set(), "rejected": set(), "dependent": set()}
    for tid, tok in enumerate(JSON_VOCAB):
        parts[classify_token(pda, start, pda.fsas[start].start, tok)].add(tid)
    vocab_ids = set(range(len(JSON_VOCAB)))
    assert parts["accepted"] | parts["rejected"] | parts["dependent"] == vocab_ids
    assert parts["accepted"].isdisjoint(parts["rejected"])
    assert parts["accepted"].isdisjoint(parts["dependent"])
    assert parts["rejected"].isdisjoint(parts["dependent"])
    acc, rej, dep = set(mask.accepted), set(mask.rejected), set(mask.dependent)
    assert acc.isdisjoint(rej) and acc.isdisjoint(dep) and rej.isdisjoint(dep)
    assert dep == parts["dependent"]
    if mask.kind == "reject_heavy":
        # Stores CI-accepted + dependent only; rejected is the complement.
        assert mask.rejected == ()
        assert acc == parts["accepted"]
        assert len(acc) + len(dep) <= len(JSON_VOCAB)
        assert mask.context_independent_accepted() == parts["accepted"]
    elif mask.kind == "accept_heavy":
        # Stores CI-rejected + dependent only; accepted is the complement.
        assert mask.accepted == ()
        assert rej == parts["rejected"]
        assert len(rej) + len(dep) <= len(JSON_VOCAB)
        assert mask.context_independent_accepted() == parts["accepted"]
    else:
        # Bitset stores the full three-way partition explicitly.
        assert acc == parts["accepted"] and rej == parts["rejected"]
        assert acc | rej | dep == vocab_ids
        assert len(acc) + len(rej) + len(dep) == len(JSON_VOCAB)
        assert mask.context_independent_accepted() == parts["accepted"]
    # JIT: legal_ids compiles stack tops on demand
    m = TokenMasker(g)
    assert m.cache_stats()["compiled_nodes"] == 0
    m.legal_ids([])
    assert m.cache_stats()["compiled_nodes"] >= 1
    # After '{', union of stack tops: "true" never CI-accepted; '"' accepted on some top.
    from pda import PDAMatcher

    matcher = PDAMatcher(pda)
    assert matcher.feed("{")
    kinds_true = {classify_token(pda, st[-1].rule, st[-1].node, "true") for st in matcher.stacks}
    kinds_quote = {classify_token(pda, st[-1].rule, st[-1].node, '"') for st in matcher.stacks}
    assert "accepted" not in kinds_true
    assert "accepted" in kinds_quote
    assert '"' in {JSON_VOCAB[i] for i in TokenMasker(g).legal_ids([JSON_VOCAB.index("{")])}


def test_earley_baseline_still_generates():
    s = generate(EarleyTokenMasker(), rng=np.random.default_rng(3))
    g = parse_ebnf(JSON_EBNF)
    assert accepts(g, s)
    json.loads(s)


def test_earley_item_scan_not_regex_stub():
    """Regression: recognizer must be Earley items, not an always-true / regex stand-in."""
    g = parse_ebnf(JSON_EBNF)
    p = Earley(g)
    assert p.feed("{")
    assert p.is_alive() and not p.is_complete()
    assert p.chart[-1]  # non-empty item set after a valid prefix char
    assert any(it.dot > 0 or it.rule == "object" for it in p.chart[-1])
    assert not accepts(g, "{")
    assert not accepts(g, "not json")
