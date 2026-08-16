#!/usr/bin/env python3
"""CPU demo: PDA+cache constrained JSON + Earley baseline agreement check."""

from decode import EarleyTokenMasker, TokenMasker, generate_n
from grammar import JSON_EBNF, accepts, parse_ebnf


def main() -> None:
    g = parse_ebnf(JSON_EBNF)
    samples = generate_n(8, seed=2)
    for s in samples:
        print(f"  {s!r}  complete={accepts(g, s)}")
    masker = TokenMasker()
    brace = [masker.vocab.index("{")]
    legal = {masker.vocab[i] for i in masker.legal_ids(brace)}
    print("legal after '{':", sorted(legal))
    print("'true' legal after '{'? ", "true" in legal)
    print("cache JIT nodes:", masker.cache_stats()["compiled_nodes"])
    ear = EarleyTokenMasker()
    assert set(masker.legal_ids(brace)) == set(ear.legal_ids(brace))
    print("PDA mask ≡ Earley baseline after '{'")


if __name__ == "__main__":
    main()
