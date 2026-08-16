# 25 — Structured output (CFG constrained decoding)

**Default:** XGrammar-style byte-level **PDA** + **JIT adaptive token-mask cache**
(Dong et al. 2024; cache amortization as in XGrammar-2). Not a regex.

**Named baseline:** incremental **Earley** (`Earley` / `EarleyTokenMasker`) for
prefix/completion correctness audits against the PDA masks.

1. Parse a tiny EBNF subset (literals, char classes, concat, `|`, `?`/`*`/`+`, named rules) into a CFG.
2. Compile each rule to an FSA with character edges and rule-reference edges; a pushdown stack handles recursion (parallel stacks under nondeterminism).
3. At each stack-top node, tokens are partitioned into context-independent accepted / rejected vs context-dependent (validity needs a parent pop). CI masks are JIT-compiled per node and stored accept-heavy / reject-heavy / bitset.
4. Runtime mask = CI lookup ∪ full-stack checks on the dependent set. Vocabulary pieces may span multiple terminals (`true`, `:true`).
5. `LogitsProcessor` sets illegal ids to `-inf` before sampling.

The built-in grammar is a nested JSON subset (objects, arrays, strings, RFC-8259 numbers, bool, null). Generated strings `json.loads`.

## Papers

- Willard & Louf, 2023. *Efficient Guided Generation for Large Language Models* (Outlines) — Earley-guided generation.
- Dong et al., 2024/2025. *XGrammar: Flexible and Efficient Structured Generation Engine* — PDA + context-independent token mask cache.
- Dong et al., 2026. *XGrammar-2* — adaptive cache / JIT for dynamic agentic grammars (Earley in the production engine; this folder keeps PDA as the default educational path and Earley as the named baseline).

## Papers on disk

- [`papers/willard-outlines-2023.pdf`](papers/willard-outlines-2023.pdf) — Willard & Louf (2023) ([arXiv:2307.09702](https://arxiv.org/abs/2307.09702))
- [`papers/dong-xgrammar-2024.pdf`](papers/dong-xgrammar-2024.pdf) — Dong et al. XGrammar (2024) ([arXiv:2411.15100](https://arxiv.org/abs/2411.15100))
- [`papers/dong-xgrammar2-2026.pdf`](papers/dong-xgrammar2-2026.pdf) — Dong et al. XGrammar-2 (2026) ([arXiv:2601.04426](https://arxiv.org/abs/2601.04426))

## Run

```bash
python main.py
python -m pytest test_structured.py -q
```
