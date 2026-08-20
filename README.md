# 25 — Structured output (CFG constrained decoding)

**Default:** byte-level **PDA** plus a Python dict of context-independent token
masks keyed by `(rule, node)` (Dong et al. 2024; cache amortization as in XGrammar-2).

**Named baseline:** incremental **Earley** (`Earley` / `EarleyTokenMasker`) for
prefix/completion correctness audits against the PDA masks.

1. Parse a tiny EBNF subset (literals, char classes, concat, `|`, `?`/`*`/`+`, named rules) into a CFG.
2. Compile each rule to an FSA with character edges and rule-reference edges; a pushdown stack handles recursion (parallel stacks under nondeterminism).
3. At each stack-top node, tokens are partitioned into context-independent accepted / rejected vs context-dependent (validity needs a parent pop). CI masks are compiled per node on first use and stored in `_nodes` as accept-heavy / reject-heavy / bitset id lists.
4. Runtime mask = dict lookup ∪ full-stack checks on the dependent set. Vocabulary pieces may span multiple terminals (`true`, `:true`).
5. `LogitsProcessor` sets illegal ids to `-inf` before sampling.

The built-in grammar is a nested JSON subset (objects, arrays, strings, RFC-8259 numbers, bool, null). Generated strings `json.loads`. `PDAMatcher` rolls back by snapshotting stack sets.

## Papers

- Willard & Louf, 2023. *Efficient Guided Generation for Large Language Models* (Outlines) — Earley-guided generation.
- Dong et al., 2024/2025. *XGrammar: Flexible and Efficient Structured Generation Engine* — PDA + context-independent token mask cache.
- Dong et al., 2026. *XGrammar-2* — adaptive cache / JIT for dynamic agentic grammars. This folder's cache is a `dict[(rule, node)] → NodeMask`; Earley is the named mask oracle.

## Papers on disk

- [`papers/willard-outlines-2023.pdf`](papers/willard-outlines-2023.pdf) — Willard & Louf (2023) ([arXiv:2307.09702](https://arxiv.org/abs/2307.09702))
- [`papers/dong-xgrammar-2024.pdf`](papers/dong-xgrammar-2024.pdf) — Dong et al. XGrammar (2024) ([arXiv:2411.15100](https://arxiv.org/abs/2411.15100))
- [`papers/dong-xgrammar2-2026.pdf`](papers/dong-xgrammar2-2026.pdf) — Dong et al. XGrammar-2 (2026) ([arXiv:2601.04426](https://arxiv.org/abs/2601.04426))

## Compared to XGrammar

**What you learn here:**
- Byte-level PDA + dict CI token-mask cache keyed by `(rule, node)`
- Earley named baseline for mask equality
- Nested JSON CFG; samples `json.loads`

| | This repo | XGrammar / XGrammar-2 |
|---|---|---|
| Engine | PDA + Python dict CI cache | Production PDA + adaptive JIT |
| Baseline | Earley mask audit | Earley in XGrammar-2 |
| Vocab | Tiny char/piece vocab | Full tokenizer vocab |
| Output | `json.loads` on samples | Serving-time constrained decode |

### Numbers (2026-08-16, Apple M5, darwin arm64 CPU)

| Metric | This repo | Baseline | Source |
|---|---|---|---|
| Samples `complete=True` | 8/8 | Constrained JSON valid | `main.py` |
| JIT compiled nodes after `{` | 9 | CI mask cache amortization | Dong et al. XGrammar |
| PDA ≡ Earley legal ids | True | Earley as correctness oracle | measured |
| `'true'` legal after `{` | False | Grammar rejects bool key | measured |

```bash
python main.py
```

## Run

```bash
python main.py
python -m pytest test_structured.py -q
```
