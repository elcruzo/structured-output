# 25 — CFG Constrained Decoding

Pushdown / Earley masker for structured generation (Outlines, XGrammar). Not a regex.

1. Parse a tiny EBNF subset (literals, char classes, concat, `|`, `?`/`*`/`+`, named rules) into a CFG.
2. Incremental **Earley** over characters: a string is a valid prefix iff the last chart column is non-empty; it is complete iff the augmented start item is finished.
3. Token vocabulary is an arbitrary list of strings (toy BPE). A token is legal iff feeding every one of its characters keeps the Earley chart alive — so a multi-char piece like `true` or `:true` is allowed only in contexts where that whole span is in the language.
4. `LogitsProcessor` sets illegal ids to `-inf` before sampling.

The built-in grammar is a nested JSON subset (objects, arrays, strings, RFC-8259 numbers, bool, null). Generated strings `json.loads`.

## Papers

- Willard & Louf, 2023. *Efficient Guided Generation for Large Language Models* (Outlines).
- Dong et al., 2024/2025. *XGrammar: Flexible and Efficient Structured Generation Engine*.

## Run

```bash
python demo.py
python -m pytest test_structured.py -q
```
