"""Tiny EBNF → CFG, plus incremental Earley as the named correctness baseline.

Default constrained decoding uses the PDA + token-mask cache in ``pda.py`` /
``decode.TokenMasker``. ``Earley`` here is the Outlines-style recognizer used to
audit PDA masks (Willard & Louf 2023). Not a regex.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Term:
    kind: str  # lit | cls
    val: str = ""
    ranges: tuple[tuple[int, int], ...] = ()
    neg: bool = False

    def match(self, ch: str) -> bool:
        if self.kind == "lit":
            return ch == self.val
        o = ord(ch)
        hit = any(a <= o <= b for a, b in self.ranges)
        return (not hit) if self.neg else hit


@dataclass(frozen=True)
class Item:
    rule: str
    alt: int
    dot: int
    origin: int


class Lexer:
    def __init__(self, text: str) -> None:
        self.s = text
        self.i = 0
        self.cur = self._lex()
        self.nxt = self._lex()

    def advance(self) -> None:
        self.cur = self.nxt
        self.nxt = self._lex()

    def _skip(self) -> None:
        s, i = self.s, self.i
        n = len(s)
        while i < n:
            if s[i] in " \t\r\n":
                i += 1
            elif s[i] == "#" or s[i : i + 2] == "//":
                while i < n and s[i] != "\n":
                    i += 1
            else:
                break
        self.i = i

    def _lex(self) -> tuple[str, object]:
        self._skip()
        if self.i >= len(self.s):
            return ("EOF", "")
        c = self.s[self.i]
        if c.isalpha() or c == "_":
            j = self.i
            while self.i < len(self.s) and (self.s[self.i].isalnum() or self.s[self.i] == "_"):
                self.i += 1
            return ("NAME", self.s[j : self.i])
        if c in "\"'":
            return ("STR", self._string())
        if c == "[":
            return ("CLASS", self._class())
        self.i += 1
        table = {"=": "EQ", "|": "BAR", "?": "Q", "*": "STAR", "+": "PLUS", "(": "LP", ")": "RP"}
        if c not in table:
            raise SyntaxError(f"unexpected {c!r} at {self.i}")
        return (table[c], c)

    def _string(self) -> str:
        q = self.s[self.i]
        self.i += 1
        out: list[str] = []
        while self.i < len(self.s) and self.s[self.i] != q:
            if self.s[self.i] == "\\" and self.i + 1 < len(self.s):
                nxt = self.s[self.i + 1]
                out.append({"n": "\n", "t": "\t", "\\": "\\", q: q}.get(nxt, nxt))
                self.i += 2
            else:
                out.append(self.s[self.i])
                self.i += 1
        if self.i >= len(self.s):
            raise SyntaxError("unterminated string")
        self.i += 1
        return "".join(out)

    def _class(self) -> Term:
        self.i += 1
        neg = False
        if self.i < len(self.s) and self.s[self.i] == "^":
            neg = True
            self.i += 1
        ranges: list[tuple[int, int]] = []
        s = self.s
        while self.i < len(s) and s[self.i] != "]":
            if s[self.i] == "\\" and self.i + 1 < len(s):
                o = ord(s[self.i + 1])
                ranges.append((o, o))
                self.i += 2
                continue
            a = s[self.i]
            if self.i + 2 < len(s) and s[self.i + 1] == "-" and s[self.i + 2] != "]":
                ranges.append((ord(a), ord(s[self.i + 2])))
                self.i += 3
            else:
                ranges.append((ord(a), ord(a)))
                self.i += 1
        if self.i >= len(s) or s[self.i] != "]":
            raise SyntaxError("unterminated class")
        self.i += 1
        return Term("cls", ranges=tuple(ranges), neg=neg)


class Grammar:
    def __init__(self, prods: dict[str, list[list[str | Term]]], start: str) -> None:
        self.prods = prods
        self.start = start
        self.nullable = _nullable(prods)

    def next_sym(self, item: Item) -> str | Term | None:
        prod = self.prods[item.rule][item.alt]
        if item.dot >= len(prod):
            return None
        return prod[item.dot]


def _nullable(prods: dict[str, list[list[str | Term]]]) -> set[str]:
    """Aycock–Horspool: NT is nullable iff some alt is all-nullable (empty alt counts)."""
    null: set[str] = set()
    changed = True
    while changed:
        changed = False
        for rule, alts in prods.items():
            if rule in null:
                continue
            for alt in alts:
                if all(isinstance(s, str) and s in null for s in alt):
                    null.add(rule)
                    changed = True
                    break
    return null


def parse_ebnf(text: str, start: str | None = None) -> Grammar:
    lx = Lexer(text)
    prods: dict[str, list[list[str | Term]]] = {}
    nfresh = [0]

    def fresh(p: str = "_g") -> str:
        nfresh[0] += 1
        name = f"{p}{nfresh[0]}"
        return name

    def parse_expr() -> list[list[str | Term]]:
        alts = [parse_term()]
        while lx.cur[0] == "BAR":
            lx.advance()
            alts.append(parse_term())
        return alts

    def parse_term() -> list[str | Term]:
        out: list[str | Term] = []
        while True:
            k = lx.cur[0]
            if k == "NAME" and lx.nxt[0] == "EQ":
                break
            if k not in ("NAME", "STR", "CLASS", "LP"):
                break
            out.extend(parse_factor())
        return out

    def parse_atom() -> list[str | Term]:
        k, v = lx.cur
        if k == "STR":
            lx.advance()
            return [Term("lit", ch) for ch in str(v)]
        if k == "CLASS":
            lx.advance()
            return [v]  # type: ignore[list-item]
        if k == "NAME":
            lx.advance()
            return [str(v)]
        if k == "LP":
            lx.advance()
            alts = parse_expr()
            if lx.cur[0] != "RP":
                raise SyntaxError("expected )")
            lx.advance()
            name = fresh()
            prods[name] = alts
            return [name]
        raise SyntaxError(f"expected atom, got {k}")

    def parse_factor() -> list[str | Term]:
        seq = parse_atom()
        q = lx.cur[0] if lx.cur[0] in ("Q", "STAR", "PLUS") else None
        if q:
            lx.advance()
        if q is None:
            return seq
        if len(seq) == 1:
            inner: str | Term = seq[0]
        else:
            nm = fresh("_seq")
            prods[nm] = [seq]
            inner = nm
        if q == "Q":
            name = fresh("_opt")
            prods[name] = [[inner], []]
            return [name]
        if q == "STAR":
            name = fresh("_star")
            prods[name] = [[inner, name], []]
            return [name]
        star = fresh("_star")
        prods[star] = [[inner, star], []]
        plus = fresh("_plus")
        prods[plus] = [[inner, star]]
        return [plus]

    first: str | None = None
    while lx.cur[0] != "EOF":
        if lx.cur[0] != "NAME":
            raise SyntaxError(f"expected rule name, got {lx.cur}")
        name = str(lx.cur[1])
        lx.advance()
        if lx.cur[0] != "EQ":
            raise SyntaxError("expected =")
        lx.advance()
        prods[name] = parse_expr()
        if first is None:
            first = name
    root = start or first
    if not root:
        raise SyntaxError("empty grammar")
    prods["_start"] = [[root]]
    return Grammar(prods, "_start")


class Earley:
    """Incremental character-level Earley recognizer (valid prefix + complete)."""

    def __init__(self, grammar: Grammar) -> None:
        self.g = grammar
        self.chart: list[set[Item]] = [set()]
        for ai in range(len(self.g.prods[self.g.start])):
            self.chart[0].add(Item(self.g.start, ai, 0, 0))
        self._close(0)

    def clone(self) -> Earley:
        other = Earley.__new__(Earley)
        other.g = self.g
        other.chart = [set(col) for col in self.chart]
        return other

    def rollback(self, n_cols: int) -> None:
        del self.chart[n_cols:]

    @property
    def n_cols(self) -> int:
        return len(self.chart)

    def _close(self, j: int) -> None:
        col = self.chart[j]
        pending = list(col)
        seen = set(col)
        while pending:
            item = pending.pop()
            nxt = self.g.next_sym(item)
            if nxt is None:
                for prev in list(self.chart[item.origin]):
                    if self.g.next_sym(prev) == item.rule:
                        adv = Item(prev.rule, prev.alt, prev.dot + 1, prev.origin)
                        if adv not in seen:
                            seen.add(adv)
                            col.add(adv)
                            pending.append(adv)
            elif isinstance(nxt, str):
                for ai in range(len(self.g.prods[nxt])):
                    ni = Item(nxt, ai, 0, j)
                    if ni not in seen:
                        seen.add(ni)
                        col.add(ni)
                        pending.append(ni)
                if nxt in self.g.nullable:
                    adv = Item(item.rule, item.alt, item.dot + 1, item.origin)
                    if adv not in seen:
                        seen.add(adv)
                        col.add(adv)
                        pending.append(adv)

    def feed(self, s: str) -> bool:
        mark = len(self.chart)
        for ch in s:
            nxt_col: set[Item] = set()
            j = len(self.chart) - 1
            for item in self.chart[j]:
                nxt = self.g.next_sym(item)
                if isinstance(nxt, Term) and nxt.match(ch):
                    nxt_col.add(Item(item.rule, item.alt, item.dot + 1, item.origin))
            self.chart.append(nxt_col)
            self._close(len(self.chart) - 1)
            if not self.chart[-1]:
                del self.chart[mark:]
                return False
        return True

    def is_complete(self) -> bool:
        for item in self.chart[-1]:
            if item.rule == self.g.start and item.origin == 0 and self.g.next_sym(item) is None:
                return True
        return False

    def is_alive(self) -> bool:
        return bool(self.chart[-1])


def accepts(grammar: Grammar, s: str) -> bool:
    p = Earley(grammar)
    return p.feed(s) and p.is_complete()


def valid_prefix(grammar: Grammar, s: str) -> bool:
    return Earley(grammar).feed(s)


# JSON subset: objects/arrays nest; numbers follow RFC 8259 (no leading zeros).
JSON_EBNF = r"""
json = value
value = object | array | string | number | boolean | null
object = "{" ws members_opt ws "}"
members_opt = members |
members = pair members_tail
members_tail = ws "," ws pair members_tail |
pair = string ws ":" ws value
array = "[" ws elements_opt ws "]"
elements_opt = elements |
elements = value elements_tail
elements_tail = ws "," ws value elements_tail |
string = '"' chars '"'
chars = char chars |
char = [a-zA-Z0-9_ ]
number = int frac_opt
int = "-" uint | uint
uint = "0" | onenine digits
onenine = [1-9]
digits = [0-9] digits |
frac_opt = "." [0-9] digits |
boolean = "true" | "false"
null = "null"
ws = " " ws |
"""
