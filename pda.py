"""Byte-level pushdown automaton + token-mask cache (XGrammar-shaped).

Dong et al. 2024: CFG → per-rule FSAs with character edges and rule-reference
edges; stack manages recursion. Tokens at each stack-top node are partitioned into
context-independent accepted / rejected vs context-dependent (need a parent pop).
Context-independent masks are compiled per `(rule, node)` on first use and stored
in a Python dict as accept-heavy / reject-heavy / bitset id lists.

Earley lives in ``grammar.py`` as the named mask oracle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from grammar import Grammar, Term


@dataclass(frozen=True)
class CharEdge:
    term: Term
    to: int


@dataclass(frozen=True)
class RuleEdge:
    rule: str
    to: int


@dataclass
class RuleFSA:
    rule: str
    n_nodes: int
    start: int
    accepting: set[int]
    char_edges: dict[int, list[CharEdge]]
    rule_edges: dict[int, list[RuleEdge]]


@dataclass(frozen=True)
class Frame:
    rule: str
    node: int


Stack = tuple[Frame, ...]


@dataclass(frozen=True)
class NodeMask:
    """Adaptive token-mask cache entry for one PDA node (XGrammar §3.1)."""

    kind: str  # accept_heavy | reject_heavy | bitset
    accepted: tuple[int, ...]
    rejected: tuple[int, ...]
    dependent: tuple[int, ...]
    vocab_size: int

    def context_independent_accepted(self) -> set[int]:
        dep = set(self.dependent)
        if self.kind == "reject_heavy":
            return set(self.accepted)
        if self.kind == "accept_heavy":
            return set(range(self.vocab_size)) - set(self.rejected) - dep
        # bitset: accepted lists CI-accepted ids
        return set(self.accepted)


def build_rule_fsa(grammar: Grammar, rule: str) -> RuleFSA:
    """Compile one CFG rule into an FSA (character + rule-reference edges)."""
    char_edges: dict[int, list[CharEdge]] = {}
    rule_edges: dict[int, list[RuleEdge]] = {}
    accepting: set[int] = set()
    n_nodes = 1  # node 0 = start

    def fresh() -> int:
        nonlocal n_nodes
        n = n_nodes
        n_nodes += 1
        return n

    def add_char(frm: int, term: Term, to: int) -> None:
        char_edges.setdefault(frm, []).append(CharEdge(term, to))

    def add_rule(frm: int, ref: str, to: int) -> None:
        rule_edges.setdefault(frm, []).append(RuleEdge(ref, to))

    for alt in grammar.prods[rule]:
        cur = 0
        if not alt:
            accepting.add(0)
            continue
        for i, sym in enumerate(alt):
            nxt = fresh()
            if isinstance(sym, Term):
                add_char(cur, sym, nxt)
            else:
                add_rule(cur, str(sym), nxt)
            cur = nxt
            if i == len(alt) - 1:
                accepting.add(cur)

    return RuleFSA(
        rule=rule,
        n_nodes=n_nodes,
        start=0,
        accepting=accepting,
        char_edges=char_edges,
        rule_edges=rule_edges,
    )


class PDA:
    """Byte-level pushdown automaton over a CFG (one FSA per rule)."""

    def __init__(self, grammar: Grammar) -> None:
        self.g = grammar
        self.fsas: dict[str, RuleFSA] = {r: build_rule_fsa(grammar, r) for r in grammar.prods}

    def initial_stacks(self) -> set[Stack]:
        start = self.g.start
        return self._epsilon_close({(Frame(start, self.fsas[start].start),)})

    def _epsilon_close(self, stacks: set[Stack]) -> set[Stack]:
        """Pop completed frames; skip nullable rule-refs; cut left-recursive re-entry."""
        out: set[Stack] = set(stacks)
        pending = list(stacks)
        while pending:
            st = pending.pop()
            if not st:
                continue
            top = st[-1]
            fsa = self.fsas[top.rule]
            if top.node in fsa.accepting and len(st) > 1:
                popped = st[:-1]
                if popped not in out:
                    out.add(popped)
                    pending.append(popped)
            on_stack = {f.rule for f in st}
            for re in fsa.rule_edges.get(top.node, []):
                if re.rule in self.g.nullable:
                    advanced = st[:-1] + (Frame(top.rule, re.to),)
                    if advanced not in out:
                        out.add(advanced)
                        pending.append(advanced)
                if re.rule in on_stack:
                    continue
                entered = st[:-1] + (Frame(top.rule, re.to), Frame(re.rule, self.fsas[re.rule].start))
                if entered not in out:
                    out.add(entered)
                    pending.append(entered)
        return out

    def step(self, stacks: set[Stack], ch: str, *, permit_base_pop: bool = True) -> tuple[set[Stack], bool]:
        """Consume one character. Returns (new_stacks, used_base_pop)."""
        stacks = self._epsilon_close(stacks)
        nxt: set[Stack] = set()
        used_base_pop = False
        base_depth = 1

        def consume_from(st: Stack, entered: frozenset[str]) -> None:
            nonlocal used_base_pop
            if not st:
                return
            top = st[-1]
            fsa = self.fsas[top.rule]
            for ce in fsa.char_edges.get(top.node, []):
                if ce.term.match(ch):
                    advanced = st[:-1] + (Frame(top.rule, ce.to),)
                    nxt.update(self._epsilon_close({advanced}))
            for re in fsa.rule_edges.get(top.node, []):
                if re.rule in entered:
                    continue
                child = st[:-1] + (Frame(top.rule, re.to), Frame(re.rule, self.fsas[re.rule].start))
                consume_from(child, entered | {re.rule})
            if top.node in fsa.accepting and len(st) > 1:
                if len(st) <= base_depth:
                    used_base_pop = True
                    if not permit_base_pop:
                        return
                consume_from(st[:-1], entered)

        for st in stacks:
            consume_from(st, frozenset())
        return nxt, used_base_pop

    def feed(self, stacks: set[Stack], s: str, *, permit_base_pop: bool = True) -> tuple[set[Stack], bool]:
        used_pop = False
        cur = stacks
        for ch in s:
            cur, pop = self.step(cur, ch, permit_base_pop=permit_base_pop)
            used_pop = used_pop or pop
            if not cur:
                return set(), used_pop
        return self._epsilon_close(cur), used_pop

    def is_alive(self, stacks: set[Stack]) -> bool:
        return bool(self._epsilon_close(stacks))

    def is_complete(self, stacks: set[Stack]) -> bool:
        stacks = self._epsilon_close(stacks)
        start = self.g.start
        fsa = self.fsas[start]
        for st in stacks:
            if len(st) == 1 and st[0].rule == start and st[0].node in fsa.accepting:
                return True
        return False


def classify_token(pda: PDA, rule: str, node: int, token: str) -> str:
    """Return 'accepted' | 'rejected' | 'dependent' for token at stack-top (rule, node)."""
    if not token:
        return "rejected"
    base: Stack = (Frame(rule, node),)
    ok_local, _ = pda.feed({base}, token, permit_base_pop=False)
    if ok_local:
        return "accepted"
    ok_pop, used_pop = pda.feed({base}, token, permit_base_pop=True)
    if ok_pop and used_pop:
        return "dependent"
    if used_pop and not ok_local:
        return "dependent"
    return "rejected"


def _adaptive_node_mask(accepted: list[int], rejected: list[int], dependent: list[int], vocab_size: int) -> NodeMask:
    dep = tuple(dependent)
    n_acc, n_rej = len(accepted), len(rejected)
    # Cost ≈ bytes for the two smaller CI lists (+ dependent always stored).
    cost_accept_heavy = n_rej  # store rejected + dependent
    cost_reject_heavy = n_acc
    cost_bitset = n_acc + n_rej
    best = min(cost_accept_heavy, cost_reject_heavy, cost_bitset)
    if best == cost_accept_heavy and n_rej <= n_acc:
        return NodeMask("accept_heavy", (), tuple(rejected), dep, vocab_size)
    if best == cost_reject_heavy:
        return NodeMask("reject_heavy", tuple(accepted), (), dep, vocab_size)
    return NodeMask("bitset", tuple(accepted), tuple(rejected), dep, vocab_size)


@dataclass
class TokenMaskCache:
    """Token-mask cache keyed by stack-top (rule, node)."""

    pda: PDA
    vocab: list[str]
    _nodes: dict[tuple[str, int], NodeMask] = field(default_factory=dict)

    def compile_node(self, rule: str, node: int) -> NodeMask:
        key = (rule, node)
        hit = self._nodes.get(key)
        if hit is not None:
            return hit
        accepted: list[int] = []
        rejected: list[int] = []
        dependent: list[int] = []
        for tid, tok in enumerate(self.vocab):
            if not tok:
                rejected.append(tid)
                continue
            kind = classify_token(self.pda, rule, node, tok)
            if kind == "accepted":
                accepted.append(tid)
            elif kind == "rejected":
                rejected.append(tid)
            else:
                dependent.append(tid)
        mask = _adaptive_node_mask(accepted, rejected, dependent, len(self.vocab))
        self._nodes[key] = mask
        return mask

    def legal_ids(self, stacks: set[Stack]) -> list[int]:
        stacks = self.pda._epsilon_close(stacks)
        if not stacks:
            return []
        legal: set[int] = set()
        for st in stacks:
            top = st[-1]
            entry = self.compile_node(top.rule, top.node)
            legal |= entry.context_independent_accepted()
            for tid in entry.dependent:
                tok = self.vocab[tid]
                nxt, _ = self.pda.feed({st}, tok, permit_base_pop=True)
                if nxt:
                    legal.add(tid)
        return sorted(legal)


class PDAMatcher:
    """Incremental PDA recognizer with rollback via stack-set snapshots."""

    def __init__(self, pda: PDA) -> None:
        self.pda = pda
        self.stacks = pda.initial_stacks()
        self._history: list[set[Stack]] = []

    def clone(self) -> PDAMatcher:
        other = PDAMatcher.__new__(PDAMatcher)
        other.pda = self.pda
        other.stacks = set(self.stacks)
        other._history = [set(s) for s in self._history]
        return other

    def mark(self) -> int:
        """Save current stacks; return a mark for ``rollback``."""
        self._history.append(set(self.stacks))
        return len(self._history)

    def rollback(self, mark: int) -> None:
        """Restore stacks to the checkpoint created by ``mark()`` (undo feeds since then)."""
        while len(self._history) > mark:
            self._history.pop()
        if mark == 0:
            self.stacks = self.pda.initial_stacks()
            self._history.clear()
            return
        self.stacks = set(self._history[mark - 1])
        del self._history[mark:]

    def feed(self, s: str) -> bool:
        if not s:
            return self.pda.is_alive(self.stacks)
        prev = set(self.stacks)
        nxt, _ = self.pda.feed(self.stacks, s, permit_base_pop=True)
        if not nxt:
            self.stacks = prev
            return False
        self.stacks = nxt
        return True

    def is_complete(self) -> bool:
        return self.pda.is_complete(self.stacks)

    def is_alive(self) -> bool:
        return self.pda.is_alive(self.stacks)


def accepts_pda(grammar: Grammar, s: str) -> bool:
    pda = PDA(grammar)
    stacks, _ = pda.feed(pda.initial_stacks(), s)
    return pda.is_complete(stacks)


def valid_prefix_pda(grammar: Grammar, s: str) -> bool:
    pda = PDA(grammar)
    if not s:
        return True
    stacks, _ = pda.feed(pda.initial_stacks(), s)
    return bool(stacks)
