"""Formula templates: each produces (NL skeletons, formula, table spec) for one function family.

Families: SUM, AVERAGE, MINMAX, COUNTIF, IF, SUMIFS, VLOOKUP, INDEX_MATCH.
Tiers: 1 = single function over one range, 2 = one condition / lookup / partial range,
       3 = multiple conditions, nesting, error handling or multi-column logic.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from .paraphrase import fill, paraphrase
from .tables import CATEGORICAL, KEYS, NUMERIC, Column, TableSpec

OPS = {
    ">": ["greater than", "over", "above", "more than", "higher than"],
    "<": ["less than", "under", "below", "lower than", "smaller than"],
    ">=": ["at least", "no less than", "greater than or equal to"],
    "<=": ["at most", "no more than", "less than or equal to"],
}
LABELS = [("Yes", "No"), ("High", "Low"), ("Pass", "Fail"), ("Good", "Bad"), ("Approved", "Rejected")]


# --- table construction ---------------------------------------------------------------------
class Ctx:
    """Random table under construction plus helpers shared by templates."""

    def __init__(self, rng: random.Random, need_key=False, n_text=0, n_num=0, key_not_last=False):
        self.rng = rng
        cols: list[Column] = []
        if need_key:
            cols.append(Column(rng.choice(list(KEYS)), "key"))
        for name in rng.sample(list(CATEGORICAL), n_text):
            cols.append(Column(name, "text"))
        for name in rng.sample(list(NUMERIC), n_num):
            cols.append(Column(name, "number"))
        # a distractor column or two, so schemas are not minimal
        for _ in range(rng.choice([0, 0, 1, 2])):
            if len(cols) >= 5:
                break
            if rng.random() < 0.5:
                free = [n for n in CATEGORICAL if all(c.name != n for c in cols)]
                cols.append(Column(rng.choice(free), "text"))
            else:
                free = [n for n in NUMERIC if all(c.name != n for c in cols)]
                cols.append(Column(rng.choice(free), "number"))
        rng.shuffle(cols)
        if key_not_last:  # VLOOKUP needs a column to the right of the key
            ki = next(i for i, c in enumerate(cols) if c.kind == "key")
            if ki == len(cols) - 1:
                cols[0], cols[ki] = cols[ki], cols[0]
        self.spec = TableSpec(cols, rng.randint(5, 14))

    def idx(self, kind: str, exclude=()):
        return self.rng.choice([i for i, c in enumerate(self.spec.columns) if c.kind == kind and i not in exclude])

    def R(self, i: int) -> str:
        return self.spec.col_range(i)

    def name(self, i: int) -> str:
        n = self.spec.columns[i].name
        return n.lower() if self.rng.random() < 0.15 else n

    def value(self, i: int, ensure=True) -> str:
        v = self.rng.choice(self.spec.columns[i].domain)
        if ensure:
            self.spec.ensure.setdefault(i, []).append(v)
        return v

    def threshold(self, i: int) -> str:
        lo, hi, dec = NUMERIC[self.spec.columns[i].name]
        if hi - lo <= 5:
            return str(self.rng.randint(lo + 1, hi - 1))
        step = 10 if hi - lo >= 100 else 5
        return str(int(round(self.rng.uniform(lo + 0.2 * (hi - lo), hi - 0.2 * (hi - lo)) / step) * step))

    def op(self):
        sym = self.rng.choice(list(OPS))
        return sym, self.rng.choice(OPS[sym])

    def row(self) -> int:
        return self.rng.randint(self.spec.first_row, self.spec.last_row)


@dataclass
class Built:
    skeletons: list[str]
    formula: str
    ctx: Ctx
    fields: dict


def _pick(rng, options):
    return rng.choice(options)


# --- SUM --------------------------------------------------------------------------------------
def sum_col(rng):
    c = Ctx(rng, n_num=1)
    i = c.idx("number")
    return Built(["Add up all the {c}", "What is the total {c}?", "Sum the {c} column", "Calculate the total of {c}",
                  "Total {c} for all rows", "Sum of {c}"], f"SUM({c.R(i)})", c, {"c": c.name(i)})


def sum_partial(rng):
    c = Ctx(rng, n_num=1)
    i = c.idx("number")
    a = rng.randint(2, c.spec.last_row - 2)
    b = rng.randint(a + 1, c.spec.last_row)
    return Built(["Sum the {c} values from row {a} to row {b}", "Calculate the total {c} between rows {a} and {b}",
                  "Add up {c} in rows {a} through {b}", "What is the sum of {c} for rows {a} to {b}?"],
                 f"SUM({c.spec.col_range(i, a, b)})", c, {"c": c.name(i), "a": a, "b": b})


def sum_two_cols(rng):
    c = Ctx(rng, n_num=2)
    i = c.idx("number")
    j = c.idx("number", (i,))
    return Built(["Add the total {c} and the total {d}", "Calculate the sum of {c} plus the sum of {d}",
                  "What is the combined total of {c} and {d}?", "Total of {c} and {d} together"],
                 f"SUM({c.R(i)})+SUM({c.R(j)})", c, {"c": c.name(i), "d": c.name(j)})


def sumproduct(rng):
    c = Ctx(rng, n_num=2)
    i = c.idx("number")
    j = c.idx("number", (i,))
    return Built(["Calculate the total of {c} multiplied by {d} for each row", "Multiply {c} by {d} row by row and add it all up",
                  "What is the sum of {c} times {d}?", "Sum product of {c} and {d}"],
                 f"SUMPRODUCT({c.R(i)},{c.R(j)})", c, {"c": c.name(i), "d": c.name(j)})


# --- AVERAGE ----------------------------------------------------------------------------------
def average_col(rng):
    c = Ctx(rng, n_num=1)
    i = c.idx("number")
    return Built(["What is the average {c}?", "Calculate the mean of {c}", "Find the average of the {c} column",
                  "Average {c} across all rows", "Mean {c}"], f"AVERAGE({c.R(i)})", c, {"c": c.name(i)})


def averageif(rng):
    c = Ctx(rng, n_text=1, n_num=1)
    t, n = c.idx("text"), c.idx("number")
    v = c.value(t)
    f = f'AVERAGEIF({c.R(t)},"{v}",{c.R(n)})'
    return Built(["What is the average {n} where {t} is {v}?", "Find the mean {n} for rows with {t} equal to {v}",
                  "Average {n} for {v} {t}", "Calculate the average of {n} where the {t} is {v}"], f, c,
                 {"n": c.name(n), "t": c.name(t), "v": v})


def averageifs(rng):
    c = Ctx(rng, n_text=1, n_num=2)
    t, n = c.idx("text"), c.idx("number")
    m = c.idx("number", (n,))
    v = c.value(t)
    sym, words = c.op()
    th = c.threshold(m)
    f = f'AVERAGEIFS({c.R(n)},{c.R(t)},"{v}",{c.R(m)},"{sym}{th}")'
    return Built(["Average {n} where {t} is {v} and {m} is {op} {th}", "Find the mean {n} for {v} {t} with {m} {op} {th}",
                  "What is the average {n} for rows where {t} equals {v} and {m} is {op} {th}?"], f, c,
                 {"n": c.name(n), "t": c.name(t), "m": c.name(m), "v": v, "op": words, "th": th})


# --- MIN / MAX --------------------------------------------------------------------------------
def minmax_col(rng):
    c = Ctx(rng, n_num=1)
    i = c.idx("number")
    if rng.random() < 0.5:
        return Built(["What is the highest {c}?", "Find the maximum {c}", "Largest value in {c}", "Get the biggest {c}"],
                     f"MAX({c.R(i)})", c, {"c": c.name(i)})
    return Built(["What is the lowest {c}?", "Find the minimum {c}", "Smallest value in {c}", "Get the smallest {c}"],
                 f"MIN({c.R(i)})", c, {"c": c.name(i)})


def minmax_spread(rng):
    c = Ctx(rng, n_num=1)
    i = c.idx("number")
    return Built(["What is the difference between the highest and lowest {c}?", "Calculate the spread of {c}, max minus min",
                  "Find the range of {c} values"], f"MAX({c.R(i)})-MIN({c.R(i)})", c, {"c": c.name(i)})


def minmax_if(rng):
    c = Ctx(rng, n_text=1, n_num=1)
    t, n = c.idx("text"), c.idx("number")
    v = c.value(t)
    if rng.random() < 0.5:
        return Built(["What is the highest {n} where {t} is {v}?", "Find the maximum {n} for {v} {t}",
                      "Largest {n} among rows with {t} equal to {v}"], f'MAXIFS({c.R(n)},{c.R(t)},"{v}")', c,
                     {"n": c.name(n), "t": c.name(t), "v": v})
    return Built(["What is the lowest {n} where {t} is {v}?", "Find the minimum {n} for {v} {t}",
                  "Smallest {n} among rows with {t} equal to {v}"], f'MINIFS({c.R(n)},{c.R(t)},"{v}")', c,
                 {"n": c.name(n), "t": c.name(t), "v": v})


# --- COUNTIF ----------------------------------------------------------------------------------
def countif_text(rng):
    c = Ctx(rng, n_text=1)
    t = c.idx("text")
    v = c.value(t)
    return Built(["How many rows have {t} equal to {v}?", "Count the {v} entries in {t}", "Count rows where {t} is {v}",
                  "Number of times {v} appears in {t}"], f'COUNTIF({c.R(t)},"{v}")', c, {"t": c.name(t), "v": v})


def countif_num(rng):
    c = Ctx(rng, n_num=1)
    n = c.idx("number")
    sym, words = c.op()
    th = c.threshold(n)
    return Built(["How many rows have {n} {op} {th}?", "Count the entries where {n} is {op} {th}", "Count {n} values {op} {th}"],
                 f'COUNTIF({c.R(n)},"{sym}{th}")', c, {"n": c.name(n), "op": words, "th": th})


def countifs(rng):
    c = Ctx(rng, n_text=1, n_num=1)
    t, n = c.idx("text"), c.idx("number")
    v = c.value(t)
    sym, words = c.op()
    th = c.threshold(n)
    return Built(["How many rows have {t} equal to {v} and {n} {op} {th}?", "Count rows where {t} is {v} and {n} is {op} {th}",
                  "Number of {v} {t} entries with {n} {op} {th}"], f'COUNTIFS({c.R(t)},"{v}",{c.R(n)},"{sym}{th}")', c,
                 {"t": c.name(t), "n": c.name(n), "v": v, "op": words, "th": th})


# --- IF ---------------------------------------------------------------------------------------
def if_simple(rng):
    c = Ctx(rng, n_num=1)
    n = c.idx("number")
    r = c.row()
    sym, words = c.op()
    th = c.threshold(n)
    y, no = rng.choice(LABELS)
    return Built(['If {n} in row {r} is {op} {th} return "{y}", otherwise "{no}"',
                  'Check whether {n} in row {r} is {op} {th}; show {y} or {no}',
                  'Return {y} if the {n} of row {r} is {op} {th}, else {no}'],
                 f'IF({c.spec.letter(n)}{r}{sym}{th},"{y}","{no}")', c,
                 {"n": c.name(n), "r": r, "op": words, "th": th, "y": y, "no": no})


def if_text(rng):
    c = Ctx(rng, n_text=1, n_num=1)
    t, n = c.idx("text"), c.idx("number")
    r = c.row()
    v = c.value(t)
    pct = rng.choice([5, 10, 20])
    mult = 1 + pct / 100
    f = f'IF({c.spec.letter(t)}{r}="{v}",{c.spec.letter(n)}{r}*{mult:g},{c.spec.letter(n)}{r})'
    return Built(["If {t} in row {r} is {v}, increase {n} by {p} percent, otherwise keep {n}",
                  "For row {r}, give {n} plus {p}% when {t} is {v}, otherwise just {n}"], f, c,
                 {"t": c.name(t), "n": c.name(n), "r": r, "v": v, "p": pct})


def if_total(rng):
    c = Ctx(rng, n_num=1)
    n = c.idx("number")
    lo, hi, _ = NUMERIC[c.spec.columns[n].name]
    mean = (lo + hi) / 2 * c.spec.n_rows
    th = max(5, int(round(mean / 50) * 50)) if mean > 50 else int(mean)
    return Built(['If the total {n} is greater than {th} show "Target met", otherwise "Below target"',
                  'Show "Target met" when the sum of {n} is over {th}, else "Below target"'],
                 f'IF(SUM({c.R(n)})>{th},"Target met","Below target")', c, {"n": c.name(n), "th": th})


def if_and(rng):
    c = Ctx(rng, n_text=1, n_num=1)
    t, n = c.idx("text"), c.idx("number")
    r = c.row()
    v = c.value(t)
    sym, words = c.op()
    th = c.threshold(n)
    y, no = rng.choice(LABELS)
    return Built(['In row {r}, return "{y}" if {n} is {op} {th} and {t} is {v}, otherwise "{no}"',
                  'If both {n} is {op} {th} and {t} equals {v} in row {r} show {y}, else {no}'],
                 f'IF(AND({c.spec.letter(n)}{r}{sym}{th},{c.spec.letter(t)}{r}="{v}"),"{y}","{no}")', c,
                 {"n": c.name(n), "t": c.name(t), "r": r, "v": v, "op": words, "th": th, "y": y, "no": no})


def if_nested(rng):
    c = Ctx(rng, n_num=1)
    n = c.idx("number")
    r = c.row()
    lo, hi, _ = NUMERIC[c.spec.columns[n].name]
    if hi - lo <= 5:
        return if_and(rng)
    t1 = int(c.threshold(n))
    t2 = max(lo, t1 - rng.choice([10, 20, 30]) if hi - lo >= 50 else t1 - 5)
    return Built(['Grade row {r}: "A" if {n} is at least {t1}, "B" if at least {t2}, otherwise "C"',
                  'For row {r} return A when {n} >= {t1}, B when {n} >= {t2}, and C for anything lower'],
                 f'IF({c.spec.letter(n)}{r}>={t1},"A",IF({c.spec.letter(n)}{r}>={t2},"B","C"))', c,
                 {"n": c.name(n), "r": r, "t1": t1, "t2": t2})


# --- SUMIFS -----------------------------------------------------------------------------------
def sumif_text(rng):
    c = Ctx(rng, n_text=1, n_num=1)
    t, n = c.idx("text"), c.idx("number")
    v = c.value(t)
    f = f'SUMIF({c.R(t)},"{v}",{c.R(n)})' if rng.random() < 0.5 else f'SUMIFS({c.R(n)},{c.R(t)},"{v}")'
    return Built(["Total {n} where {t} is {v}", "Sum the {n} for all rows with {t} equal to {v}", "What is the total {n} for {v}?",
                  "Add up {n} only for {v} {t}"], f, c, {"n": c.name(n), "t": c.name(t), "v": v})


def sumif_num(rng):
    c = Ctx(rng, n_num=1)
    n = c.idx("number")
    sym, words = c.op()
    th = c.threshold(n)
    f = f'SUMIF({c.R(n)},"{sym}{th}")' if rng.random() < 0.5 else f'SUMIFS({c.R(n)},{c.R(n)},"{sym}{th}")'
    return Built(["Sum the {n} values that are {op} {th}", "Total of {n} where {n} is {op} {th}", "Add up only the {n} {op} {th}"],
                 f, c, {"n": c.name(n), "op": words, "th": th})


def sumifs_two_text(rng):
    c = Ctx(rng, n_text=2, n_num=1)
    t = c.idx("text")
    u = c.idx("text", (t,))
    n = c.idx("number")
    v, w = c.value(t), c.value(u)
    return Built(["Total {n} where {t} is {v} and {u} is {w}", "Sum {n} for rows with {t} equal to {v} and {u} equal to {w}",
                  "What is the total {n} for {v} {t} and {w} {u}?"], f'SUMIFS({c.R(n)},{c.R(t)},"{v}",{c.R(u)},"{w}")', c,
                 {"n": c.name(n), "t": c.name(t), "u": c.name(u), "v": v, "w": w})


def sumifs_text_num(rng):
    c = Ctx(rng, n_text=1, n_num=2)
    t, n = c.idx("text"), c.idx("number")
    m = c.idx("number", (n,))
    v = c.value(t)
    sym, words = c.op()
    th = c.threshold(m)
    return Built(["Total {n} where {t} is {v} and {m} is {op} {th}", "Sum {n} for {v} {t} rows whose {m} is {op} {th}",
                  "Add up {n} when {t} equals {v} and {m} is {op} {th}"],
                 f'SUMIFS({c.R(n)},{c.R(t)},"{v}",{c.R(m)},"{sym}{th}")', c,
                 {"n": c.name(n), "t": c.name(t), "m": c.name(m), "v": v, "op": words, "th": th})


def sumifs_between(rng):
    c = Ctx(rng, n_num=2)
    n = c.idx("number")
    m = c.idx("number", (n,))
    a = int(c.threshold(m))
    lo, hi, _ = NUMERIC[c.spec.columns[m].name]
    if hi - lo <= 5:
        return sumifs_text_num(rng)
    b = a + rng.choice([10, 20, 30, 50])
    return Built(["Total {n} where {m} is between {a} and {b} inclusive", "Sum {n} for rows with {m} from {a} to {b}",
                  "Add up {n} when {m} is at least {a} and at most {b}"],
                 f'SUMIFS({c.R(n)},{c.R(m)},">={a}",{c.R(m)},"<={b}")', c,
                 {"n": c.name(n), "m": c.name(m), "a": a, "b": b})


# --- VLOOKUP / INDEX-MATCH --------------------------------------------------------------------
def _lookup_ctx(rng, vlookup: bool):
    c = Ctx(rng, need_key=True, n_text=rng.choice([0, 1]), n_num=rng.choice([1, 2]), key_not_last=vlookup)
    k = c.idx("key")
    rets = [i for i, col in enumerate(c.spec.columns) if col.kind != "key" and (not vlookup or i > k)]
    return c, k, rng.choice(rets)


def vlookup_basic(rng):
    c, k, r = _lookup_ctx(rng, True)
    key = c.value(k)
    f = f'VLOOKUP("{key}",{c.spec.block_range(k, r)},{r - k + 1},FALSE)'
    return Built(["Look up {key} and return its {r}", "Find the {r} for {key}", "What is the {r} of {key}?",
                  "Use VLOOKUP to get {r} for {key} from the table"], f, c, {"key": key, "r": c.name(r)})


def vlookup_iferror(rng):
    c, k, r = _lookup_ctx(rng, True)
    key = rng.choice(c.spec.columns[k].domain)
    c.spec.absent.setdefault(k, []).append(key)  # make the fallback branch the one that is executed
    f = f'IFERROR(VLOOKUP("{key}",{c.spec.block_range(k, r)},{r - k + 1},FALSE),"Not found")'
    return Built(['Look up {key} and return its {r}, or "Not found" if it is missing',
                  'Find the {r} for {key}; show Not found when there is no match'], f, c, {"key": key, "r": c.name(r)})


def index_match(rng):
    c, k, r = _lookup_ctx(rng, False)
    key = c.value(k)
    f = f'INDEX({c.R(r)},MATCH("{key}",{c.R(k)},0))'
    return Built(["Look up {key} in the {k} column and return the matching {r}", "What is the {r} for {key}?",
                  "Use INDEX and MATCH to find the {r} of {key}", "Return {r} where {k} is {key}"], f, c,
                 {"key": key, "r": c.name(r), "k": c.name(k)})


def index_match_extreme(rng):
    c = Ctx(rng, need_key=True, n_num=1)
    k, n = c.idx("key"), c.idx("number")
    if rng.random() < 0.5:
        fn, word = "MAX", rng.choice(["highest", "largest", "greatest"])
    else:
        fn, word = "MIN", rng.choice(["lowest", "smallest", "least"])
    return Built(["Which {k} has the {w} {n}?", "Find the {k} with the {w} {n}", "Return the {k} that has the {w} {n}"],
                 f"INDEX({c.R(k)},MATCH({fn}({c.R(n)}),{c.R(n)},0))", c, {"k": c.name(k), "n": c.name(n), "w": word})


def index_match_iferror(rng):
    c, k, r = _lookup_ctx(rng, False)
    key = rng.choice(c.spec.columns[k].domain)
    c.spec.absent.setdefault(k, []).append(key)
    f = f'IFERROR(INDEX({c.R(r)},MATCH("{key}",{c.R(k)},0)),"Not found")'
    return Built(['Use INDEX and MATCH to find the {r} for {key}, showing "Not found" if it does not exist',
                  'Return the {r} where {k} is {key}, or Not found if there is no match'], f, c,
                 {"key": key, "r": c.name(r), "k": c.name(k)})


@dataclass
class Template:
    name: str
    family: str
    tier: int
    weight: float
    build: Callable


TEMPLATES = [
    Template("sum_col", "SUM", 1, 3, sum_col),
    Template("sum_partial", "SUM", 2, 2, sum_partial),
    Template("sum_two_cols", "SUM", 3, 1.5, sum_two_cols),
    Template("sumproduct", "SUM", 3, 1.5, sumproduct),
    Template("average_col", "AVERAGE", 1, 3, average_col),
    Template("averageif", "AVERAGE", 2, 2, averageif),
    Template("averageifs", "AVERAGE", 3, 1.5, averageifs),
    Template("minmax_col", "MINMAX", 1, 3, minmax_col),
    Template("minmax_spread", "MINMAX", 2, 1.5, minmax_spread),
    Template("minmax_if", "MINMAX", 3, 1.5, minmax_if),
    Template("countif_text", "COUNTIF", 1, 3, countif_text),
    Template("countif_num", "COUNTIF", 2, 2, countif_num),
    Template("countifs", "COUNTIF", 3, 2, countifs),
    Template("if_simple", "IF", 1, 3, if_simple),
    Template("if_text", "IF", 2, 2, if_text),
    Template("if_total", "IF", 2, 1.5, if_total),
    Template("if_and", "IF", 3, 2, if_and),
    Template("if_nested", "IF", 3, 2, if_nested),
    Template("sumif_text", "SUMIFS", 2, 3, sumif_text),
    Template("sumif_num", "SUMIFS", 2, 2, sumif_num),
    Template("sumifs_two_text", "SUMIFS", 3, 2, sumifs_two_text),
    Template("sumifs_text_num", "SUMIFS", 3, 2, sumifs_text_num),
    Template("sumifs_between", "SUMIFS", 3, 2, sumifs_between),
    Template("vlookup_basic", "VLOOKUP", 2, 3, vlookup_basic),
    Template("vlookup_iferror", "VLOOKUP", 3, 2, vlookup_iferror),
    Template("index_match", "INDEX_MATCH", 2, 3, index_match),
    Template("index_match_extreme", "INDEX_MATCH", 3, 2, index_match_extreme),
    Template("index_match_iferror", "INDEX_MATCH", 3, 2, index_match_iferror),
]
FAMILIES = sorted({t.family for t in TEMPLATES})


def sample_example(rng: random.Random, templates: list[Template] | None = None, paraphrase_level: float = 0.5) -> dict:
    templates = templates or TEMPLATES
    t = rng.choices(templates, weights=[x.weight for x in templates])[0]
    b = t.build(rng)
    skeleton = paraphrase(rng, rng.choice(b.skeletons), paraphrase_level)
    return {
        "nl": fill(skeleton, **b.fields),
        "formula": b.formula,
        "family": t.family,
        "tier": t.tier,
        "template": t.name,
        "spec": b.ctx.spec.to_dict(),
    }
