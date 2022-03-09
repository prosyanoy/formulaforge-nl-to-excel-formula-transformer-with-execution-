"""Synthetic spreadsheet tables.

A table lives on one sheet: header in row 1, data in rows 2..n_rows+1, columns A, B, C...
Tables are fully described by a small `TableSpec` (schema + size + constraints), so a record
only stores the spec and the table can be regenerated with *any* seed. That is what lets the
evaluator run predicted and gold formulas on fresh held-out tables.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --- column catalog -------------------------------------------------------------------------
CATEGORICAL = {
    "Region": ["East", "West", "North", "South"],
    "Product": ["Apples", "Bananas", "Cherries", "Grapes", "Mangoes"],
    "Department": ["Sales", "HR", "IT", "Finance", "Legal"],
    "Status": ["Open", "Closed", "Pending"],
    "Category": ["Food", "Toys", "Tools", "Books"],
    "Team": ["Red", "Blue", "Green", "Gold"],
    "Channel": ["Online", "Retail", "Partner"],
    "Grade": ["A", "B", "C"],
}
NUMERIC = {  # header -> (low, high, decimals)
    "Revenue": (100, 1000, 0),
    "Quantity": (1, 50, 0),
    "Price": (5, 200, 0),
    "Score": (0, 100, 0),
    "Salary": (30, 90, 0),
    "Age": (20, 60, 0),
    "Cost": (10, 500, 0),
    "Hours": (1, 40, 0),
    "Rating": (1, 5, 1),
    "Discount": (0, 30, 0),
}
KEYS = {
    "Employee": ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", "Irene", "Jack",
                 "Karen", "Liam", "Maria", "Noah", "Olivia", "Peter", "Quinn", "Rachel", "Steve", "Tina"],
    "Customer": ["Acme", "Globex", "Initech", "Umbrella", "Hooli", "Vandelay", "Stark", "Wayne", "Wonka",
                 "Cyberdyne", "Soylent", "Tyrell", "Oscorp", "Aperture", "Gringotts", "Dunder", "Pied", "Massive"],
    "Item": ["Widget", "Gadget", "Gizmo", "Doohickey", "Sprocket", "Cog", "Lever", "Valve", "Bolt", "Nut",
             "Washer", "Pulley", "Spring", "Hinge", "Bracket", "Clamp", "Rivet", "Gear"],
}

COL_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass
class Column:
    name: str
    kind: str  # "text" | "number" | "key"

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind}

    @property
    def domain(self) -> list[str]:
        if self.kind == "key":
            return KEYS[self.name]
        return CATEGORICAL[self.name]


@dataclass
class TableSpec:
    columns: list[Column]
    n_rows: int
    # column index -> values that must appear / must not appear in that column
    ensure: dict[int, list[str]] = field(default_factory=dict)
    absent: dict[int, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "columns": [c.to_dict() for c in self.columns],
            "n_rows": self.n_rows,
            "ensure": {str(k): v for k, v in self.ensure.items()},
            "absent": {str(k): v for k, v in self.absent.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TableSpec":
        return cls(
            [Column(**c) for c in d["columns"]],
            d["n_rows"],
            {int(k): v for k, v in d.get("ensure", {}).items()},
            {int(k): v for k, v in d.get("absent", {}).items()},
        )

    @property
    def first_row(self) -> int:
        return 2

    @property
    def last_row(self) -> int:
        return self.n_rows + 1

    def letter(self, idx: int) -> str:
        return COL_LETTERS[idx]

    def col_range(self, idx: int, r1: int | None = None, r2: int | None = None) -> str:
        L = self.letter(idx)
        return f"{L}{r1 or self.first_row}:{L}{r2 or self.last_row}"

    def block_range(self, i: int, j: int) -> str:
        return f"{self.letter(i)}{self.first_row}:{self.letter(j)}{self.last_row}"


def random_value(rng: random.Random, col: Column):
    if col.kind == "number":
        lo, hi, dec = NUMERIC[col.name]
        return round(rng.uniform(lo, hi), dec) if dec else rng.randint(lo, hi)
    return rng.choice(col.domain)


def generate_table(spec: TableSpec, seed: int) -> pd.DataFrame:
    """Instantiate `spec` with the given seed. Honors `ensure` / `absent` constraints."""
    rng = random.Random(seed)
    data: dict[str, list] = {}
    for idx, col in enumerate(spec.columns):
        absent = set(spec.absent.get(idx, []))
        if col.kind == "key":
            pool = [v for v in col.domain if v not in absent]
            forced = [v for v in spec.ensure.get(idx, []) if v not in absent]
            rest = [v for v in pool if v not in forced]
            rng.shuffle(rest)
            vals = (forced + rest)[: spec.n_rows]
            rng.shuffle(vals)
        elif col.kind == "text":
            pool = [v for v in col.domain if v not in absent]
            vals = [rng.choice(pool) for _ in range(spec.n_rows)]
            forced = [v for v in spec.ensure.get(idx, []) if v not in absent]
            slots = rng.sample(range(spec.n_rows), min(len(forced), spec.n_rows))
            for s, v in zip(slots, forced):
                vals[s] = v
        else:
            vals = [random_value(rng, col) for _ in range(spec.n_rows)]
        data[col.name] = vals
    df = pd.DataFrame(data)
    df.index = range(2, spec.n_rows + 2)  # Excel row numbers
    return df


# --- range resolution for the evaluator -------------------------------------------------------
_CELL = re.compile(r"^\$?([A-Z]+)\$?(\d+)$")


def _parse_cell(ref: str) -> tuple[int, int]:
    m = _CELL.match(ref)
    if not m:
        raise ValueError(f"bad cell reference {ref!r}")
    col = 0
    for ch in m.group(1):
        col = col * 26 + (ord(ch) - 64)
    return int(m.group(2)), col  # row (1-based), col (1-based)


def resolve_ref(df: pd.DataFrame, ref: str):
    """Value(s) of a cell/range reference on the sheet laid out from `df` (header in row 1)."""
    ref = ref.replace("$", "").upper()
    if ":" in ref:
        a, b = ref.split(":")
    else:
        a = b = ref
    (r1, c1), (r2, c2) = _parse_cell(a), _parse_cell(b)
    n_rows = len(df)
    grid = np.empty((r2 - r1 + 1, c2 - c1 + 1), dtype=object)
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            if r == 1:
                v = df.columns[c - 1] if c <= df.shape[1] else None
            elif 2 <= r <= n_rows + 1 and c <= df.shape[1]:
                v = df.iat[r - 2, c - 1]
                if isinstance(v, np.generic):
                    v = v.item()
            else:
                v = None  # empty cell
            grid[r - r1, c - c1] = v if v is not None else formulas_empty()
    if ":" not in ref:
        return grid[0, 0]
    return grid


def formulas_empty():
    from schedula import EMPTY  # the `formulas` library's marker for an empty cell

    return EMPTY
