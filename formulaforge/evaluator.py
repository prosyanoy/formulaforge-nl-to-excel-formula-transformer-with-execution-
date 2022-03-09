"""Execution-based grading: run formulas on tables and compare *results*, not strings."""
from __future__ import annotations

import functools
import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .tables import TableSpec, generate_table, resolve_ref

ERROR = "#ERROR"  # sentinel for formulas that fail to parse/compile/evaluate


@functools.lru_cache(maxsize=4096)
def _compile(formula: str):
    import formulas

    f = formula if formula.startswith("=") else "=" + formula
    return formulas.Parser().ast(f)[1].compile()


def normalize_formula(formula: str) -> str:
    """Canonical string for exact-match comparison: no '=', no whitespace outside strings."""
    f = formula.strip()
    if f.startswith("="):
        f = f[1:]
    parts = re.split(r'("[^"]*")', f)
    return "".join(p if p.startswith('"') else re.sub(r"\s+", "", p).upper() for p in parts)


def _unwrap(res):
    """Reduce a `formulas` result to a python scalar (or a tuple of scalars for arrays)."""
    if isinstance(res, np.ndarray):
        if res.size == 1:
            return _unwrap(res.reshape(-1)[0])
        return tuple(_unwrap(x) for x in res.reshape(-1))
    if isinstance(res, np.generic):
        res = res.item()
    if isinstance(res, bool):
        return res
    if isinstance(res, (int, float)):
        return float(res)
    if isinstance(res, str):
        return res
    text = str(res)
    return text if text.startswith("#") else (text if text else "")


def evaluate_formula(formula: str, df: pd.DataFrame):
    """Evaluate `formula` against table `df`; returns scalar/tuple or ERROR (never raises)."""
    try:
        fn = _compile(formula)
        args = [resolve_ref(df, ref) for ref in fn.inputs]
        out = _unwrap(fn(*args))
    except Exception:  # unknown function, syntax error, bad reference, ...
        return ERROR
    if isinstance(out, str) and out.startswith("#"):
        return ERROR  # Excel error value (#N/A, #DIV/0!, ...): never counts as a correct answer
    if out == "" or (isinstance(out, float) and math.isnan(out)):
        return ERROR
    return out


def results_equal(a, b, tol: float = 1e-6) -> bool:
    if a == ERROR or b == ERROR:
        return False
    if isinstance(a, tuple) or isinstance(b, tuple):
        return isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b) and all(
            results_equal(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b or (isinstance(a, bool) and isinstance(b, bool) and a == b)
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=tol, abs_tol=tol)
    if isinstance(a, str) and isinstance(b, str):
        return a.casefold() == b.casefold()  # Excel string comparison is case-insensitive
    return False


@dataclass
class ExecResult:
    exact_match: bool
    exec_match: bool
    pred_valid: bool  # evaluates without error on every held-out table
    gold_results: list
    pred_results: list


def held_out_tables(spec: TableSpec, base_seed: int, n_tables: int) -> list[pd.DataFrame]:
    return [generate_table(spec, base_seed * 1000 + i + 17) for i in range(n_tables)]


def grade(pred: str, gold: str, spec: TableSpec, base_seed: int = 0, n_tables: int = 3) -> ExecResult:
    """Execution accuracy requires the prediction to agree with gold on *every* held-out table.

    Tables on which the gold formula itself is undefined (e.g. #DIV/0! because no row matched a
    condition) are skipped; the dataset builder rejects such examples, so this is only a safeguard.
    """
    tables = held_out_tables(spec, base_seed, n_tables)
    gold_all = [evaluate_formula(gold, t) for t in tables]
    keep = [i for i, g in enumerate(gold_all) if g != ERROR]
    gold_r = [gold_all[i] for i in keep]
    pred_r = [evaluate_formula(pred, tables[i]) for i in keep]
    return ExecResult(
        exact_match=normalize_formula(pred) == normalize_formula(gold),
        exec_match=bool(keep) and all(results_equal(p, g) for p, g in zip(pred_r, gold_r)),
        pred_valid=bool(keep) and all(p != ERROR for p in pred_r),
        gold_results=gold_r,
        pred_results=pred_r,
    )
