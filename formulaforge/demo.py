"""Shared logic for the CLI and Streamlit demos: sample tables, schema inference, answering."""
from __future__ import annotations

import pandas as pd

from .evaluator import evaluate_formula
from .tables import CATEGORICAL, KEYS, NUMERIC, Column, TableSpec, generate_table


def preset_table(name: str, seed: int = 7) -> pd.DataFrame:
    presets = {
        "sales": [Column("Region", "text"), Column("Product", "text"), Column("Revenue", "number"), Column("Quantity", "number")],
        "employees": [Column("Employee", "key"), Column("Department", "text"), Column("Salary", "number"), Column("Age", "number")],
        "orders": [Column("Customer", "key"), Column("Status", "text"), Column("Price", "number"), Column("Discount", "number")],
    }
    return generate_table(TableSpec(presets[name], 10), seed)


PRESETS = ["sales", "employees", "orders"]


def infer_spec(df: pd.DataFrame) -> TableSpec:
    """Column kinds for an arbitrary DataFrame: numeric -> number, unique strings -> key, else text."""
    cols = []
    for name in df.columns:
        s = df[name]
        if pd.api.types.is_numeric_dtype(s):
            kind = "number"
        elif s.nunique() == len(s):
            kind = "key"
        else:
            kind = "text"
        cols.append(Column(str(name), kind))
    return TableSpec(cols, len(df))


def answer(predictor, request: str, df: pd.DataFrame, num_beams: int = 4) -> dict:
    spec = infer_spec(df)
    formula = predictor.predict(request, spec, num_beams=num_beams)
    result = evaluate_formula(formula, df)
    return {"formula": "=" + formula, "result": result, "ok": result != "#ERROR", "spec": spec}


def format_table(df: pd.DataFrame) -> str:
    """Render with Excel-style column letters and row numbers."""
    view = df.copy()
    view.columns = [f"{chr(65 + i)}: {c}" for i, c in enumerate(df.columns)]
    view.index = range(2, len(df) + 2)
    return view.to_string()


assert CATEGORICAL and KEYS and NUMERIC  # catalogs are re-exported for UI code
