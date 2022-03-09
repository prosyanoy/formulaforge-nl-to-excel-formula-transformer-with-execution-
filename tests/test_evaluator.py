import pandas as pd
import pytest

from formulaforge.evaluator import ERROR, evaluate_formula, grade, normalize_formula, results_equal
from formulaforge.tables import Column, TableSpec, resolve_ref


@pytest.fixture
def df():
    d = pd.DataFrame({
        "Employee": ["Ann", "Bob", "Cy", "Di"],
        "Region": ["East", "West", "East", "West"],
        "Revenue": [10, 20, 30, 40],
    })
    d.index = range(2, 6)
    return d


def test_resolve_ref_layout(df):
    assert resolve_ref(df, "C3") == 20
    assert resolve_ref(df, "A1") == "Employee"
    assert resolve_ref(df, "C2:C5").shape == (4, 1)


@pytest.mark.parametrize("formula,expected", [
    ("SUM(C2:C5)", 100.0),
    ("AVERAGE(C2:C5)", 25.0),
    ("MAX(C2:C5)-MIN(C2:C5)", 30.0),
    ('COUNTIF(B2:B5,"East")', 2.0),
    ('SUMIFS(C2:C5,B2:B5,"East")', 40.0),
    ('SUMIF(B2:B5,"West",C2:C5)', 60.0),
    ('SUMIFS(C2:C5,C2:C5,">=20",C2:C5,"<=30")', 50.0),
    ('VLOOKUP("Cy",A2:C5,3,FALSE)', 30.0),
    ('INDEX(C2:C5,MATCH("Bob",A2:A5,0))', 20.0),
    ("INDEX(A2:A5,MATCH(MAX(C2:C5),C2:C5,0))", "Di"),
    ('IF(C3>15,"Yes","No")', "Yes"),
    ('IF(AND(C2>5,B2="West"),"Y","N")', "N"),
    ('IFERROR(VLOOKUP("Zed",A2:C5,3,FALSE),"Not found")', "Not found"),
    ("=SUM(C2:C5)", 100.0),  # leading '=' accepted
])
def test_evaluate_formula(df, formula, expected):
    assert evaluate_formula(formula, df) == expected


@pytest.mark.parametrize("bad", ["SUM(C2:C5", "FOO(C2)", "", 'VLOOKUP("Zed",A2:C5,3,FALSE)', "C2/0"])
def test_invalid_or_error_formulas_are_errors(df, bad):
    assert evaluate_formula(bad, df) == ERROR


def test_results_equal_semantics():
    assert results_equal(1.0, 1.0 + 1e-9)
    assert not results_equal(1.0, 1.1)
    assert results_equal("Yes", "yes")
    assert not results_equal("1", 1.0)
    assert not results_equal(ERROR, ERROR)  # two failures are never "equal"


def test_normalize_formula():
    assert normalize_formula("= sum( b2:b5 )") == "SUM(B2:B5)"
    assert normalize_formula('IF(A1="a b", 1, 2)') == 'IF(A1="a b",1,2)'


SPEC = TableSpec([Column("Region", "text"), Column("Revenue", "number"), Column("Cost", "number")], 8)


def test_exec_match_without_exact_match():
    # SUMIF and SUMIFS are different strings but the same computation
    gold = 'SUMIF(A2:A9,"East",B2:B9)'
    res = grade('SUMIFS(B2:B9,A2:A9,"East")', gold, SPEC, base_seed=1)
    assert res.exec_match and not res.exact_match


def test_wrong_column_is_caught_by_execution():
    res = grade("SUM(C2:C9)", "SUM(B2:B9)", SPEC, base_seed=1)
    assert not res.exec_match and res.pred_valid


def test_exact_match_implies_exec_match_and_invalid_is_flagged():
    ok = grade("SUM(B2:B9)", "SUM(B2:B9)", SPEC, base_seed=2)
    assert ok.exact_match and ok.exec_match
    bad = grade("SUM(B2:B9", "SUM(B2:B9)", SPEC, base_seed=2)
    assert not bad.exec_match and not bad.pred_valid
