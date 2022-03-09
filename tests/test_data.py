import random

import pytest

from formulaforge.dataset import build_dataset, example_key, summarize
from formulaforge.evaluator import ERROR, evaluate_formula, grade
from formulaforge.paraphrase import fill, paraphrase
from formulaforge.tables import TableSpec, generate_table
from formulaforge.templates import FAMILIES, TEMPLATES, sample_example
from formulaforge.tokenization import WordTokenizer, serialize_input


def test_generate_table_is_deterministic_and_respects_constraints():
    ex = sample_example(random.Random(3), [t for t in TEMPLATES if t.name == "vlookup_basic"])
    spec = TableSpec.from_dict(ex["spec"])
    a, b = generate_table(spec, 5), generate_table(spec, 5)
    assert a.equals(b)
    assert not a.equals(generate_table(spec, 6))
    for idx, vals in spec.ensure.items():
        assert set(vals) <= set(a.iloc[:, idx])
    key_col = next(i for i, c in enumerate(spec.columns) if c.kind == "key")
    assert a.iloc[:, key_col].is_unique


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda t: t.name)
def test_every_template_gold_formula_executes_on_many_tables(template):
    rng = random.Random(11)
    for _ in range(15):
        ex = sample_example(rng, [template])
        spec = TableSpec.from_dict(ex["spec"])
        for seed in range(3):
            df = generate_table(spec, seed)
            assert evaluate_formula(ex["formula"], df) != ERROR or template.name == "averageifs", (ex["nl"], ex["formula"])
        assert ex["family"] == template.family and ex["tier"] == template.tier
        assert grade(ex["formula"], ex["formula"], spec, 1).exact_match


def test_templates_cover_required_families_and_tiers():
    assert {"SUM", "IF", "VLOOKUP", "SUMIFS", "INDEX_MATCH"} <= set(FAMILIES)
    assert {t.tier for t in TEMPLATES} == {1, 2, 3}


def test_iferror_templates_actually_take_the_fallback_branch():
    rng = random.Random(1)
    for name in ("vlookup_iferror", "index_match_iferror"):
        ex = sample_example(rng, [t for t in TEMPLATES if t.name == name])
        df = generate_table(TableSpec.from_dict(ex["spec"]), 9)
        assert evaluate_formula(ex["formula"], df) == "Not found"


def test_build_dataset_is_reproducible_deduped_and_disjoint():
    a = build_dataset(300, 40, 40, seed=5)
    b = build_dataset(300, 40, 40, seed=5)
    assert a == b
    inputs = {k: [serialize_input(r["nl"], TableSpec.from_dict(r["spec"])) for r in v] for k, v in a.items()}
    all_inputs = sum(inputs.values(), [])
    assert len(all_inputs) == len(set(all_inputs))  # dedup within and across splits
    assert len({example_key(r) for v in a.values() for r in v}) == 380
    s = summarize(a["train"])
    assert s["n"] == 300 and set(s["by_tier"]) == {1, 2, 3}


def test_paraphrase_preserves_placeholders_and_varies():
    rng = random.Random(0)
    outs = {paraphrase(rng, "Calculate the total {c} where {t} is {v}") for _ in range(40)}
    assert len(outs) > 5
    assert all("{c}" in o and "{t}" in o and "{v}" in o for o in outs)
    assert fill("sum {c}", c="Revenue") == "sum Revenue"
    with pytest.raises(AssertionError):
        fill("sum {c} {d}", c="x")


def test_word_tokenizer_roundtrip():
    formulas = ['IF(AND(B6>=600,A6="Legal"),"Yes","No")', 'SUMIFS(C2:C14,A2:A14,"IT",B2:B14,"<=60")', 'COUNTIF(A2:A9,"<>x")',
                'IF(SUM(B2:B9)>350,"Target met","Below target")', 'IFERROR(VLOOKUP("Zed",A2:C9,3,FALSE),"Not found")']
    tok = WordTokenizer.build(formulas)
    for f in formulas:
        assert tok.decode(tok.encode_tgt(f)) == f


def test_dataset_gold_formulas_are_defined_on_all_held_out_tables():
    from formulaforge.dataset import MAX_TABLES

    rows = build_dataset(200, 10, 200, seed=8)["test"]
    for r in rows:
        res = grade(r["formula"], r["formula"], TableSpec.from_dict(r["spec"]), r["seed"], MAX_TABLES)
        assert res.exec_match and res.pred_valid, (r["nl"], r["formula"], res.gold_results)
