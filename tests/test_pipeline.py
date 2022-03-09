import json

from formulaforge.analysis import evaluate_predictions, failure_category, render_markdown, summarize
from formulaforge.cli import main
from formulaforge.config import load_config
from formulaforge.dataset import build_dataset
from formulaforge.demo import infer_spec, preset_table
from formulaforge.tokenization import WordTokenizer


def _rows():
    return build_dataset(10, 10, 80, seed=2)["test"]


def test_failure_categories():
    gold = 'SUMIFS(B2:B9,A2:A9,"East")'
    assert failure_category("", gold, False) == "empty_output"
    assert failure_category("SUMIFS(B2:B9", gold, False) == "invalid_formula"
    assert failure_category('AVERAGEIFS(B2:B9,A2:A9,"East")', gold, True) == "wrong_function"
    assert failure_category('SUMIFS(C2:C9,A2:A9,"East")', gold, True) == "wrong_reference"
    assert failure_category('SUMIFS(B2:B9,A2:A9,"West")', gold, True) == "wrong_string_literal"
    assert failure_category('SUMIFS(B2:B9,A2:A9,"East")+1', gold, True) == "wrong_operator_or_number"


def test_oracle_scores_100_and_broken_model_scores_low():
    rows = _rows()
    oracle = summarize(evaluate_predictions(rows, [r["formula"] for r in rows]))
    assert oracle["overall"]["exec_accuracy"] == 1.0 and oracle["overall"]["exact_match"] == 1.0
    assert oracle["failure_modes"] == {}
    broken = evaluate_predictions(rows, ["SUM(" for _ in rows])
    s = summarize(broken)
    assert s["overall"]["exec_accuracy"] == 0.0
    assert s["failure_modes"] == {"invalid_formula": len(rows)}


def test_report_breaks_down_by_family_and_tier():
    rows = _rows()
    preds = [r["formula"] if i % 2 else "SUM(A2:A3)" for i, r in enumerate(rows)]
    recs = evaluate_predictions(rows, preds)
    summary = summarize(recs)
    md = render_markdown(summary, recs)
    assert "By function family" in md and "By difficulty tier" in md and "family x tier" in md
    assert sum(b["n"] for b in summary["by_tier"].values()) == len(rows)
    assert 0.3 < summary["overall"]["exec_accuracy"] < 0.7


def test_demo_schema_inference():
    spec = infer_spec(preset_table("employees"))
    assert [c.kind for c in spec.columns] == ["key", "text", "number", "number"]


def test_config_defaults_and_override(tmp_path, monkeypatch):
    p = tmp_path / "c.yaml"
    p.write_text("train:\n  epochs: 3\n")
    monkeypatch.setenv("FORMULAFORGE_RUNS_DIR", str(tmp_path / "r"))
    cfg = load_config(p)
    assert cfg["train"]["epochs"] == 3 and cfg["train"]["batch_size"] == 64
    assert cfg["train"]["output_dir"].startswith(str(tmp_path / "r"))


def test_end_to_end_tiny_train_evaluate(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"""
seed: 0
data: {{dir: {tmp_path}/data, n_train: 96, n_val: 16, n_test: 16}}
model: {{kind: scratch, d_model: 32, d_kv: 8, d_ff: 64, num_layers: 1, num_heads: 4}}
train: {{output_dir: {tmp_path}/run, epochs: 1, batch_size: 32, val_exec_samples: 8}}
eval: {{n_tables: 2, num_beams: 1}}
""")
    main(["generate", "--config", str(cfg)])
    main(["train", "--config", str(cfg)])
    main(["evaluate", "--config", str(cfg), "--split", "test"])
    blob = json.loads((tmp_path / "run" / "eval_test.json").read_text())
    assert blob["summary"]["overall"]["n"] == 16
    assert (tmp_path / "run" / "report_test.md").exists()
    # the tokenizer shipped with the run decodes what the model emits
    assert isinstance(WordTokenizer.load(tmp_path / "run" / "model").itos, list)
