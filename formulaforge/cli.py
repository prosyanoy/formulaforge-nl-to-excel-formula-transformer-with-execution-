"""Command-line entry point: `python -m formulaforge <command>`."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .analysis import evaluate_predictions, render_markdown, summarize
from .config import load_config
from .dataset import build_dataset, read_jsonl, summarize as summarize_rows, write_jsonl


def cmd_generate(args):
    cfg = load_config(args.config)
    d = cfg["data"]
    splits = build_dataset(d["n_train"], d["n_val"], d["n_test"], cfg["seed"], d["paraphrase_level"])
    for name, rows in splits.items():
        write_jsonl(Path(d["dir"]) / f"{name}.jsonl", rows)
        print(name, json.dumps(summarize_rows(rows)))


def cmd_train(args):
    from .train import copy_config, train

    cfg = load_config(args.config)
    out = train(cfg)
    if args.config:
        copy_config(args.config, out)
    print(f"saved run to {out}")


def cmd_evaluate(args):
    from .predict import Predictor

    cfg = load_config(args.config)
    run = Path(args.run or cfg["train"]["output_dir"])
    rows = read_jsonl(Path(cfg["data"]["dir"]) / f"{args.split}.jsonl")
    if args.limit:
        rows = rows[: args.limit]
    ev = cfg["eval"]
    if args.tables:
        ev["n_tables"] = args.tables
    beams = args.beams or ev["num_beams"]
    preds = Predictor.load(run).predict_rows(rows, num_beams=beams, batch_size=ev["batch_size"])
    records = evaluate_predictions(rows, preds, ev["n_tables"])
    summary = summarize(records)
    run.mkdir(parents=True, exist_ok=True)
    (run / f"eval_{args.split}.json").write_text(json.dumps({"summary": summary, "records": records}, indent=1))
    report = render_markdown(summary, records, f"FormulaForge error analysis ({args.split}, beams={beams})")
    (run / f"report_{args.split}.md").write_text(report)
    o = summary["overall"]
    print(f"{args.split}: n={o['n']} exact_match={o['exact_match']:.3f} exec_accuracy={o['exec_accuracy']:.3f} valid={o['valid_rate']:.3f}")
    print(f"report -> {run / f'report_{args.split}.md'}")


def cmd_report(args):
    path = Path(args.eval_json)
    blob = json.loads(path.read_text())
    print(render_markdown(blob["summary"], blob["records"]))


def cmd_demo(args):
    from .demo import answer, format_table, preset_table
    from .predict import Predictor

    cfg = load_config(args.config)
    run = args.run or cfg["train"]["output_dir"]
    predictor = Predictor.load(run)
    df = pd.read_csv(args.csv) if args.csv else preset_table(args.table, args.seed)
    print(format_table(df), "\n")

    def respond(req):
        out = answer(predictor, req, df, args.beams)
        print(f"formula: {out['formula']}\nresult:  {out['result'] if out['ok'] else 'could not evaluate'}\n")

    if args.request:
        respond(args.request)
        return
    print("Type a spreadsheet request (empty line to quit).")
    while True:
        try:
            req = input("> ").strip()
        except EOFError:
            break
        if not req:
            break
        respond(req)


def main(argv=None):
    p = argparse.ArgumentParser(prog="formulaforge", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn, help_ in [("generate", cmd_generate, "build the synthetic dataset"),
                            ("train", cmd_train, "train the seq2seq model"),
                            ("evaluate", cmd_evaluate, "execution-based evaluation + error analysis report"),
                            ("demo", cmd_demo, "ask for a formula on a sample table")]:
        s = sub.add_parser(name, help=help_)
        s.add_argument("--config", default="configs/scratch.yaml")
        s.set_defaults(fn=fn)
        if name in ("evaluate", "demo"):
            s.add_argument("--run", help="run directory (default: train.output_dir from config)")
            s.add_argument("--beams", type=int, default=0 if name == "evaluate" else 4)
        if name == "evaluate":
            s.add_argument("--split", default="test", choices=["val", "test"])
            s.add_argument("--limit", type=int, default=0)
            s.add_argument("--tables", type=int, default=0, help="held-out tables per example (default: eval.n_tables)")
        if name == "demo":
            s.add_argument("--request", help="single request; omit for an interactive prompt")
            s.add_argument("--table", default="sales", choices=["sales", "employees", "orders"])
            s.add_argument("--csv", help="use your own CSV instead of a preset table")
            s.add_argument("--seed", type=int, default=7)
    s = sub.add_parser("report", help="re-render a markdown report from an eval_*.json")
    s.add_argument("eval_json")
    s.set_defaults(fn=cmd_report)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
