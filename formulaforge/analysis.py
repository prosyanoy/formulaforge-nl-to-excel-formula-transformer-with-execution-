"""Aggregate execution-based evaluation and error analysis."""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from .evaluator import grade, normalize_formula
from .tables import TableSpec

_FUNC = re.compile(r"([A-Z][A-Z0-9.]*)\(")
_REF = re.compile(r"\$?[A-Z]{1,2}\$?\d+(?::\$?[A-Z]{1,2}\$?\d+)?")
_STR = re.compile(r'"[^"]*"')


def failure_category(pred: str, gold: str, valid: bool) -> str:
    """Why did a non-matching prediction fail? Coarse, mutually exclusive buckets."""
    if not pred.strip():
        return "empty_output"
    if not valid:
        return "invalid_formula"  # parse error, unknown function, #REF!, #N/A, ...
    p, g = normalize_formula(pred), normalize_formula(gold)
    if sorted(_FUNC.findall(p)) != sorted(_FUNC.findall(g)):
        return "wrong_function"
    if sorted(_REF.findall(p)) != sorted(_REF.findall(g)):
        return "wrong_reference"  # right functions, wrong column / range / row
    if sorted(_STR.findall(p)) != sorted(_STR.findall(g)):
        return "wrong_string_literal"
    return "wrong_operator_or_number"


def evaluate_predictions(rows: list[dict], preds: list[str], n_tables: int = 3) -> list[dict]:
    records = []
    for r, p in zip(rows, preds):
        res = grade(p, r["formula"], TableSpec.from_dict(r["spec"]), r["seed"], n_tables)
        rec = {
            "id": r["id"], "nl": r["nl"], "family": r["family"], "tier": r["tier"], "template": r["template"],
            "gold": r["formula"], "pred": p, "exact_match": res.exact_match, "exec_match": res.exec_match,
            "valid": res.pred_valid, "gold_result": repr(res.gold_results[0]), "pred_result": repr(res.pred_results[0]),
        }
        rec["failure"] = None if res.exec_match else failure_category(p, r["formula"], res.pred_valid)
        # exact match that doesn't execute equal would be a bug in the dataset; surface it
        records.append(rec)
    return records


def _rate(rs, key):
    return sum(r[key] for r in rs) / len(rs) if rs else float("nan")


def summarize(records: list[dict]) -> dict:
    def block(rs):
        return {"n": len(rs), "exact_match": _rate(rs, "exact_match"), "exec_accuracy": _rate(rs, "exec_match"),
                "valid_rate": _rate(rs, "valid")}
    groups: dict[str, dict] = {"overall": block(records)}
    for key in ("family", "tier"):
        buckets = defaultdict(list)
        for r in records:
            buckets[r[key]].append(r)
        groups[f"by_{key}"] = {str(k): block(v) for k, v in sorted(buckets.items())}
    cross = defaultdict(list)
    for r in records:
        cross[(r["family"], r["tier"])].append(r)
    groups["by_family_tier"] = {f"{f}|{t}": block(v) for (f, t), v in sorted(cross.items())}
    groups["failure_modes"] = dict(Counter(r["failure"] for r in records if r["failure"]).most_common())
    groups["exec_correct_but_not_exact"] = sum(r["exec_match"] and not r["exact_match"] for r in records)
    return groups


def _pct(x):
    return "n/a" if x != x else f"{100 * x:.1f}%"


def render_markdown(summary: dict, records: list[dict], title: str = "FormulaForge error analysis", n_examples: int = 3) -> str:
    o = summary["overall"]
    L = [f"# {title}", "",
         f"Examples: **{o['n']}** | Exact match: **{_pct(o['exact_match'])}** | "
         f"Execution accuracy: **{_pct(o['exec_accuracy'])}** | Valid-formula rate: **{_pct(o['valid_rate'])}**", "",
         f"{summary['exec_correct_but_not_exact']} predictions were execution-correct but not string-identical to gold "
         "(equivalent formulas that exact match would wrongly penalise).", ""]
    for key, label in (("by_tier", "Difficulty tier"), ("by_family", "Function family")):
        L += [f"## By {label.lower()}", "", f"| {label} | N | Exact match | Execution acc. | Valid |", "|---|---|---|---|---|"]
        for k, b in summary[key].items():
            L.append(f"| {k} | {b['n']} | {_pct(b['exact_match'])} | {_pct(b['exec_accuracy'])} | {_pct(b['valid_rate'])} |")
        L.append("")
    tiers = sorted(summary["by_tier"])
    fams = list(summary["by_family"])
    L += ["## Execution accuracy: function family x tier", "", "| Family | " + " | ".join(f"tier {t}" for t in tiers) + " |",
          "|---|" + "---|" * len(tiers)]
    for f in fams:
        cells = []
        for t in tiers:
            b = summary["by_family_tier"].get(f"{f}|{t}")
            cells.append(f"{_pct(b['exec_accuracy'])} (n={b['n']})" if b else "-")
        L.append(f"| {f} | " + " | ".join(cells) + " |")
    L += ["", "## Failure modes (execution mismatches)", ""]
    n_fail = sum(summary["failure_modes"].values())
    if n_fail:
        L += ["| Failure mode | Count | Share of failures |", "|---|---|---|"]
        for k, v in summary["failure_modes"].items():
            L.append(f"| {k} | {v} | {_pct(v / n_fail)} |")
    else:
        L.append("No failures.")
    L += ["", "## Example failures", ""]
    by_fam = defaultdict(list)
    for r in records:
        if r["failure"]:
            by_fam[r["family"]].append(r)
    for fam, rs in sorted(by_fam.items()):
        L += [f"### {fam}", ""]
        for r in rs[:n_examples]:
            L += [f"- **{r['failure']}** (tier {r['tier']}): _{r['nl']}_", f"  - gold: `{r['gold']}` -> {r['gold_result']}",
                  f"  - pred: `{r['pred']}` -> {r['pred_result']}"]
        L.append("")
    return "\n".join(L)
