"""Synthetic dataset builder: sample, dedup, split, and JSONL I/O."""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from .evaluator import ERROR, evaluate_formula, held_out_tables
from .tables import TableSpec
from .templates import sample_example
from .tokenization import serialize_input


MAX_TABLES = 5  # gold formulas are validated on this many held-out tables


def example_key(ex: dict) -> str:
    """Identity used for dedup: the exact encoder input plus the target formula."""
    return serialize_input(ex["nl"], TableSpec.from_dict(ex["spec"])) + " => " + ex["formula"]


def build_dataset(n_train: int, n_val: int, n_test: int, seed: int = 0, paraphrase_level: float = 0.5) -> dict[str, list[dict]]:
    """Sample until every split is full. Splits are disjoint on the *encoder input* (not just on
    the (input, formula) pair), so no test request ever appears verbatim in training."""
    rng = random.Random(seed)
    targets = {"train": n_train, "val": n_val, "test": n_test}
    # Sample the eval splits first so that they are never starved by train-set dedup.
    order = ["test", "val", "train"]
    splits: dict[str, list[dict]] = {k: [] for k in targets}
    seen_inputs: set[str] = set()
    attempts, limit = 0, 50 * sum(targets.values())
    for name in order:
        while len(splits[name]) < targets[name] and attempts < limit:
            attempts += 1
            ex = sample_example(rng, paraphrase_level=paraphrase_level)
            inp = serialize_input(ex["nl"], TableSpec.from_dict(ex["spec"]))
            if inp in seen_inputs:
                continue
            ex["seed"] = rng.randrange(10**6)
            spec = TableSpec.from_dict(ex["spec"])
            # the gold formula must be defined on every held-out table the evaluator will use
            if any(evaluate_formula(ex["formula"], t) == ERROR for t in held_out_tables(spec, ex["seed"], MAX_TABLES)):
                continue
            seen_inputs.add(inp)
            ex["id"] = f"{name}-{len(splits[name]):06d}"
            splits[name].append(ex)
    for name, n in targets.items():
        if len(splits[name]) < n:
            raise RuntimeError(f"could only build {len(splits[name])}/{n} unique examples for {name}")
    return splits


def write_jsonl(path: str | Path, rows: list[dict]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "by_family": dict(sorted(Counter(r["family"] for r in rows).items())),
        "by_tier": dict(sorted(Counter(r["tier"] for r in rows).items())),
    }
