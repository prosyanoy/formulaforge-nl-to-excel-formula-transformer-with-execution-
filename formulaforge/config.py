"""YAML config loading with env-var overrides for output locations."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

DEFAULTS = {
    "seed": 0,
    "data": {"dir": "data", "n_train": 20000, "n_val": 500, "n_test": 1000, "paraphrase_level": 0.5},
    "model": {"kind": "scratch", "d_model": 256, "d_kv": 32, "d_ff": 768, "num_layers": 4, "num_heads": 8,
              "dropout": 0.1, "pretrained_name": "t5-small"},
    "train": {"output_dir": "runs/scratch", "epochs": 12, "batch_size": 64, "lr": 1e-3, "warmup_ratio": 0.05,
              "weight_decay": 0.01, "max_src_len": 128, "max_tgt_len": 96, "grad_clip": 1.0, "log_every": 50,
              "val_exec_samples": 200},
    "eval": {"n_tables": 3, "num_beams": 1, "batch_size": 64},
}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load_config(path: str | Path | None) -> dict:
    cfg = DEFAULTS
    if path:
        with open(path) as f:
            cfg = _merge(DEFAULTS, yaml.safe_load(f) or {})
    # environment overrides keep paths configurable without editing YAML (see .env.example)
    if os.environ.get("FORMULAFORGE_DATA_DIR"):
        cfg["data"]["dir"] = os.environ["FORMULAFORGE_DATA_DIR"]
    if os.environ.get("FORMULAFORGE_RUNS_DIR"):
        run = Path(cfg["train"]["output_dir"]).name
        cfg["train"]["output_dir"] = str(Path(os.environ["FORMULAFORGE_RUNS_DIR"]) / run)
    return cfg
