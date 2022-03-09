"""Reproducible training script for the NL -> formula seq2seq model (T5 architecture)."""
from __future__ import annotations

import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import read_jsonl
from .evaluator import grade
from .predict import Predictor
from .tables import TableSpec
from .tokenization import HFTokenizer, WordTokenizer, serialize_input


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(cfg: dict, tokenizer):
    from transformers import T5Config, T5ForConditionalGeneration

    m = cfg["model"]
    if m["kind"] == "pretrained":
        return T5ForConditionalGeneration.from_pretrained(m["pretrained_name"])
    conf = T5Config(
        vocab_size=len(tokenizer), d_model=m["d_model"], d_kv=m["d_kv"], d_ff=m["d_ff"],
        num_layers=m["num_layers"], num_decoder_layers=m["num_layers"], num_heads=m["num_heads"],
        dropout_rate=m["dropout"], pad_token_id=tokenizer.pad_id, eos_token_id=tokenizer.eos_id,
        decoder_start_token_id=tokenizer.pad_id, feed_forward_proj="relu",
    )
    return T5ForConditionalGeneration(conf)


def encode_rows(rows, tokenizer, max_src, max_tgt):
    out = []
    for r in rows:
        src = serialize_input(r["nl"], TableSpec.from_dict(r["spec"]))
        out.append((tokenizer.encode_src(src, max_src), tokenizer.encode_tgt(r["formula"], max_tgt)))
    return out


def make_collate(pad_id):
    def collate(batch):
        def pad(seqs, value):
            L = max(len(s) for s in seqs)
            return torch.tensor([s + [value] * (L - len(s)) for s in seqs])
        src = pad([b[0] for b in batch], pad_id)
        tgt = pad([b[1] for b in batch], -100)
        return {"input_ids": src, "attention_mask": (src != pad_id).long(), "labels": tgt}
    return collate


@torch.no_grad()
def val_loss(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        total += model(**b).loss.item() * b["input_ids"].size(0)
        n += b["input_ids"].size(0)
    return total / max(n, 1)


def train(cfg: dict, data_dir: str | None = None) -> Path:
    from torch.utils.tensorboard import SummaryWriter

    tc, mc = cfg["train"], cfg["model"]
    set_seed(cfg["seed"])
    data_dir = Path(data_dir or cfg["data"]["dir"])
    train_rows, val_rows = read_jsonl(data_dir / "train.jsonl"), read_jsonl(data_dir / "val.jsonl")
    out = Path(tc["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    if mc["kind"] == "pretrained":
        tokenizer = HFTokenizer(mc["pretrained_name"])
    else:
        texts = [serialize_input(r["nl"], TableSpec.from_dict(r["spec"])) + " " + r["formula"] for r in train_rows]
        tokenizer = WordTokenizer.build(texts)
    model = build_model(cfg, tokenizer)
    device = pick_device()
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} params={n_params / 1e6:.2f}M vocab={len(tokenizer)} train={len(train_rows)} val={len(val_rows)}")

    collate = make_collate(tokenizer.pad_id)
    g = torch.Generator().manual_seed(cfg["seed"])
    train_loader = DataLoader(encode_rows(train_rows, tokenizer, tc["max_src_len"], tc["max_tgt_len"]),
                              batch_size=tc["batch_size"], shuffle=True, collate_fn=collate, generator=g)
    val_loader = DataLoader(encode_rows(val_rows, tokenizer, tc["max_src_len"], tc["max_tgt_len"]),
                            batch_size=tc["batch_size"], collate_fn=collate)

    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    total_steps = tc["epochs"] * len(train_loader)
    warmup = max(1, int(tc["warmup_ratio"] * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))
    writer = SummaryWriter(str(out / "tb"))
    history, step, t0 = [], 0, time.time()

    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % tc["log_every"] == 0:
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/lr", sched.get_last_lr()[0], step)
                print(f"epoch {epoch} step {step}/{total_steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
        vl = val_loss(model, val_loader, device)
        writer.add_scalar("val/loss", vl, step)
        entry = {"epoch": epoch, "step": step, "val_loss": vl}
        # quick execution-accuracy probe on a slice of the validation set
        if tc["val_exec_samples"] and (epoch == tc["epochs"] or epoch % 4 == 0):
            entry["val_exec_acc"] = _probe_exec(model, tokenizer, val_rows[: tc["val_exec_samples"]], device, cfg)
            writer.add_scalar("val/exec_acc", entry["val_exec_acc"], step)
        history.append(entry)
        print(json.dumps(entry), flush=True)

    model.save_pretrained(out / "model")
    tokenizer.save(out / "model")
    (out / "history.json").write_text(json.dumps({"params": n_params, "history": history, "config": cfg}, indent=2))
    writer.close()
    return out


def _probe_exec(model, tokenizer, rows, device, cfg) -> float:
    pred = Predictor(model, tokenizer, device)
    preds = pred.predict_rows(rows, num_beams=1, batch_size=cfg["eval"]["batch_size"])
    ok = sum(grade(p, r["formula"], TableSpec.from_dict(r["spec"]), r["seed"], 2).exec_match for p, r in zip(preds, rows))
    return ok / max(len(rows), 1)


def copy_config(src: str, out_dir: Path):
    shutil.copy(src, out_dir / "config.yaml")
