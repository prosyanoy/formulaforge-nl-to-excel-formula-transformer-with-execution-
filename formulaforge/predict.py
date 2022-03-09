"""Inference: natural-language request + table schema -> Excel formula."""
from __future__ import annotations

from pathlib import Path

import torch

from .tables import TableSpec
from .tokenization import load_tokenizer, serialize_input


class Predictor:
    def __init__(self, model, tokenizer, device=None, max_src_len: int = 128, max_new_tokens: int = 96):
        self.model, self.tok = model, tokenizer
        self.device = device or torch.device("cpu")
        self.model.to(self.device).eval()
        self.max_src_len, self.max_new_tokens = max_src_len, max_new_tokens

    @classmethod
    def load(cls, run_dir: str | Path, device=None) -> "Predictor":
        from transformers import T5ForConditionalGeneration

        model_dir = Path(run_dir) / "model" if (Path(run_dir) / "model").exists() else Path(run_dir)
        return cls(T5ForConditionalGeneration.from_pretrained(model_dir), load_tokenizer(model_dir), device)

    @torch.no_grad()
    def predict_texts(self, texts: list[str], num_beams: int = 1, batch_size: int = 64) -> list[str]:
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = [""] * len(texts)
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            enc = [self.tok.encode_src(texts[i], self.max_src_len) for i in idx]
            L = max(map(len, enc))
            ids = torch.tensor([e + [self.tok.pad_id] * (L - len(e)) for e in enc], device=self.device)
            gen = self.model.generate(input_ids=ids, attention_mask=(ids != self.tok.pad_id).long(),
                                      max_new_tokens=self.max_new_tokens, num_beams=num_beams, do_sample=False)
            for i, g in zip(idx, gen):
                out[i] = self.tok.decode(g.tolist())
        return out

    def predict_rows(self, rows: list[dict], **kw) -> list[str]:
        return self.predict_texts([serialize_input(r["nl"], TableSpec.from_dict(r["spec"])) for r in rows], **kw)

    def predict(self, nl: str, spec: TableSpec, **kw) -> str:
        return self.predict_texts([serialize_input(nl, spec)], **kw)[0]
