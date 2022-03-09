"""Input serialization and tokenizers.

`serialize_input` turns (request, table schema) into the encoder text. Two tokenizers share one
interface (`encode_src`, `encode_tgt`, `decode`):

* `WordTokenizer` - regex word/digit/punctuation vocabulary built from the training data, used
  when training the T5 architecture from scratch (T5's sentencepiece vocab has no '<' or '{').
* `HFTokenizer` - wraps a pretrained tokenizer (e.g. t5-small) and escapes characters it lacks.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .tables import TableSpec

PAD, EOS, UNK = "<pad>", "</s>", "<unk>"
_TOKEN = re.compile(r"[A-Za-z]+|\d|<=|>=|<>|[^\sA-Za-z\d]")


def serialize_input(nl: str, spec: TableSpec) -> str:
    cols = "; ".join(f"{spec.letter(i)} {c.name} {c.kind}" for i, c in enumerate(spec.columns))
    return f"{nl} schema: {cols}; rows {spec.first_row} to {spec.last_row}"


_QUOTED = re.compile(r'"[^"]*"')


def protect_spaces(text: str) -> str:
    """Spaces inside "quoted strings" become '_' so that e.g. "Target met" survives tokenization."""
    return _QUOTED.sub(lambda m: m.group(0).replace(" ", "_"), text)


def split_tokens(text: str) -> list[str]:
    return _TOKEN.findall(protect_spaces(text))


class WordTokenizer:
    def __init__(self, vocab: list[str]):
        self.itos = list(vocab)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad_id, self.eos_id, self.unk_id = (self.stoi[PAD], self.stoi[EOS], self.stoi[UNK])

    @classmethod
    def build(cls, texts: list[str]) -> "WordTokenizer":
        seen = sorted({t for text in texts for t in split_tokens(text)})
        return cls([PAD, EOS, UNK] + seen)

    def __len__(self):
        return len(self.itos)

    def _encode(self, text: str, max_len: int) -> list[int]:
        ids = [self.stoi.get(t, self.unk_id) for t in split_tokens(text)][: max_len - 1]
        return ids + [self.eos_id]

    encode_src = encode_tgt = lambda self, text, max_len=128: self._encode(text, max_len)

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i == self.eos_id:
                break
            if i not in (self.pad_id, self.unk_id):
                out.append(self.itos[i])
        # formulas contain no spaces outside strings, so tokens concatenate losslessly
        return "".join(out).replace("_", " ")

    def save(self, directory: str | Path):
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "vocab.json").write_text(json.dumps(self.itos))

    @classmethod
    def load(cls, directory: str | Path) -> "WordTokenizer":
        return cls(json.loads((Path(directory) / "vocab.json").read_text()))


class HFTokenizer:
    """Pretrained sentencepiece tokenizer with reversible escaping of unsupported characters."""

    ESCAPES = {"<": "LESSTHAN"}

    def __init__(self, name_or_path: str):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name_or_path)
        self.pad_id, self.eos_id = self.tok.pad_token_id, self.tok.eos_token_id

    def __len__(self):
        return len(self.tok)

    def _esc(self, text: str) -> str:
        for k, v in self.ESCAPES.items():
            text = text.replace(k, v)
        return text

    def encode_src(self, text: str, max_len: int = 128) -> list[int]:
        return self.tok(self._esc(text), truncation=True, max_length=max_len).input_ids

    encode_tgt = encode_src

    def decode(self, ids) -> str:
        text = self.tok.decode([int(i) for i in ids], skip_special_tokens=True)
        text = re.sub(r"\s+", "", text)
        for k, v in self.ESCAPES.items():
            text = text.replace(v, k)
        return text

    def save(self, directory: str | Path):
        self.tok.save_pretrained(directory)

    @classmethod
    def load(cls, directory: str | Path) -> "HFTokenizer":
        return cls(str(directory))


def load_tokenizer(directory: str | Path):
    return WordTokenizer.load(directory) if (Path(directory) / "vocab.json").exists() else HFTokenizer.load(directory)
