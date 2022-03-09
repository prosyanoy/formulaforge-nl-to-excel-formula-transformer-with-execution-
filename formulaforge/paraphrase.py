"""Lightweight, deterministic paraphrasing of NL request skeletons.

Skeletons contain `{placeholders}` that are filled *after* paraphrasing, so column names and
literal values are never rewritten.
"""
from __future__ import annotations

import random
import re

SYNONYMS = [
    ["Calculate", "Compute", "Find", "Get", "Work out", "Determine", "Give me"],
    ["calculate", "compute", "find", "get", "work out", "determine"],
    ["rows", "records", "entries"],
    ["where", "for which"],
    ["column", "field"],
    ["each", "every"],
    ["all", "all the"],
]
PREFIXES = ["", "", "", "Please ", "Can you ", "I need to ", "How do I ", "Show me how to ", "Help me "]
SUFFIXES = ["", "", "", " in Excel", " please", " in the sheet"]
_PLACEHOLDER = re.compile(r"\{[a-z0-9_]+\}")


def _swap_synonyms(rng: random.Random, text: str, p: float) -> str:
    for group in SYNONYMS:
        for word in group:
            pat = re.compile(rf"(?<![\w{{]){re.escape(word)}(?![\w}}])")
            if pat.search(text) and rng.random() < p:
                text = pat.sub(rng.choice(group), text, count=1)
                break
    return text


def paraphrase(rng: random.Random, skeleton: str, level: float = 0.5) -> str:
    """Return a surface variant of `skeleton` (placeholders preserved)."""
    text = _swap_synonyms(rng, skeleton, level)
    if rng.random() < 0.25:
        text = re.sub(r"\bthe (?!\{)", "", text, count=1)
    text = text.rstrip("?.")
    ends_question = skeleton.rstrip().endswith("?")
    prefix = rng.choice(PREFIXES)
    # "How do I"/"Can you" style prefixes need a verb-initial request; only use on imperatives
    imperative = bool(re.match(r"^(Calculate|Compute|Find|Get|Work out|Determine|Give me|Add|Sum|Count|Look|Return|Show|Fetch|Lookup|Average|Total|Check)\b", text))
    if prefix and imperative:
        text = prefix + text[0].lower() + text[1:]
    suffix = rng.choice(SUFFIXES)
    text += suffix
    if ends_question or prefix.startswith(("Can you", "How do")) and imperative:
        text += "?"
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def fill(skeleton: str, **values: str) -> str:
    out = skeleton
    for k, v in values.items():
        out = out.replace("{" + k + "}", str(v))
    assert not _PLACEHOLDER.search(out), f"unfilled placeholder in {out!r}"
    return out
