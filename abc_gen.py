#!/usr/bin/env python3
# synthetic_abc.py · v0.1.0
"""
Generate synthetic—yet syntactically valid—ABC notation files.

Major features
--------------
* Deterministic output with --seed
* Plugin architecture for new element generators
* No hidden global state
* --out to save directly to disk

Usage
-----
python synthetic_abc.py 50
python synthetic_abc.py 100 --seed 42 --out tune.abc
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List

__version__ = "0.1.0"

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ABCConfig:
    loc: int = 50               # target total lines (including headers)
    seed: int | None = None
    weights: Dict[str, float] = field(default_factory=lambda: {
        "comment":    0.10,
        "note_line":  0.60,
        "chord_line": 0.30,
    })


# ──────────────────────────────────────────────────────────────
# Generator registry
# ──────────────────────────────────────────────────────────────

GeneratorFn = Callable[[Dict], str]
_REGISTRY: Dict[str, GeneratorFn] = {}

def register(kind: str) -> Callable[[GeneratorFn], GeneratorFn]:
    def inner(fn: GeneratorFn) -> GeneratorFn:
        if kind in _REGISTRY:
            raise ValueError(f"Duplicate generator: {kind}")
        _REGISTRY[kind] = fn
        return fn
    return inner


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

NOTES = ["A","B","C","D","E","F","G"]
ACCIDENTALS = ["", "^", "_"]  # natural, sharp, flat
DURATIONS = ["", "2", "/", "3/2"]  # quarter, half, eighth, dotted

def make_note(rng: random.Random) -> str:
    acc = rng.choice(ACCIDENTALS)
    note = rng.choice(NOTES)
    dur = rng.choice(DURATIONS)
    return f"{acc}{note}{dur}"

def make_rest(rng: random.Random) -> str:
    dur = rng.choice(DURATIONS) or "1"
    return f"z{dur}"


# ──────────────────────────────────────────────────────────────
# Generators
# ──────────────────────────────────────────────────────────────

@register("comment")
def gen_comment(state: Dict) -> str:
    rng = state["rng"]
    texts = ["riff", "lively", "slow", "swing", "fast", "melodic"]
    return f"%% {rng.choice(texts)}\n"

@register("note_line")
def gen_note_line(state: Dict) -> str:
    rng = state["rng"]
    # number of tokens per measure
    count = rng.randint(4, 8)
    tokens: List[str] = []
    for _ in range(count):
        if rng.random() < 0.1:
            tokens.append(make_rest(rng))
        else:
            tokens.append(make_note(rng))
    line = "| " + " ".join(tokens) + " |"
    return line + "\n"

@register("chord_line")
def gen_chord_line(state: Dict) -> str:
    rng = state["rng"]
    # number of chords in line
    count = rng.randint(2, 5)
    chords: List[str] = []
    for _ in range(count):
        # pick three distinct notes
        notes = rng.sample(NOTES, 3)
        # optional accidentals inside chords? skip for simplicity
        chord = "[" + "".join(notes) + "]"
        dur = rng.choice(DURATIONS)
        chords.append(chord + dur)
    line = "| " + " ".join(chords) + " |"
    return line + "\n"


# ──────────────────────────────────────────────────────────────
# Build & CLI
# ──────────────────────────────────────────────────────────────

def build_abc(cfg: ABCConfig) -> str:
    rng = random.Random(cfg.seed)
    state = {"rng": rng}
    parts: List[str] = []

    # Mandatory header fields
    parts.append("X:1\n")
    parts.append(f"T:Auto-generated Tune {rng.randint(1,100)}\n")
    meter = rng.choice(["4/4", "3/4", "6/8"])
    parts.append(f"M:{meter}\n")
    parts.append("L:1/8\n")
    tempo = rng.randint(80, 150)
    parts.append(f"Q:1/4={tempo}\n")
    key = rng.choice(["Cmaj", "Gmaj", "Dmin", "Fmaj", "Am"])
    parts.append(f"K:{key}\n\n")

    lines = len(parts)
    kinds, weights = zip(*cfg.weights.items())

    # Generate content lines until reaching loc
    while lines < cfg.loc:
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        snippet = _REGISTRY[kind](state)
        parts.append(snippet)
        lines += 1  # each snippet is exactly one line

    return "".join(parts)


def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate a synthetic ABC notation file.")
    p.add_argument("loc", nargs="?", type=int, default=50, help="Approximate line count")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out", type=Path, help="Path to save generated .abc file")
    args = p.parse_args()

    cfg = ABCConfig(loc=args.loc, seed=args.seed)
    abc = build_abc(cfg)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(abc, encoding="utf-8")
        print(f"✔ Saved generated ABC to {args.out}")
    else:
        sys.stdout.write(abc)


if __name__ == "__main__":
    _cli()
