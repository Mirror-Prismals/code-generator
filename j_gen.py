#!/usr/bin/env python3
# synthetic_j.py · v0.1.0
"""
Generate synthetic—yet syntactically valid—J source files.

Major features
--------------
* Deterministic output with --seed
* Plugin architecture for new snippet generators
* No hidden global state
* --out to save directly to disk

Usage
-----
python synthetic_j.py 200
python synthetic_j.py 300 --seed 42 --out fake.ijs
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

__version__ = "0.1.0"

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class JConfig:
    loc: int = 200                # approximate number of lines
    seed: Optional[int] = None
    weights: Dict[str, float] = field(default_factory=lambda: {
        "comment":        0.10,
        "noun_def":       0.30,
        "verb_explicit":  0.20,
        "adverb_def":     0.20,
        "iota_noun":      0.20,
    })

# ──────────────────────────────────────────────────────────────
# Registry
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

LETTERS = "abcdefghijklmnopqrstuvwxyz"

def fresh_name(rng: random.Random, length: int = 4) -> str:
    return "".join(rng.choice(LETTERS) for _ in range(length))

def literal(rng: random.Random) -> str:
    # scalar or list
    if rng.random() < 0.5:
        return str(rng.randint(0, 100))
    else:
        n = rng.randint(2, 5)
        return " ".join(str(rng.randint(0, 20)) for _ in range(n))

# ──────────────────────────────────────────────────────────────
# Generators
# ──────────────────────────────────────────────────────────────

@register("comment")
def gen_comment(state: Dict) -> str:
    rng = state["rng"]
    tags = ["NOTE", "FIXME", "TODO", "HACK"]
    text = fresh_name(rng, length=rng.randint(3, 7))
    return f"NB. {rng.choice(tags)}: {text}\n"

@register("noun_def")
def gen_noun(state: Dict) -> str:
    rng = state["rng"]
    name = fresh_name(rng)
    value = literal(rng)
    return f"{name} =. {value}\n"

@register("verb_explicit")
def gen_verb_explicit(state: Dict) -> str:
    rng = state["rng"]
    name = fresh_name(rng)
    # explicit definition: dyadic x y add
    body = rng.choice([
        "   x + y",
        "   x * y",
        "   x - y",
        "   x % y"
    ])
    return f"{name} =: 3 : 0\n{body}\n)\n"

@register("adverb_def")
def gen_adverb_def(state: Dict) -> str:
    rng = state["rng"]
    # pick a common adverb or fork
    choices = [
        ("sum", "+/"),
        ("prod", "*/"),
        ("max", ">/"),
        ("min", "</"),
        ("count", "#"),
        ("mean", "+/ % #")
    ]
    name, expr = rng.choice(choices)
    return f"{name} =. {expr}\n"

@register("iota_noun")
def gen_iota(state: Dict) -> str:
    rng = state["rng"]
    name = fresh_name(rng)
    n = rng.randint(3, 10)
    return f"{name} =. i. {n}\n"

# ──────────────────────────────────────────────────────────────
# Build & CLI
# ──────────────────────────────────────────────────────────────

def build_j(cfg: JConfig) -> str:
    rng = random.Random(cfg.seed)
    state = {"cfg": cfg, "rng": rng}
    parts: List[str] = ["NB. Auto-generated J script\n\n"]
    lines = parts[0].count("\n")
    kinds, weights = zip(*cfg.weights.items())

    while lines < cfg.loc:
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        chunk = _REGISTRY[kind](state)
        parts.append(chunk)
        lines += chunk.count("\n")

    return "".join(parts)

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate a synthetic J script.")
    p.add_argument("loc", nargs="?", type=int, default=200, help="Approximate line count")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out", type=Path, help="Path to save the generated .ijs file")
    args = p.parse_args()

    cfg = JConfig(loc=args.loc, seed=args.seed)
    code = build_j(cfg)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved generated J to {args.out}")
    else:
        sys.stdout.write(code)

if __name__ == "__main__":
    _cli()
