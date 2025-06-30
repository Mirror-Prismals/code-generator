#!/usr/bin/env python3
# synthetic_color_mixing.py · v0.1.1
"""
Generate synthetic color‐mixing problems in natural language with answers.

Major features
--------------
* Deterministic output with --seed
* Configurable number of problems
* Natural‐language question templates
* Extended pigment mixing map with primaries, secondaries, tertiaries, and neutrals
* --out to save directly to disk

Usage
-----
python synthetic_color_mixing.py 10
python synthetic_color_mixing.py 20 --seed 42 --out mix.txt
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, FrozenSet

__version__ = "0.1.1"

@dataclass(frozen=True)
class MixConfig:
    count: int = 10
    seed: Optional[int] = None
    out: Path | None = None

# Extended pigment mixing map
MIX_MAP: dict[FrozenSet[str], str] = {
    # primaries → secondary
    frozenset({"red",   "blue"}):    "purple",
    frozenset({"red",   "yellow"}):  "orange",
    frozenset({"blue",  "yellow"}):  "green",
    # primary + adjacent secondary → tertiary
    frozenset({"red",    "orange"}):      "red–orange",
    frozenset({"orange", "yellow"}):      "yellow–orange",
    frozenset({"yellow", "green"}):       "yellow–green",
    frozenset({"green",  "blue"}):        "blue–green",
    frozenset({"blue",   "purple"}):      "blue–purple",
    frozenset({"purple", "red"}):         "red–purple",
    # opposing mixtures → brown/gray
    frozenset({"red",    "green"}):       "brown",
    frozenset({"blue",   "orange"}):      "brown",
    frozenset({"yellow", "purple"}):      "brown",
    # secondary + secondary blends
    frozenset({"orange",  "purple"}):     "russet",
    frozenset({"green",   "purple"}):     "slate",
    frozenset({"green",   "orange"}):     "olive",
}

# question templates
TEMPLATES = [
    "What color do you get when you mix {a} and {b}?",
    "Combine {a} with {b} to produce what color?",
    "Mixing {a} and {b} yields which color?",
    "If you blend {a} and {b}, what is the result?",
    "What is the result of mixing {a} and {b}?",
    "When {a} is mixed with {b}, what color appears?",
]

def generate_problem(rng: random.Random) -> Tuple[str, str]:
    # pick a random valid pair
    pair = rng.choice(list(MIX_MAP.keys()))
    a, b = tuple(pair)
    # randomize order in question
    if rng.random() < 0.5:
        a, b = b, a
    question = rng.choice(TEMPLATES).format(a=a, b=b)
    answer = MIX_MAP[pair]
    return question, answer

def build_problems(cfg: MixConfig) -> List[str]:
    rng = random.Random(cfg.seed)
    lines: List[str] = []
    for _ in range(cfg.count):
        q, a = generate_problem(rng)
        lines.append(f"{q} = {a}")
    return lines

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate color‐mixing problems with answers.")
    p.add_argument("count", nargs="?", type=int, default=10, help="Number of problems")
    p.add_argument("--seed", type=int, help="Random seed for reproducibility")
    p.add_argument("--out", type=Path, help="Path to save problems to")
    args = p.parse_args()

    cfg = MixConfig(count=args.count, seed=args.seed, out=args.out)
    problems = build_problems(cfg)
    output = "\n".join(problems) + "\n"

    if cfg.out:
        cfg.out.parent.mkdir(parents=True, exist_ok=True)
        cfg.out.write_text(output, encoding="utf-8")
        print(f"✔ Saved {cfg.count} color‐mixing problems to {cfg.out}")
    else:
        sys.stdout.write(output)

if __name__ == "__main__":
    _cli()
