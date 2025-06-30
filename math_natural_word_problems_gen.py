#!/usr/bin/env python3
# synthetic_word_problems.py · v0.1.1
"""
Generate synthetic natural‐language subtraction word problems with answers.

Major features
--------------
* Deterministic output with --seed
* Configurable number of problems
* Random names, objects, and transfer counts
* Supports multiple recipients per problem
* --out to save directly to disk

Usage
-----
python synthetic_word_problems.py 10
python synthetic_word_problems.py 20 --seed 42 --out problems.txt
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

__version__ = "0.1.1"

@dataclass(frozen=True)
class WPConfig:
    count: int = 10
    seed: int | None = None
    out: Path | None = None

# sample proper names
NAMES = [
    "Johnny", "Sarah", "Tom", "Alice", "Bob", "Emily", "Lucas", "Mia",
    "Ethan", "Olivia", "Liam", "Ava", "Noah", "Isabella", "Mason", "Sophia",
    "Logan", "Charlotte", "Jacob", "Amelia"
]

# objects to count
OBJECTS = [
    "apples", "oranges", "marbles", "cookies", "books", "stickers",
    "pens", "tickets", "candies", "balls"
]

# templates (use pronoun_lower instead of pronoun.lower())
TEMPLATES = [
    "{name} has {start} {obj}. {pronoun} gives {r1_count} to {r1} and {r2_count} to {r2}. How many {obj} does {pronoun_lower} have left?",
    "{name} started with {start} {obj}. {pronoun} gave {r1_count} to {r1}, then gave {r2_count} to {r2}. How many {obj} remain with {pronoun_lower}?",
    "If {name} has {start} {obj} and gives {r1_count} to {r1} and {r2_count} to {r2}, how many {obj} are left with {pronoun_lower}?",
    "{name} owned {start} {obj}. After giving {r1_count} to {r1} and {r2_count} to {r2}, how many {obj} does {pronoun_lower} have?"
]

def generate_problem(rng: random.Random) -> Tuple[str, str]:
    name = rng.choice(NAMES)
    # pick pronoun and its lowercase form
    if name in {"Johnny","Tom","Bob","Lucas","Ethan","Liam","Noah","Mason","Logan","Jacob"}:
        pronoun = "He"
    else:
        pronoun = "She"
    pronoun_lower = pronoun.lower()
    obj = rng.choice(OBJECTS)
    start = rng.randint(5, 20)
    # choose two distinct recipients
    recipients = rng.sample([n for n in NAMES if n != name], 2)
    # choose two positive counts summing to <= start
    r1_count = rng.randint(1, start - 1)
    r2_count = rng.randint(1, start - r1_count)
    remaining = start - (r1_count + r2_count)

    tmpl = rng.choice(TEMPLATES)
    question = tmpl.format(
        name=name,
        start=start,
        obj=obj,
        pronoun=pronoun,
        pronoun_lower=pronoun_lower,
        r1=recipients[0],
        r1_count=r1_count,
        r2=recipients[1],
        r2_count=r2_count
    )
    answer = str(remaining)
    return question, answer

def build_problems(cfg: WPConfig) -> List[str]:
    rng = random.Random(cfg.seed)
    lines: List[str] = []
    for _ in range(cfg.count):
        q, a = generate_problem(rng)
        lines.append(f"{q} = {a}")
    return lines

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate subtraction word problems with answers.")
    p.add_argument("count", nargs="?", type=int, default=10, help="Number of problems")
    p.add_argument("--seed", type=int, help="Random seed for reproducibility")
    p.add_argument("--out", type=Path, help="Path to save problems to")
    args = p.parse_args()

    cfg = WPConfig(count=args.count, seed=args.seed, out=args.out)
    problems = build_problems(cfg)
    output = "\n".join(problems) + "\n"

    if cfg.out:
        cfg.out.parent.mkdir(parents=True, exist_ok=True)
        cfg.out.write_text(output, encoding="utf-8")
        print(f"✔ Saved {cfg.count} word problems to {cfg.out}")
    else:
        sys.stdout.write(output)

if __name__ == "__main__":
    _cli()
