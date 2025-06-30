#!/usr/bin/env python3
# synthetic_geometry.py · v0.1.0
"""
Generate synthetic plane-geometry word problems *with* answers.

Major features
--------------
* Deterministic output with --seed
* Choose how many problems (--count) and decimal precision (--decimals)
* Problem types:  
  • Circle area / circumference  
  • Rectangle / square area & perimeter  
  • Right-triangle hypotenuse (Pythagoras)  
  • Triangle area (base-height)  
  • Trapezoid area  
* Answers are computed and rounded
* --out writes “problem = answer” lines to disk

Usage
-----
python synthetic_geometry.py 25
python synthetic_geometry.py 40 --seed 123 --decimals 3 --out geo.txt
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple, List, Dict

__version__ = "0.1.0"

# ────────────────────────────────────────── config
@dataclass(frozen=True)
class GeoConfig:
    count:    int = 25
    decimals: int = 2
    seed:     int | None = None
    out:      Path | None = None

# ────────────────────────────────────────── helpers
def rnd(val: float, d: int) -> float:
    return round(val, d)

ProblemFn = Callable[[random.Random, int], Tuple[str, str]]

def circle_area(rng: random.Random, d: int) -> Tuple[str, str]:
    r = rng.randint(1, 20)
    area = rnd(math.pi * r*r, d)
    q = f"A circle has radius {r}. What is its area?"
    return q, str(area)

def circle_circ(rng: random.Random, d: int) -> Tuple[str, str]:
    r = rng.randint(1, 20)
    c = rnd(2 * math.pi * r, d)
    q = f"Find the circumference of a circle with radius {r}."
    return q, str(c)

def rect_area(rng: random.Random, d: int) -> Tuple[str, str]:
    a, b = rng.randint(2, 30), rng.randint(2, 30)
    q = f"What is the area of a rectangle with sides {a} and {b}?"
    return q, str(rnd(a*b, d))

def rect_perim(rng: random.Random, d: int) -> Tuple[str, str]:
    a, b = rng.randint(2, 30), rng.randint(2, 30)
    q = f"Find the perimeter of a rectangle {a} by {b}."
    return q, str(rnd(2*(a+b), d))

def pythag(rng: random.Random, d: int) -> Tuple[str, str]:
    a, b = rng.randint(3, 15), rng.randint(3, 15)
    c = rnd(math.hypot(a, b), d)
    q = f"A right triangle has legs {a} and {b}. What is the length of the hypotenuse?"
    return q, str(c)

def tri_area(rng: random.Random, d: int) -> Tuple[str, str]:
    b, h = rng.randint(3, 25), rng.randint(3, 25)
    area = rnd(b*h/2, d)
    q = f"Find the area of a triangle with base {b} and height {h}."
    return q, str(area)

def trap_area(rng: random.Random, d: int) -> Tuple[str, str]:
    a, b = rng.randint(3, 20), rng.randint(3, 20)
    h = rng.randint(3, 15)
    area = rnd((a+b)/2 * h, d)
    q = f"A trapezoid has bases {a} and {b} with height {h}. What is its area?"
    return q, str(area)

# register problem generators with weights
GENS: Dict[str, Tuple[ProblemFn, float]] = {
    "circle_area": (circle_area, 0.15),
    "circle_circ": (circle_circ, 0.15),
    "rect_area":   (rect_area,   0.15),
    "rect_perim":  (rect_perim,  0.10),
    "pythag":      (pythag,      0.15),
    "tri_area":    (tri_area,    0.15),
    "trap_area":   (trap_area,   0.15),
}

def build_problems(cfg: GeoConfig) -> List[str]:
    rng = random.Random(cfg.seed)
    probs: List[str] = []
    gen_list, weights = zip(*[(fn, w) for fn, w in GENS.values()])
    for _ in range(cfg.count):
        fn = rng.choices(gen_list, weights=weights, k=1)[0]
        q, ans = fn(rng, cfg.decimals)
        probs.append(f"{q} = {ans}")
    return probs

# ────────────────────────────────────────── CLI
def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate geometry word problems with answers.")
    p.add_argument("count", nargs="?", type=int, default=25,
                   help="Number of problems")
    p.add_argument("--decimals", type=int, default=2,
                   help="Decimal places to round answers")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out", type=Path, help="Path to save problems")
    args = p.parse_args()

    cfg = GeoConfig(count=args.count, decimals=args.decimals,
                    seed=args.seed, out=args.out)
    problems = build_problems(cfg)
    output = "\n".join(problems) + "\n"

    if cfg.out:
        cfg.out.parent.mkdir(parents=True, exist_ok=True)
        cfg.out.write_text(output, encoding="utf-8")
        print(f"✔ Saved {cfg.count} geometry problems to {cfg.out}")
    else:
        sys.stdout.write(output)

if __name__ == "__main__":
    _cli()
