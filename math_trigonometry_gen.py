#!/usr/bin/env python3
# synthetic_trigonometry.py · v0.1.0
"""
Generate synthetic trigonometry word-problems *with* answers.

Problem types
-------------
✓ Evaluate sin / cos / tan for common angles  
✓ Convert degrees ↔ radians  
✓ Find an angle given a sine / cosine value  
✓ Right-triangle missing side (SOH-CAH-TOA)  
✓ Law of cosines (find third side)

Features
--------
* Deterministic with --seed
* Choose number of problems (--count) and answer precision (--decimals)
* All answers rounded
* --out writes “problem = answer” lines to disk
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Callable, Dict

__version__ = "0.1.0"

# ────────────────────────────────────────── configuration
@dataclass(frozen=True)
class TrigConfig:
    count:    int = 25
    decimals: int = 3
    seed:     int | None = None
    out:      Path | None = None

# ────────────────────────────────────────── helpers
def rnd(x: float, d: int) -> float:
    return round(x, d)

def deg2rad(deg: float) -> float:
    return math.radians(deg)

def rad2deg(rad: float) -> float:
    return math.degrees(rad)

# ────────────────────────────────────────── problem generators
ProblemFn = Callable[[random.Random, int], Tuple[str, str]]

# 1–3 evaluate trig at “nice” angles
ANGLES = [0, 30, 45, 60, 90, 120, 135, 150, 180]

def sin_val(rng: random.Random, d: int) -> Tuple[str, str]:
    a = rng.choice(ANGLES)
    q = f"Evaluate sin({a}°)."
    return q, str(rnd(math.sin(deg2rad(a)), d))

def cos_val(rng: random.Random, d: int) -> Tuple[str, str]:
    a = rng.choice(ANGLES)
    q = f"Evaluate cos({a}°)."
    return q, str(rnd(math.cos(deg2rad(a)), d))

def tan_val(rng: random.Random, d: int) -> Tuple[str, str]:
    a = rng.choice([ang for ang in ANGLES if (ang % 180) != 90])  # avoid undefined
    q = f"Evaluate tan({a}°)."
    return q, str(rnd(math.tan(deg2rad(a)), d))

# 4–5 conversion
def to_radians(rng: random.Random, d: int) -> Tuple[str, str]:
    a = rng.randint(5, 355)
    q = f"Convert {a} degrees to radians."
    return q, str(rnd(deg2rad(a), d))

def to_degrees(rng: random.Random, d: int) -> Tuple[str, str]:
    r = rnd(rng.uniform(0.1, math.pi*2), d+2)
    q = f"Convert {r} radians to degrees."
    return q, str(rnd(rad2deg(r), d))

# 6 find angle given sin / cos
SPECIAL = {30:0.5, 45:math.sqrt(2)/2, 60:math.sqrt(3)/2}

def angle_from_sin(rng: random.Random, d:int)->Tuple[str,str]:
    ang = rng.choice(list(SPECIAL.keys()))
    val = rnd(SPECIAL[ang], d+1)
    q = f"Find an angle (0°–90°) whose sine is {val}."
    return q, str(ang)

def angle_from_cos(rng:random.Random,d:int)->Tuple[str,str]:
    ang = rng.choice(list(SPECIAL.keys()))
    val = rnd(SPECIAL[ang], d+1)
    q = f"Find an angle (0°–90°) whose cosine is {val}."
    return q, str(90-ang)

# 7 right-triangle missing side using SOH/CAH/TOA
def right_triangle_side(rng: random.Random, d:int)->Tuple[str,str]:
    angle = rng.choice([30,45,60])
    hyp   = rng.randint(5,25)
    opp   = rnd(hyp*math.sin(deg2rad(angle)), d)
    q = (f"In a right triangle, the hypotenuse is {hyp} and one acute angle "
         f"is {angle}°. Find the length of the side opposite that angle.")
    return q, str(opp)

# 8 law of cosines
def law_of_cosines(rng: random.Random, d:int)->Tuple[str,str]:
    a, b = rng.randint(5,20), rng.randint(5,20)
    Cdeg = rng.choice([40,60,80,100,120])
    c = rnd(math.sqrt(a*a + b*b - 2*a*b*math.cos(deg2rad(Cdeg))), d)
    q = (f"Triangle sides a={a}, b={b} with included angle γ={Cdeg}°. "
         f"Find side c using the law of cosines.")
    return q, str(c)

# weighted registry
GENS: Dict[str, Tuple[ProblemFn, float]] = {
    "sin":   (sin_val,           0.10),
    "cos":   (cos_val,           0.10),
    "tan":   (tan_val,           0.08),
    "rad":   (to_radians,        0.10),
    "deg":   (to_degrees,        0.10),
    "asin":  (angle_from_sin,    0.12),
    "acos":  (angle_from_cos,    0.12),
    "rt":    (right_triangle_side,0.18),
    "lawcos":(law_of_cosines,    0.10),
}

# ────────────────────────────────────────── generator
def build_problems(cfg: TrigConfig) -> List[str]:
    rng = random.Random(cfg.seed)
    problems: List[str] = []
    gen_list, weights = zip(*[(fn, w) for fn, w in GENS.values()])
    for _ in range(cfg.count):
        fn = rng.choices(gen_list, weights=weights, k=1)[0]
        q, ans = fn(rng, cfg.decimals)
        problems.append(f"{q} = {ans}")
    return problems

# ────────────────────────────────────────── CLI
def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate trigonometry problems with answers.")
    p.add_argument("count", nargs="?", type=int, default=25,
                   help="Number of problems")
    p.add_argument("--decimals", type=int, default=3,
                   help="Decimal places for answers")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out", type=Path, help="Output file path")
    args = p.parse_args()

    cfg = TrigConfig(count=args.count, decimals=args.decimals,
                     seed=args.seed, out=args.out)
    probs = build_problems(cfg)
    output = "\n".join(probs) + "\n"

    if cfg.out:
        cfg.out.parent.mkdir(parents=True, exist_ok=True)
        cfg.out.write_text(output, encoding="utf-8")
        print(f"✔ Saved {cfg.count} trigonometry problems to {cfg.out}")
    else:
        sys.stdout.write(output)

if __name__ == "__main__":
    _cli()
