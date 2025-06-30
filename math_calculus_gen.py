#!/usr/bin/env python3
# synthetic_calculus.py · v0.1.0
"""
Generate synthetic calculus problems (derivatives and integrals) of random polynomials.

Major features
--------------
* Deterministic output with --seed
* Configurable number of problems
* Polynomials up to a given degree with random integer coefficients
* Generates both differentiation and indefinite integration problems
* Integration answers include "+ C"
* --out to save directly to disk

Dependencies
------------
pip install sympy

Usage
-----
python synthetic_calculus.py 20
python synthetic_calculus.py 30 --seed 42 --max-degree 4 --max-coeff 10 --out calc.txt
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import sympy as sp

# symbol for calculus
x = sp.symbols('x')

@dataclass(frozen=True)
class CalcConfig:
    count: int = 20
    seed: int | None = None
    max_degree: int = 3
    max_coeff: int = 10
    out: Path | None = None

def generate_polynomial(rng: random.Random, max_degree: int, max_coeff: int) -> sp.Expr:
    # choose degree at least 1
    deg = rng.randint(1, max_degree)
    # leading coefficient non-zero
    coeffs = []
    for i in range(deg + 1):
        if i == deg:
            # non-zero leading
            c = 0
            while c == 0:
                c = rng.randint(-max_coeff, max_coeff)
        else:
            c = rng.randint(-max_coeff, max_coeff)
        coeffs.append(c)
    # build polynomial
    expr = sum(coeffs[i] * x**i for i in range(deg + 1))
    return sp.simplify(expr)

def build_problem(rng: random.Random, cfg: CalcConfig) -> str:
    expr = generate_polynomial(rng, cfg.max_degree, cfg.max_coeff)
    mode = rng.choice(['derivative', 'integral'])
    if mode == 'derivative':
        deriv = sp.diff(expr, x)
        return f"d/dx ({sp.srepr(expr) if False else expr}) = {deriv}"
    else:
        integral = sp.integrate(expr, x)
        return f"∫ ({expr}) dx = {integral} + C"

def build_problems(cfg: CalcConfig) -> List[str]:
    rng = random.Random(cfg.seed)
    problems: List[str] = []
    for _ in range(cfg.count):
        problems.append(build_problem(rng, cfg))
    return problems

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate calculus problems (derivatives & integrals).")
    p.add_argument("count", nargs="?", type=int, default=20, help="Number of problems")
    p.add_argument("--seed", type=int, help="Random seed for reproducibility")
    p.add_argument("--max-degree", type=int, default=3, help="Maximum degree of polynomial")
    p.add_argument("--max-coeff", type=int, default=10, help="Maximum absolute value of coefficients")
    p.add_argument("--out", type=Path, help="Path to save problems to")
    args = p.parse_args()

    cfg = CalcConfig(
        count=args.count,
        seed=args.seed,
        max_degree=args.max_degree,
        max_coeff=args.max_coeff,
        out=args.out,
    )

    problems = build_problems(cfg)
    output = "\n".join(problems) + "\n"

    if cfg.out:
        cfg.out.parent.mkdir(parents=True, exist_ok=True)
        cfg.out.write_text(output, encoding="utf-8")
        print(f"✔ Saved {cfg.count} calculus problems to {cfg.out}")
    else:
        sys.stdout.write(output)

if __name__ == "__main__":
    _cli()
