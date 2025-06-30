#!/usr/bin/env python3
# synthetic_math_mod_pow.py · v0.1.0
"""
Generate synthetic arithmetic problems using exponentiation (^) and modulo (%) with nested expressions.

Major features
--------------
* Deterministic output with --seed
* Configurable number of problems
* Configurable max operand and max exponent values
* Configurable max nesting depth for expressions
* All results are integers
* --out to save directly to disk

Usage
-----
python synthetic_math_mod_pow.py 20
python synthetic_math_mod_pow.py 50 --seed 42 --max-operand 20 --max-exponent 4 --max-depth 3 --out problems.txt
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

__version__ = "0.1.0"

@dataclass(frozen=True)
class MathConfig:
    count: int = 20
    seed: int | None = None
    max_operand: int = 100
    max_exponent: int = 3
    max_depth: int = 2
    out: Path | None = None

OPERATORS = ['^', '%']

def generate_expr(rng: random.Random, cfg: MathConfig, depth: int = 0) -> Tuple[str, int]:
    """
    Recursively generate an integer expression using ^ and % up to cfg.max_depth.
    Returns (expr_string, integer_value).
    """
    # Base case: return a literal integer
    if depth >= cfg.max_depth or rng.random() < 0.3:
        val = rng.randint(0, cfg.max_operand)
        return str(val), val

    op = rng.choice(OPERATORS)

    # Exponentiation: ensure exponent is a small literal
    if op == '^':
        left_str, left_val = generate_expr(rng, cfg, depth + 1)
        exp = rng.randint(0, cfg.max_exponent)
        right_str, right_val = str(exp), exp
        expr = f"({left_str} ^ {right_str})"
        val = left_val ** right_val
        return expr, val

    # Modulo: both sides can be nested, but divisor must be non-zero
    left_str, left_val = generate_expr(rng, cfg, depth + 1)
    right_str, right_val = generate_expr(rng, cfg, depth + 1)
    if right_val == 0:
        # avoid zero divisor
        right_val = 1
        right_str = "1"
    expr = f"({left_str} % {right_str})"
    val = left_val % right_val
    return expr, val

def build_problems(cfg: MathConfig) -> List[str]:
    rng = random.Random(cfg.seed)
    lines: List[str] = []
    for _ in range(cfg.count):
        expr, val = generate_expr(rng, cfg)
        lines.append(f"{expr} = {val}")
    return lines

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate nested ^ and % arithmetic problems with answers.")
    p.add_argument("count", nargs="?", type=int, default=20, help="Number of problems")
    p.add_argument("--seed", type=int, help="Random seed for reproducibility")
    p.add_argument("--max-operand", type=int, default=100, help="Maximum integer operand value")
    p.add_argument("--max-exponent", type=int, default=3, help="Maximum exponent for ^")
    p.add_argument("--max-depth", type=int, default=2, help="Maximum nesting depth for expressions")
    p.add_argument("--out", type=Path, help="Path to save problems to")
    args = p.parse_args()

    cfg = MathConfig(
        count=args.count,
        seed=args.seed,
        max_operand=args.max_operand,
        max_exponent=args.max_exponent,
        max_depth=args.max_depth,
        out=args.out,
    )

    problems = build_problems(cfg)
    output = "\n".join(problems) + "\n"

    if cfg.out:
        cfg.out.parent.mkdir(parents=True, exist_ok=True)
        cfg.out.write_text(output, encoding="utf-8")
        print(f"✔ Saved {cfg.count} problems to {cfg.out}")
    else:
        sys.stdout.write(output)

if __name__ == "__main__":
    _cli()
