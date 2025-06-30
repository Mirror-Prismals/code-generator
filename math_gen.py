#!/usr/bin/env python3
# synthetic_math.py · v0.2.0
"""
Generate synthetic arithmetic problems (addition, subtraction, multiplication,
division) with nested expressions and their answers.

Major features
--------------
* Deterministic output with --seed
* Configurable number of problems
* Configurable max operand value
* Configurable max nesting depth for expressions
* Division yields a float rounded to a given number of decimal places
* --out to save directly to disk

Usage
-----
python synthetic_math.py 20
python synthetic_math.py 50 --seed 42 --max-operand 50 --max-depth 3 --decimals 3 --out problems.txt
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

__version__ = "0.2.0"

@dataclass(frozen=True)
class MathConfig:
    count: int = 20
    seed: int | None = None
    max_operand: int = 100
    max_depth: int = 2
    decimals: int = 2
    out: Path | None = None

# supported operations
OPERATIONS = ['+', '-', '*', '/']

def generate_expr(rng: random.Random, cfg: MathConfig, depth: int = 0) -> Tuple[str, float]:
    """
    Recursively generate an arithmetic expression up to cfg.max_depth.
    Returns (expression_string, numeric_value).
    """
    # base case: simple operand
    if depth >= cfg.max_depth or rng.random() < 0.3:
        val = rng.randint(0, cfg.max_operand)
        return str(val), float(val)
    # otherwise, build a binary expression
    op = rng.choice(OPERATIONS)
    # ensure divisor non-zero
    if op == '/':
        # generate right side as non-zero at base or deeper
        right_str, right_val = generate_expr(rng, cfg, depth + 1)
        # if right_val is zero, replace with 1
        if right_val == 0:
            right_val = 1.0
            right_str = "1"
    else:
        right_str, right_val = generate_expr(rng, cfg, depth + 1)
    left_str, left_val = generate_expr(rng, cfg, depth + 1)
    expr = f"({left_str} {op} {right_str})"
    # compute value
    if op == '+':
        val = left_val + right_val
    elif op == '-':
        val = left_val - right_val
    elif op == '*':
        val = left_val * right_val
    else:  # '/'
        val = left_val / right_val
    return expr, val

def build_problems(cfg: MathConfig) -> List[str]:
    rng = random.Random(cfg.seed)
    lines: List[str] = []
    for _ in range(cfg.count):
        expr, val = generate_expr(rng, cfg)
        if '/' in expr:
            ans = f"{val:.{cfg.decimals}f}"
        else:
            # for +, -, * use integer if possible
            if val.is_integer():
                ans = str(int(val))
            else:
                ans = f"{val:.{cfg.decimals}f}"
        lines.append(f"{expr} = {ans}")
    return lines

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate nested arithmetic problems with answers.")
    p.add_argument("count", nargs="?", type=int, default=20, help="Number of problems")
    p.add_argument("--seed", type=int, help="Random seed for reproducibility")
    p.add_argument("--max-operand", type=int, default=100, help="Maximum operand value")
    p.add_argument("--max-depth", type=int, default=2, help="Maximum nesting depth for expressions")
    p.add_argument("--decimals", type=int, default=2, help="Decimal places for division results")
    p.add_argument("--out", type=Path, help="Path to save problems to")
    args = p.parse_args()

    cfg = MathConfig(
        count=args.count,
        seed=args.seed,
        max_operand=args.max_operand,
        max_depth=args.max_depth,
        decimals=args.decimals,
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
