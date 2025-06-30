#!/usr/bin/env python3
# synthetic_matrix_mul.py · v0.1.0
"""
Generate synthetic matrix multiplication problems with solutions.

Major features
--------------
* Deterministic output with --seed
* Configurable number of problems
* Random matrix dimensions between min_size and max_size
* Random integer entries between min_val and max_val
* --out to save directly to disk

Usage
-----
python synthetic_matrix_mul.py 10
python synthetic_matrix_mul.py 20 --seed 42 --min-size 2 --max-size 5 --min-val -10 --max-val 10 --out mats.txt
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

__version__ = "0.1.0"

@dataclass(frozen=True)
class MatrixMulConfig:
    count: int = 10
    seed: int | None = None
    min_size: int = 2
    max_size: int = 4
    min_val: int = -5
    max_val: int = 5
    out: Path | None = None

Matrix = List[List[int]]

def gen_matrix(rng: random.Random, rows: int, cols: int, min_val: int, max_val: int) -> Matrix:
    return [
        [rng.randint(min_val, max_val) for _ in range(cols)]
        for _ in range(rows)
    ]

def mat_mult(A: Matrix, B: Matrix) -> Matrix:
    rows, common, cols = len(A), len(B), len(B[0])
    return [
        [
            sum(A[i][k] * B[k][j] for k in range(common))
            for j in range(cols)
        ]
        for i in range(rows)
    ]

def build_problems(cfg: MatrixMulConfig) -> List[str]:
    rng = random.Random(cfg.seed)
    lines: List[str] = []
    for _ in range(cfg.count):
        # choose dimensions
        rA = rng.randint(cfg.min_size, cfg.max_size)
        cA = rng.randint(cfg.min_size, cfg.max_size)
        rB = cA
        cB = rng.randint(cfg.min_size, cfg.max_size)
        A = gen_matrix(rng, rA, cA, cfg.min_val, cfg.max_val)
        B = gen_matrix(rng, rB, cB, cfg.min_val, cfg.max_val)
        C = mat_mult(A, B)
        lines.append(f"A = {A}")
        lines.append(f"B = {B}")
        lines.append(f"A × B = {C}")
        lines.append("")  # blank line between problems
    return lines

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate matrix multiplication problems with answers.")
    p.add_argument("count", nargs="?", type=int, default=10, help="Number of problems")
    p.add_argument("--seed", type=int, help="Random seed for reproducibility")
    p.add_argument("--min-size", type=int, default=2, help="Minimum dimension size")
    p.add_argument("--max-size", type=int, default=4, help="Maximum dimension size")
    p.add_argument("--min-val", type=int, default=-5, help="Minimum matrix entry")
    p.add_argument("--max-val", type=int, default=5, help="Maximum matrix entry")
    p.add_argument("--out", type=Path, help="Path to save problems to")
    args = p.parse_args()

    cfg = MatrixMulConfig(
        count=args.count,
        seed=args.seed,
        min_size=args.min_size,
        max_size=args.max_size,
        min_val=args.min_val,
        max_val=args.max_val,
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
