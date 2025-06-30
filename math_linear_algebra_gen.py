#!/usr/bin/env python3
# synthetic_linear_algebra.py · v0.1.1
"""
Generate synthetic linear algebra problems (vectors and matrices) with solutions.

Major features
--------------
* Deterministic output with --seed
* Configurable number of problems
* Supports:
    - Vector addition, subtraction, dot product
    - Matrix addition, multiplication (always square)
    - Matrix determinant (2x2, 3x3)
    - Matrix inverse (2x2 only when invertible)
    - Solve 2×2 linear systems (only when system is non‐singular)
* Results formatted as integers or floats with fixed decimals
* --out to save directly to disk
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

__version__ = "0.1.1"

@dataclass(frozen=True)
class LAConfig:
    count: int = 20
    seed: int | None = None
    decimals: int = 2
    out: Path | None = None

Vector = List[int]
Matrix = List[List[int]]

def fmt_num(x: float, decimals: int) -> str:
    if abs(x - round(x)) < 10**-decimals:
        return str(int(round(x)))
    return f"{x:.{decimals}f}"

def vec_add(a: Vector, b: Vector) -> Vector:
    return [x + y for x, y in zip(a, b)]

def vec_sub(a: Vector, b: Vector) -> Vector:
    return [x - y for x, y in zip(a, b)]

def vec_dot(a: Vector, b: Vector) -> int:
    return sum(x * y for x, y in zip(a, b))

def mat_add(A: Matrix, B: Matrix) -> Matrix:
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]

def mat_mult(A: Matrix, B: Matrix) -> Matrix:
    n = len(A)
    return [
        [
            sum(A[i][k] * B[k][j] for k in range(n))
            for j in range(n)
        ]
        for i in range(n)
    ]

def det2(A: Matrix) -> int:
    return A[0][0]*A[1][1] - A[0][1]*A[1][0]

def det3(A: Matrix) -> int:
    return (
        A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
      - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
      + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0])
    )

def inv2(A: Matrix) -> Union[List[List[float]], None]:
    d = det2(A)
    if d == 0:
        return None
    return [
        [ A[1][1]/d, -A[0][1]/d],
        [-A[1][0]/d,  A[0][0]/d],
    ]

def solve2(A: Matrix, b: Vector) -> Union[List[float], None]:
    d = det2(A)
    if d == 0:
        return None
    x = det2([[b[0], A[0][1]], [b[1], A[1][1]]]) / d
    y = det2([[A[0][0], b[0]], [A[1][0], b[1]]]) / d
    return [x, y]

def gen_vector(rng: random.Random) -> Vector:
    dim = rng.choice([2, 3])
    return [rng.randint(-10, 10) for _ in range(dim)]

def gen_matrix(rng: random.Random) -> Matrix:
    n = rng.choice([2, 3])
    return [[rng.randint(-5, 5) for _ in range(n)] for _ in range(n)]

def gen_square_matrix(rng: random.Random, n: int) -> Matrix:
    return [[rng.randint(-5, 5) for _ in range(n)] for _ in range(n)]

def build_problem(rng: random.Random, decimals: int) -> str:
    kind = rng.choice([
        "vec_add", "vec_sub", "vec_dot",
        "mat_add", "mat_mult", "determinant",
        "inverse", "solve"
    ])

    if kind.startswith("vec"):
        a = gen_vector(rng)
        b = gen_vector(rng)[:len(a)]
        if kind == "vec_add":
            return f"{a} + {b} = {vec_add(a,b)}"
        if kind == "vec_sub":
            return f"{a} - {b} = {vec_sub(a,b)}"
        return f"{a} · {b} = {vec_dot(a,b)}"

    # matrix operations
    A = gen_matrix(rng)
    n = len(A)

    if kind == "mat_add":
        B = gen_square_matrix(rng, n)
        return f"{A} + {B} = {mat_add(A,B)}"

    if kind == "mat_mult":
        B = gen_square_matrix(rng, n)
        return f"{A} × {B} = {mat_mult(A,B)}"

    if kind == "determinant":
        d = det2(A) if n==2 else det3(A)
        return f"det({A}) = {d}"

    if kind == "inverse":
        if n != 2:
            d = det3(A)
            return f"det({A}) = {d}"
        inv = inv2(A)
        if inv is None:
            return f"det({A}) = 0 (not invertible)"
        inv_fmt = [[fmt_num(x,decimals) for x in row] for row in inv]
        return f"{A}⁻¹ = {inv_fmt}"

    # solve 2×2 system
    if kind == "solve":
        if n != 2:
            d = det3(A)
            return f"det({A}) = {d}"
        b = gen_vector(rng)[:2]
        sol = solve2(A, b)
        if sol is None:
            return f"det({A}) = 0 (no unique solution)"
        sol_fmt = [fmt_num(x,decimals) for x in sol]
        return f"Solve {A}·x = {b} ⇒ x = {sol_fmt}"

    # fallback
    return "?"

def build_problems(cfg: LAConfig) -> List[str]:
    rng = random.Random(cfg.seed)
    return [build_problem(rng, cfg.decimals) for _ in range(cfg.count)]

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate linear algebra problems with answers.")
    p.add_argument("count", nargs="?", type=int, default=20, help="Number of problems")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--decimals", type=int, default=2, help="Decimal places for inverses/solutions")
    p.add_argument("--out", type=Path, help="Path to save problems to")
    args = p.parse_args()

    cfg = LAConfig(count=args.count, seed=args.seed, decimals=args.decimals, out=args.out)
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
