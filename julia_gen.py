#!/usr/bin/env python3
# synthetic_julia.py · v0.1.1
"""
Generate synthetic—yet syntactically valid—Julia source files.

Major features
--------------
* Deterministic output with --seed
* Approximate line-count control (--loc)
* Random comments, using/imports, constants, vectors/matrices, functions,
  multiple dispatch, macros, loops, simple Plots.jl calls, and @time / @assert
* Writes a runnable script that prints a function result at the end
* --out to save directly to disk
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

__version__ = "0.1.1"

# ─────────────────────────────────── configuration
@dataclass(frozen=True)
class JLConfig:
    loc:  int = 160
    seed: int | None = None
    out:  Path | None = None

# ─────────────────────────────────── helpers / state
class NameGen:
    """
    Generate fresh identifiers.  
    If all names of a given length are used, grow the length automatically.
    """
    def __init__(self, rng: random.Random) -> None:
        self.rng   = rng
        self.used  : set[str] = set()

    def fresh(self, length: int = 6) -> str:
        letters = "abcdefghijklmnopqrstuvwxyz"
        attempt_len = length
        while True:
            for _ in range(3000):
                name = "".join(self.rng.choice(letters) for _ in range(attempt_len))
                if name not in self.used:
                    self.used.add(name)
                    return name
            attempt_len += 1  # expand the namespace when exhausted

@dataclass
class JLState:
    rng:   random.Random
    names: NameGen
    funcs: List[str]

# ─────────────────────────────────── registry system
GeneratorFn = Callable[[JLState], str]
_REGISTRY: Dict[str, GeneratorFn] = {}

def register(kind: str) -> Callable[[GeneratorFn], GeneratorFn]:
    def deco(fn: GeneratorFn) -> GeneratorFn:
        if kind in _REGISTRY:
            raise ValueError(f"duplicate generator {kind}")
        _REGISTRY[kind] = fn
        return fn
    return deco

# ─────────────────────────────────── snippet builders
@register("comment")
def gen_comment(st: JLState) -> str:
    txt = "".join(st.rng.choice("abcdefghijklmnopqrstuvwxyz ")
                  for _ in range(st.rng.randint(10, 35))).strip()
    return f"# {txt}\n"

@register("using")
def gen_using(st: JLState) -> str:
    return st.rng.choice([
        "using LinearAlgebra\n",
        "using Statistics\n",
        "using Random\n",
        "using Printf\n",
    ])

@register("const")
def gen_const(st: JLState) -> str:
    name = st.names.fresh().upper()
    val  = round(st.rng.uniform(0, 3.14), 3)
    return f"const {name} = {val}\n"

@register("vector")
def gen_vector(st: JLState) -> str:
    name = st.names.fresh()
    vals = ", ".join(str(st.rng.randint(0, 50)) for _ in range(st.rng.randint(3, 7)))
    return f"{name} = [{vals}]\n"

@register("matrix")
def gen_matrix(st: JLState) -> str:
    name = st.names.fresh()
    rows = [" ".join(str(st.rng.randint(0, 9)) for _ in range(3)) for _ in range(3)]
    mat  = "; ".join(rows)
    return f"{name} = [{mat}]\n"

@register("function")
def gen_function(st: JLState) -> str:
    fname = st.names.fresh()
    st.funcs.append(fname)
    arg   = st.names.fresh()
    body  = st.rng.choice([
        f"return {arg} ^ 2",
        f"sum({arg})",
        f"mean({arg})",
        f"sort({arg})",
        f"{arg}[{arg} .> mean({arg})]",
    ])
    return f"function {fname}({arg})\n    {body}\nend\n"

@register("dispatch")
def gen_dispatch(st: JLState) -> str:
    base = st.names.fresh()
    st.funcs.append(base)
    t1, t2 = st.names.fresh().capitalize(), st.names.fresh().capitalize()
    return (
        f"{base}(x::Int)               = x + 1\n"
        f"{base}(x::AbstractVector)    = sum(x)\n"
        f"{base}(x::{t1})              = x\n"
        f"{base}(x::{t2})              = string(x)\n"
    )

@register("macro")
def gen_macro(st: JLState) -> str:
    name = st.names.fresh()
    return (
        f"macro {name}()\n"
        f"    :( println(\"Executed macro in \" * string(__module__)) )\n"
        f"end\n"
    )

@register("if_block")
def gen_if(st: JLState) -> str:
    a, b = st.rng.randint(1, 100), st.rng.randint(1, 100)
    return (
        f"if {a} > {b}\n"
        f"    @info \"{a} bigger\"\n"
        f"else\n"
        f"    @warn \"{b} bigger\"\n"
        f"end\n"
    )

@register("for_loop")
def gen_for(st: JLState) -> str:
    idx = st.names.fresh(1)
    body = st.rng.choice([f"println({idx})", f"push!(Vector{{Int}}(), {idx})"])
    return (
        f"for {idx} in 1:{st.rng.randint(3, 7)}\n"
        f"    {body}\n"
        f"end\n"
    )

@register("plot")
def gen_plot(st: JLState) -> str:
    vec = st.names.fresh()
    return (
        "using Plots\n"
        f"{vec} = rand(10)\n"
        f"scatter({vec})\n"
    )

@register("time")
def gen_time(st: JLState) -> str:
    func = st.rng.choice(st.funcs) if st.funcs else "sqrt"
    return f"@time {func}(10)\n"

# ─────────────────────────────────── builder
def build_julia(cfg: JLConfig) -> str:
    rng   = random.Random(cfg.seed)
    st    = JLState(rng=rng, names=NameGen(rng), funcs=[])
    parts: List[str] = []
    lines = 0

    kinds, weights = zip(*{
        "comment":   0.07,
        "using":     0.06,
        "const":     0.05,
        "vector":    0.10,
        "matrix":    0.07,
        "function":  0.20,
        "dispatch":  0.08,
        "macro":     0.05,
        "if_block":  0.08,
        "for_loop":  0.08,
        "plot":      0.10,
        "time":      0.06,
    }.items())

    while lines < cfg.loc:
        kind  = rng.choices(kinds, weights=weights, k=1)[0]
        chunk = _REGISTRY[kind](st)
        parts.append(chunk)
        lines += chunk.count("\n")

    # Ensure Plots is loaded if any scatter produced
    if "scatter(" in "".join(parts) and not any("using Plots" in p for p in parts):
        parts.insert(0, "using Plots\n")

    # Call one generated function at bottom
    if st.funcs:
        fn  = rng.choice(st.funcs)
        arg = "10" if rng.random() < 0.5 else "[1,2,3]"
        parts.append(f"println({fn}({arg}))\n")

    return "".join(parts)

# ─────────────────────────────────── CLI
def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic Julia code.")
    p.add_argument("loc", nargs="?", type=int, default=160,
                   help="Approximate line count")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out", type=Path, help="Path to save generated .jl")
    args = p.parse_args()

    cfg = JLConfig(loc=args.loc, seed=args.seed, out=args.out)
    code = build_julia(cfg)

    if cfg.out:
        cfg.out.parent.mkdir(parents=True, exist_ok=True)
        cfg.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved synthetic Julia script to {cfg.out}")
    else:
        sys.stdout.write(code)

if __name__ == "__main__":
    _cli()
