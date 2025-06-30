#!/usr/bin/env python3
# synthetic_gleam.py · v0.1.0
"""
Generate synthetic—yet syntactically valid—Gleam source files.

Major features
--------------
* Deterministic output with --seed
* Approximate line-count control (--loc)
* Random comments, imports, constants, lists/tuples, custom types, functions,
  case expressions, and simple io.println calls
* Writes a runnable script: pub fn main(_) that prints one function result
* --out to save directly to disk
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

__version__ = "0.1.0"

# ───────────────────────────────────────────────────────── config & helpers
@dataclass(frozen=True)
class GLConfig:
    loc:  int = 160
    seed: int | None = None
    out:  Path | None = None

class NameGen:
    """Fresh identifiers that grow in length if the pool is exhausted."""
    def __init__(self, rng: random.Random):
        self.rng  = rng
        self.used : set[str] = set()
    def fresh(self, length: int = 5, capital: bool = False) -> str:
        letters = "abcdefghijklmnopqrstuvwxyz"
        n = length
        while True:
            for _ in range(3000):
                name = "".join(self.rng.choice(letters) for _ in range(n))
                if name not in self.used:
                    self.used.add(name)
                    return name.capitalize() if capital else name
            n += 1

@dataclass
class GLState:
    rng:   random.Random
    names: NameGen
    funcs: List[str]
    uses_io: bool

# ───────────────────────────────────────────────────────── registry helpers
GeneratorFn = Callable[[GLState], str]
_REGISTRY: Dict[str, GeneratorFn] = {}
def register(kind: str) -> Callable[[GeneratorFn], GeneratorFn]:
    def deco(fn: GeneratorFn) -> GeneratorFn:
        if kind in _REGISTRY:
            raise ValueError(f"duplicate generator {kind}")
        _REGISTRY[kind] = fn
        return fn
    return deco

# ───────────────────────────────────────────────────────── snippet generators
@register("comment")
def gen_comment(st: GLState) -> str:
    txt = "".join(st.rng.choice("abcdefghijklmnopqrstuvwxyz ")
                  for _ in range(st.rng.randint(8,32))).strip()
    return f"// {txt}\n"

@register("import")
def gen_import(st: GLState) -> str:
    return st.rng.choice([
        "import gleam/list\n",
        "import gleam/int\n",
        "import gleam/string\n",
    ])

@register("const")
def gen_const(st: GLState) -> str:
    name = st.names.fresh()
    val  = st.rng.randint(0, 100)
    return f"const {name} = {val}\n"

@register("list")
def gen_list(st: GLState) -> str:
    name = st.names.fresh()
    vals = ", ".join(str(st.rng.randint(0, 20)) for _ in range(st.rng.randint(3,6)))
    return f"let {name} = [{vals}]\n"

@register("tuple")
def gen_tuple(st: GLState) -> str:
    name = st.names.fresh()
    a, b = st.rng.randint(1,9), st.rng.randint(1,9)
    return f"let {name} = {{ {a}, {b} }}\n"

@register("type")
def gen_type(st: GLState) -> str:
    tname = st.names.fresh(capital=True)
    v1    = st.names.fresh(capital=True)
    v2    = st.names.fresh(capital=True)
    return (
        f"type {tname}(a) {{\n"
        f"  {v1}\n"
        f"  {v2}(a)\n"
        f"}}\n"
    )

@register("function")
def gen_function(st: GLState) -> str:
    fname = st.names.fresh()
    st.funcs.append(fname)
    param = st.names.fresh()
    body  = st.rng.choice([
        f"{param} * 2",
        f"{param} + 1",
        f"list.sum({param})",
        f"case {param} {{ 0 -> 0 _ -> 1 }}",
    ])
    return (
        f"pub fn {fname}({param}: Int) -> Int {{\n"
        f"  {body}\n"
        f"}}\n"
    )

@register("case_fn")
def gen_case(st: GLState) -> str:
    fname = st.names.fresh()
    st.funcs.append(fname)
    param = st.names.fresh()
    return (
        f"pub fn {fname}({param}: option(option(Int))) -> Int {{\n"
        f"  case {param} {{\n"
        f"    None -> 0\n"
        f"    Some(None) -> 0\n"
        f"    Some(Some(x)) -> x\n"
        f"  }}\n"
        f"}}\n"
    )

@register("io")
def gen_io(st: GLState) -> str:
    st.uses_io = True
    msg = st.names.fresh(4)
    return f"io.println(\"{msg}\")\n"

# ───────────────────────────────────────────────────────── builder
def build_gleam(cfg: GLConfig) -> str:
    rng   = random.Random(cfg.seed)
    st    = GLState(rng=rng, names=NameGen(rng), funcs=[], uses_io=False)
    parts: List[str] = ["// Auto-generated Gleam module\n\n"]
    lines = 2

    kinds, weights = zip(*{
        "comment":  0.08,
        "import":   0.06,
        "const":    0.07,
        "list":     0.08,
        "tuple":    0.07,
        "type":     0.06,
        "function": 0.28,
        "case_fn":  0.10,
        "io":       0.20,
    }.items())

    while lines < cfg.loc:
        kind  = rng.choices(kinds, weights=weights, k=1)[0]
        chunk = _REGISTRY[kind](st)
        parts.append(chunk)
        lines += chunk.count("\n")

    # ensure io import if needed
    if st.uses_io and not any("import gleam/io" in p for p in parts):
        parts.insert(1, "import gleam/io\n")

    # add main(_) entry point
    if st.funcs:
        caller = rng.choice(st.funcs)
        parts.append("\npub fn main(_, _) {\n")
        parts.append(f"  io.println(int.to_string({caller}(5)))\n")
        parts.append("  Nil\n")
        parts.append("}\n")

    return "".join(parts)

# ───────────────────────────────────────────────────────── CLI
def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic Gleam code.")
    p.add_argument("loc", nargs="?", type=int, default=160,
                   help="Approximate line count")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out",  type=Path, help="Path to save generated .gleam")
    args = p.parse_args()

    cfg  = GLConfig(loc=args.loc, seed=args.seed, out=args.out)
    code = build_gleam(cfg)

    if cfg.out:
        cfg.out.parent.mkdir(parents=True, exist_ok=True)
        cfg.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved synthetic Gleam module to {cfg.out}")
    else:
        sys.stdout.write(code)

if __name__ == "__main__":
    _cli()
