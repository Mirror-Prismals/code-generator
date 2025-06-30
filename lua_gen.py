#!/usr/bin/env python3
# synthetic_lua.py · v0.1.0
"""
Generate synthetic—yet syntactically valid—Lua source files.

Major features
--------------
* Deterministic output with --seed
* Approximate line-count control (--loc)
* Plugin architecture for snippet generators
* Random comments, require() calls, globals, tables, functions,
  metatable tricks, if/elseif/else, numeric & ipairs loops, and coroutine
  examples
* Calls one randomly-generated function at the bottom of the script
* --out to save directly to disk

Usage
-----
python synthetic_lua.py 150
python synthetic_lua.py 250 --seed 42 --loc 220 --out fake.lua
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

__version__ = "0.1.0"

# ───────────────────────────────────────────────── configuration
@dataclass(frozen=True)
class LuaConfig:
    loc:  int = 150
    seed: int | None = None
    out:  Path | None = None

# ───────────────────────────────────────────────── helpers / state
class NameGen:
    """Fresh identifiers that auto-grow when 1-letter names are exhausted."""
    def __init__(self, rng: random.Random) -> None:
        self.rng   = rng
        self.used  : set[str] = set()
    def fresh(self, length: int = 3) -> str:
        letters, n = "abcdefghijklmnopqrstuvwxyz", length
        while True:
            for _ in range(2000):
                name = "".join(self.rng.choice(letters) for _ in range(n))
                if name not in self.used:
                    self.used.add(name)
                    return name
            n += 1  # widen namespace

@dataclass
class LuaState:
    rng:   random.Random
    names: NameGen
    funcs: List[str]

# ───────────────────────────────────────────────── registry
GeneratorFn = Callable[[LuaState], str]
_REGISTRY: Dict[str, GeneratorFn] = {}
def register(kind: str) -> Callable[[GeneratorFn], GeneratorFn]:
    def deco(fn: GeneratorFn) -> GeneratorFn:
        if kind in _REGISTRY:
            raise ValueError(f"duplicate generator {kind}")
        _REGISTRY[kind] = fn
        return fn
    return deco

# ───────────────────────────────────────────────── snippet generators
@register("comment")
def gen_comment(st: LuaState) -> str:
    txt = "".join(st.rng.choice("abcdefghijklmnopqrstuvwxyz ") for _ in range(st.rng.randint(8,30))).strip()
    return f"-- {txt}\n"

@register("require")
def gen_require(st: LuaState) -> str:
    mod = st.rng.choice(["math", "string", "table", "coroutine"])
    alias = st.names.fresh(1)
    return f"local {alias} = require('{mod}')\n"

@register("global")
def gen_global(st: LuaState) -> str:
    name = st.names.fresh()
    val  = st.rng.randint(0, 100)
    return f"{name} = {val}\n"

@register("table")
def gen_table(st: LuaState) -> str:
    name = st.names.fresh()
    vals = ", ".join(str(st.rng.randint(0,20)) for _ in range(st.rng.randint(3,6)))
    return f"local {name} = {{ {vals} }}\n"

@register("function")
def gen_function(st: LuaState) -> str:
    fname = st.names.fresh()
    st.funcs.append(fname)
    arg   = st.names.fresh(1)
    body  = st.rng.choice([
        f"  return {arg} * 2\n",
        f"  local s = 0; for _,v in ipairs({arg}) do s=s+v end; return s\n",
        f"  return ({arg} % 2 == 0) and 'even' or 'odd'\n",
    ])
    return f"local function {fname}({arg})\n{body}end\n"

@register("if_block")
def gen_if(st: LuaState) -> str:
    a, b = st.rng.randint(1,50), st.rng.randint(1,50)
    return (
        f"if {a} > {b} then\n"
        f"  print('{a} bigger')\n"
        f"else\n"
        f"  print('{b} bigger')\n"
        f"end\n"
    )

@register("for_numeric")
def gen_for_num(st: LuaState) -> str:
    idx = st.names.fresh(1)
    n   = st.rng.randint(3,7)
    return (
        f"for {idx}=1,{n} do\n"
        f"  print({idx})\n"
        f"end\n"
    )

@register("for_ipairs")
def gen_for_ipairs(st: LuaState) -> str:
    tbl = st.names.fresh()
    return (
        f"for i,v in ipairs({tbl}) do\n"
        f"  print(i,v)\n"
        f"end\n"
    )

@register("metatable")
def gen_meta(st: LuaState) -> str:
    tbl = st.names.fresh()
    return (
        f"local {tbl} = setmetatable({{}}, {{\n"
        f"  __index = function(t,k) return k*2 end\n"
        f"}})\n"
        f"print({tbl}[10])\n"
    )

@register("coroutine")
def gen_co(st: LuaState) -> str:
    co = st.names.fresh()
    return (
        f"local {co} = coroutine.create(function()\n"
        f"  for i=1,3 do print('co', i); coroutine.yield() end\n"
        f"end)\n"
        f"coroutine.resume({co})\n"
    )

# ───────────────────────────────────────────────── builder
def build_lua(cfg: LuaConfig) -> str:
    rng   = random.Random(cfg.seed)
    st    = LuaState(rng=rng, names=NameGen(rng), funcs=[])
    parts: List[str] = []
    lines = 0

    kinds, weights = zip(*{
        "comment":     0.08,
        "require":     0.05,
        "global":      0.07,
        "table":       0.10,
        "function":    0.20,
        "if_block":    0.10,
        "for_numeric": 0.10,
        "for_ipairs":  0.08,
        "metatable":   0.07,
        "coroutine":   0.05,
    }.items())

    while lines < cfg.loc:
        kind  = rng.choices(kinds, weights=weights, k=1)[0]
        chunk = _REGISTRY[kind](st)
        parts.append(chunk)
        lines += chunk.count("\n")

    # call one generated function at bottom (if any)
    if st.funcs:
        fn  = rng.choice(st.funcs)
        arg = rng.choice(["10", "{1,2,3,4}"])
        parts.append(f"print({fn}({arg}))\n")

    return "".join(parts)

# ───────────────────────────────────────────────── CLI
def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic Lua code.")
    p.add_argument("loc", nargs="?", type=int, default=150,
                   help="Approximate line count")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out", type=Path, help="Path to save generated .lua")
    args = p.parse_args()

    cfg = LuaConfig(loc=args.loc, seed=args.seed, out=args.out)
    code = build_lua(cfg)

    if cfg.out:
        cfg.out.parent.mkdir(parents=True, exist_ok=True)
        cfg.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved synthetic Lua script to {cfg.out}")
    else:
        sys.stdout.write(code)

if __name__ == "__main__":
    _cli()
