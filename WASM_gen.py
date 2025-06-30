#!/usr/bin/env python3
# synthetic_wasm.py · v0.1.0
"""
Generate synthetic WebAssembly (WAT) modules with random imports, globals, memory,
functions, exports, and data segments.

Major features
--------------
* Deterministic output with --seed
* Plugin architecture for snippet generators
* Configurable approximate number of lines (--loc)
* Randomized types, imports, globals, memory, functions, exports, and data
* --out to save directly to disk

Usage
-----
python synthetic_wasm.py 200
python synthetic_wasm.py 300 --seed 42 --out fake.wat
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List

__version__ = "0.1.0"

@dataclass(frozen=True)
class WasmConfig:
    loc: int = 200                  # approximate number of lines
    seed: int | None = None

GeneratorFn = Callable[["WasmState"], str]
_REGISTRY: Dict[str, GeneratorFn] = {}

def register(kind: str):
    def decorator(fn: GeneratorFn) -> GeneratorFn:
        if kind in _REGISTRY:
            raise ValueError(f"Duplicate generator: {kind}")
        _REGISTRY[kind] = fn
        return fn
    return decorator

class NameGen:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.counts: Dict[str,int] = {}

    def fresh(self, prefix: str) -> str:
        n = self.counts.get(prefix, 0)
        self.counts[prefix] = n + 1
        return f"${prefix}{n}"

@dataclass
class WasmState:
    rng: random.Random
    names: NameGen
    imported: bool = False
    global_count: int = 0
    funcs: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    data_count: int = 0

@register("comment")
def gen_comment(state: WasmState) -> str:
    msg = "".join(state.rng.choice("abcdefghijklmnopqrstuvwxyz ") for _ in range(state.rng.randint(10,40)))
    return f";; {msg}\n"

@register("import")
def gen_import(state: WasmState) -> str:
    if state.imported:
        return ""
    state.imported = True
    # generate a single import of a logging function
    return '(import "env" "log" (func $log (param i32)))\n'

@register("global")
def gen_global(state: WasmState) -> str:
    idx = state.global_count
    state.global_count += 1
    name = state.names.fresh("g")
    # mutable global initialized to a random constant
    val = state.rng.randint(0, 100)
    return f"(global {name} (mut i32) (i32.const {val}))\n"

@register("memory")
def gen_memory(state: WasmState) -> str:
    # declare a single page memory
    return "(memory $mem 1)\n"

@register("func_type")
def gen_func_type(state: WasmState) -> str:
    # no explicit types needed in WAT text, skip
    return ""

_instrs = ["i32.add", "i32.sub", "i32.mul", "i32.div_s"]

@register("func")
def gen_func(state: WasmState) -> str:
    # generate one function with random locals and instructions
    name = state.names.fresh("f")
    state.funcs.append(name)
    # params and locals
    n_params = state.rng.randint(0, 2)
    n_locals = state.rng.randint(0, 2)
    params = " ".join(f"(param ${name}p{i} i32)" for i in range(n_params))
    result = "(result i32)"
    locals_ = " ".join(f"(local $l{i} i32)" for i in range(n_locals))
    # body: a sequence of get_local/get_global/const and one binary op
    body: List[str] = []
    # init locals
    for i in range(n_locals):
        val = state.rng.randint(0, 10)
        body.append(f"    i32.const {val}")
        body.append(f"    set_local $l{i}")
    # push params/globals
    if n_params > 0:
        body.append(f"    get_local ${name}p0")
    elif state.global_count > 0:
        gidx = state.rng.randrange(state.global_count)
        body.append(f"    get_global $g{gidx}")
    else:
        body.append(f"    i32.const {state.rng.randint(0,10)}")
    # add an operation
    op = state.rng.choice(_instrs)
    body.append(f"    {op}")
    # return top of stack
    body_text = "\n".join(body)
    func = f"(func {name} {params} {result} {locals_}\n{body_text}\n)\n"
    return func

@register("export")
def gen_export(state: WasmState) -> str:
    # export one of the generated functions
    if not state.funcs or state.exports:
        return ""
    name = state.funcs[0]
    state.exports.append(name)
    return f'(export "run" (func {name}))\n'

@register("data")
def gen_data(state: WasmState) -> str:
    # one data segment with a short string
    if state.data_count > 0:
        return ""
    state.data_count += 1
    text = "".join(state.rng.choice("ABCDEF ") for _ in range(16))
    return f'(data (i32.const 0) "{text}")\n'

def build_wasm(cfg: WasmConfig) -> str:
    rng = random.Random(cfg.seed)
    state = WasmState(rng=rng, names=NameGen(rng))
    parts: List[str] = ["(module\n"]
    lines = 1
    kinds, weights = zip(*{
        "comment":   0.05,
        "import":    0.10,
        "global":    0.10,
        "memory":    0.10,
        "func":      0.40,
        "export":    0.10,
        "data":      0.15,
    }.items())

    while lines < cfg.loc:
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        snippet = _REGISTRY[kind](state)
        if not snippet:
            continue
        # indent all lines inside module
        snippet = "".join("  "+line for line in snippet.splitlines(True))
        parts.append(snippet)
        lines += snippet.count("\n")
    parts.append(")\n")
    return "".join(parts)

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic WebAssembly (WAT) modules.")
    p.add_argument("loc", nargs="?", type=int, default=200, help="Approx. line count")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out", type=Path, help="Path to save generated .wat")
    args = p.parse_args()

    cfg = WasmConfig(loc=args.loc, seed=args.seed)
    wat = build_wasm(cfg)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(wat, encoding="utf-8")
        print(f"✔ Saved synthetic WebAssembly to {args.out}")
    else:
        sys.stdout.write(wat)

if __name__ == "__main__":
    _cli()
