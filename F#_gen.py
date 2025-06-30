#!/usr/bin/env python3
# synthetic_fsharp.py · v0.1.1
"""
Generate synthetic—yet syntactically valid—F# source files.

Major features
--------------
* Deterministic output with --seed
* Approximate line count control (--loc)
* Plugin architecture for snippet generators
* Random comments, opens, let bindings, functions, modules, records, discriminated unions,
  list expressions, piping, match expressions, and class definitions
* --out to save directly to disk

Usage
-----
python synthetic_fsharp.py 200
python synthetic_fsharp.py 300 --seed 42 --out Fake.fs
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List

__version__ = "0.1.1"

@dataclass(frozen=True)
class FsConfig:
    loc: int = 200
    seed: int | None = None
    out: Path | None = None

GeneratorFn = Callable[[dict], str]
_REGISTRY: Dict[str, GeneratorFn] = {}

def register(kind: str):
    def decorator(fn: GeneratorFn) -> GeneratorFn:
        if kind in _REGISTRY:
            raise ValueError(f"Duplicate generator: {kind}")
        _REGISTRY[kind] = fn
        return fn
    return decorator

LETTERS = "abcdefghijklmnopqrstuvwxyz"
MODULE_NAMES = ["Utils", "Math", "Data", "Logic", "IO", "Net", "UI"]
TYPES = ["int", "float", "string", "bool", "list<int>", "unit"]

class NameGen:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.reserved = set()
    def fresh(self, prefix: str = "", length: int = 6, capital: bool = False) -> str:
        for _ in range(1000):
            name = prefix + "".join(self.rng.choice(LETTERS) for _ in range(length))
            if name not in self.reserved:
                self.reserved.add(name)
                return name.capitalize() if capital else name
        raise RuntimeError("Identifier space exhausted")

@register("comment")
def gen_comment(state: dict) -> str:
    rng = state["rng"]
    tags = ["// TODO", "// FIXME", "// NOTE", "// HACK"]
    return f"{rng.choice(tags)}: {state['names'].fresh()}\n"

@register("open")
def gen_open(state: dict) -> str:
    rng = state["rng"]
    if state.get("opened"):
        return ""
    state["opened"] = True
    mod = rng.choice(MODULE_NAMES)
    return f"open System.{mod}\n"

@register("module")
def gen_module(state: dict) -> str:
    if state.get("moduled"):
        return ""
    state["moduled"] = True
    rng = state["rng"]
    name = rng.choice(MODULE_NAMES)
    state["module_name"] = name
    return f"module {name}\n\n"

@register("let")
def gen_let(state: dict) -> str:
    rng = state["rng"]
    name = state["names"].fresh()
    ty = rng.choice(TYPES)
    if ty == "list<int>":
        vals = ", ".join(str(rng.randint(0,10)) for _ in range(rng.randint(2,5)))
        expr = f"[{vals}]"
    elif ty == "string":
        txt = "".join(rng.choice(LETTERS) for _ in range(rng.randint(3,8)))
        expr = f"\"{txt}\""
    elif ty == "bool":
        expr = rng.choice(["true","false"])
    else:
        expr = str(rng.randint(0,100))
    return f"let {name} : {ty} = {expr}\n"

@register("function")
def gen_function(state: dict) -> str:
    rng = state["rng"]
    name = state["names"].fresh()
    n_params = rng.randint(0, 2)
    params = [state["names"].fresh() for _ in range(n_params)]
    body = ""
    if n_params >= 1:
        body = " + ".join(params)
    else:
        body = str(rng.randint(0,100))
    sig = f"({', '.join(params)})" if params else ""
    return f"let {name} {sig} = {body}\n"

@register("pipe")
def gen_pipe(state: dict) -> str:
    rng = state["rng"]
    x = rng.randint(1,10)
    ops = " |> ".join(state["names"].fresh() for _ in range(rng.randint(1,3)))
    return f"{x} |> {ops}\n"

@register("match")
def gen_match(state: dict) -> str:
    rng = state["rng"]
    val = rng.choice([str(rng.randint(0,3)), "None", "Some 5"])
    cases = [
        "| Some x -> printfn \"%d\" x",
        "| None   -> printfn \"none\""
    ]
    return f"match {val} with\n    {cases[0]}\n    {cases[1]}\n\n"

@register("record")
def gen_record(state: dict) -> str:
    rng = state["rng"]
    rec = state["names"].fresh(capital=True)
    field = state["names"].fresh()
    return f"type {rec} = {{ {field} : int }}\n"

@register("union")
def gen_union(state: dict) -> str:
    rng = state["rng"]
    du = state["names"].fresh(capital=True)
    cases = ["A", "B", "C"]
    cases_str = " | ".join(cases)
    return f"type {du} = {cases_str}\n"

@register("class")
def gen_class(state: dict) -> str:
    rng = state["rng"]
    cls = state["names"].fresh(capital=True)
    member = state["names"].fresh()
    return (
        f"type {cls}() =\n"
        f"    member this.{member}() = printfn \"{member} called\"\n\n"
    )

def build_fsharp(cfg: FsConfig) -> str:
    rng = random.Random(cfg.seed)
    names = NameGen(rng)
    state = {"rng": rng, "names": names}
    parts: List[str] = []
    lines = 0
    kinds, weights = zip(*{
        "comment":  0.10,
        "open":     0.05,
        "module":   0.05,
        "let":      0.20,
        "function": 0.20,
        "pipe":     0.10,
        "match":    0.10,
        "record":   0.05,
        "union":    0.05,
        "class":    0.10
    }.items())

    while lines < cfg.loc:
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        snippet = _REGISTRY[kind](state)
        if not snippet:
            continue
        parts.append(snippet)
        lines += snippet.count("\n")

    return "".join(parts)

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic F# code.")
    p.add_argument("loc", nargs="?", type=int, default=200, help="Approx. line count")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out", type=Path, help="Path to save generated .fs")
    args = p.parse_args()

    cfg = FsConfig(loc=args.loc, seed=args.seed, out=args.out)
    code = build_fsharp(cfg)

    if cfg.out:
        cfg.out.parent.mkdir(parents=True, exist_ok=True)
        cfg.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved synthetic F# code to {cfg.out}")
    else:
        sys.stdout.write(code)

if __name__ == "__main__":
    _cli()
