#!/usr/bin/env python3
# synthetic_ocaml.py · v0.1.2
"""
Generate synthetic—yet syntactically valid—OCaml source files.

Major features
--------------
* Deterministic output with --seed
* Approximate line count control (--loc)
* Plugin architecture for snippet generators
* Random comments, opens, let bindings, functions, modules, records, variants,
  match expressions, list operations, pipe operators, and class definitions
* Auto-generates a print_endline or print_int call at end
* --out to save directly to disk
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

__version__ = "0.1.2"

@dataclass(frozen=True)
class OcamlConfig:
    loc: int = 200
    seed: int | None = None
    out: Path | None = None

# Forward‐reference to NameGen works via PEP 563
@dataclass
class OcamlState:
    rng: random.Random
    names: NameGen
    vals: List[str]
    mods: List[str]

# Generator function type
GeneratorFn = Callable[[OcamlState], str]
_REGISTRY: Dict[str, GeneratorFn] = {}

def register(kind: str):
    """Decorator to register a snippet generator under a given key."""
    def decorator(fn: GeneratorFn) -> GeneratorFn:
        if kind in _REGISTRY:
            raise ValueError(f"Duplicate generator: {kind}")
        _REGISTRY[kind] = fn
        return fn
    return decorator

LETTERS = "abcdefghijklmnopqrstuvwxyz"
MODULE_NAMES = ["Util", "Math", "Data", "Logic", "Io", "Net", "Ui"]
TYPES = ["int", "float", "string", "bool", "unit", "int list"]

class NameGen:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.used = set()
    def fresh(self, prefix: str = "", length: int = 6, capital: bool = False) -> str:
        for _ in range(1000):
            name = prefix + "".join(self.rng.choice(LETTERS) for _ in range(length))
            if name not in self.used:
                self.used.add(name)
                return name.capitalize() if capital else name
        raise RuntimeError("Identifier space exhausted")

@register("comment")
def gen_comment(state: OcamlState) -> str:
    tags = ["(* TODO *)", "(* FIXME *)", "(* NOTE *)", "(* HACK *)"]
    return f"{state.rng.choice(tags)}\n"

@register("open")
def gen_open(state: OcamlState) -> str:
    if state.mods:
        return ""
    mod = state.rng.choice(MODULE_NAMES)
    state.mods.append(mod)
    return f"open {mod}\n"

@register("module")
def gen_module(state: OcamlState) -> str:
    name = state.names.fresh(capital=True)
    state.mods.append(name)
    inner = "".join("  " + gen_comment(state) for _ in range(2))
    return f"module {name} = struct\n{inner}end\n\n"

@register("let")
def gen_let(state: OcamlState) -> str:
    name = state.names.fresh()
    ty = state.rng.choice(TYPES)
    if ty == "int list":
        vals = [str(state.rng.randint(0,10)) for _ in range(state.rng.randint(2,5))]
        expr = "[ " + "; ".join(vals) + " ]"
    elif ty == "string":
        txt = "".join(state.rng.choice(LETTERS) for _ in range(state.rng.randint(3,8)))
        expr = f"\"{txt}\""
    elif ty == "bool":
        expr = state.rng.choice(["true","false"])
    elif ty == "float":
        expr = f"{state.rng.uniform(0.0,10.0):.2f}"
    else:
        expr = str(state.rng.randint(0,100))
    state.vals.append(name)
    return f"let {name} : {ty} = {expr}\n"

@register("function")
def gen_function(state: OcamlState) -> str:
    name = state.names.fresh()
    n = state.rng.randint(0,2)
    params = [state.names.fresh() for _ in range(n)]
    body = " + ".join(params) if n > 0 else str(state.rng.randint(0,42))
    state.vals.append(name)
    return f"let {name} {' '.join(params)} = {body}\n"

@register("match")
def gen_match(state: OcamlState) -> str:
    if not state.vals:
        return ""
    var = state.rng.choice(state.vals)
    return (
        f"match {var} with\n"
        "  | x when x mod 2 = 0 -> \"even\"\n"
        "  | _ -> \"odd\"\n\n"
    )

@register("record")
def gen_record(state: OcamlState) -> str:
    name = state.names.fresh(capital=True)
    field = state.names.fresh()
    return f"type {name} = {{ {field} : int }}\n"

@register("variant")
def gen_variant(state: OcamlState) -> str:
    name = state.names.fresh(capital=True)
    return f"type {name} = A | B of int | C of string\n"

@register("pipe")
def gen_pipe(state: OcamlState) -> str:
    if not state.vals:
        return ""
    var = state.rng.choice(state.vals)
    op = state.rng.choice([
        "List.map ((+) 1)",
        "List.filter (fun x -> x > 0)",
        "List.tl"
    ])
    return f"{var} |> {op}\n"

@register("if")
def gen_if(state: OcamlState) -> str:
    cond = state.rng.choice(["true", "false", "1 = 1", "List.length [] = 0"])
    return f"if {cond} then 1 else 0\n"

@register("class")
def gen_class(state: OcamlState) -> str:
    name = state.names.fresh(capital=True)
    field = state.names.fresh()
    return (
        f"class {name} init = object\n"
        f"  val mutable {field} = init\n"
        f"  method get = {field}\n"
        f"  method set x = {field} <- x\n"
        f"end\n\n"
    )

def build_ocaml(cfg: OcamlConfig) -> str:
    rng = random.Random(cfg.seed)
    state = OcamlState(rng=rng, names=NameGen(rng), vals=[], mods=[])
    parts: List[str] = []
    lines = 0
    kinds, weights = zip(*{
        "comment":  0.05,
        "open":     0.05,
        "module":   0.05,
        "let":      0.20,
        "function": 0.20,
        "match":    0.10,
        "record":   0.05,
        "variant":  0.05,
        "pipe":     0.10,
        "if":       0.10,
        "class":    0.10,
    }.items())

    while lines < cfg.loc:
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        snippet = _REGISTRY[kind](state)
        if snippet:
            parts.append(snippet)
            lines += snippet.count("\n")

    # final print
    if state.vals:
        v = state.rng.choice(state.vals)
        if state.rng.choice([True, False]):
            parts.append(f"print_int {v};;\n")
        else:
            parts.append(f"print_endline {v};;\n")
    return "".join(parts)

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic OCaml code.")
    p.add_argument("loc", nargs="?", type=int, default=200, help="Approx. line count")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out", type=Path, help="Path to save generated .ml")
    args = p.parse_args()

    cfg = OcamlConfig(loc=args.loc, seed=args.seed, out=args.out)
    code = build_ocaml(cfg)

    if cfg.out:
        cfg.out.parent.mkdir(parents=True, exist_ok=True)
        cfg.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved synthetic OCaml code to {cfg.out}")
    else:
        sys.stdout.write(code)

if __name__ == "__main__":
    _cli()
