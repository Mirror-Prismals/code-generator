#!/usr/bin/env python3
# synthetic_php.py · v0.1.0
"""
Generate synthetic—yet syntactically valid—PHP source files.

Major features
--------------
* Deterministic output with --seed
* Plugin architecture for snippet generators
* No hidden global state
* --out to save directly to disk

Usage
-----
python synthetic_php.py 200
python synthetic_php.py 300 --seed 42 --out fake.php
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

__version__ = "0.1.0"

@dataclass(frozen=True)
class PhpConfig:
    loc: int = 200
    seed: Optional[int] = None
    weights: Dict[str, float] = field(default_factory=lambda: {
        "comment":    0.10,
        "include":    0.05,
        "var":        0.15,
        "echo":       0.10,
        "function":   0.20,
        "class":      0.10,
        "if":         0.10,
        "loop":       0.10,
        "array":      0.10
    })

GeneratorFn = Callable[[Dict], str]
_REGISTRY: Dict[str, GeneratorFn] = {}

def register(kind: str) -> Callable[[GeneratorFn], GeneratorFn]:
    def inner(fn: GeneratorFn) -> GeneratorFn:
        if kind in _REGISTRY:
            raise ValueError(f"Duplicate generator: {kind}")
        _REGISTRY[kind] = fn
        return fn
    return inner

LETTERS = "abcdefghijklmnopqrstuvwxyz"
PHP_TYPES = ["int", "float", "string", "bool", "array"]

class NameGen:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.used = set()

    def fresh(self, prefix: str = "", length: int = 6) -> str:
        for _ in range(1000):
            name = "".join(self.rng.choice(LETTERS) for _ in range(length))
            if name not in self.used:
                self.used.add(name)
                return prefix + name
        raise RuntimeError("Name space exhausted")

def literal(rng: random.Random, ty: str) -> str:
    if ty == "int":
        return str(rng.randint(0, 100))
    if ty == "float":
        return f"{rng.uniform(0,100):.2f}"
    if ty == "bool":
        return rng.choice(["true", "false"])
    if ty == "string":
        s = "".join(rng.choice(LETTERS) for _ in range(rng.randint(3,8)))
        return f"\"{s}\""
    return "null"

@register("comment")
def gen_comment(state: Dict) -> str:
    rng = state["rng"]
    msgs = ["TODO", "FIXME", "NOTE", "HACK"]
    msg = rng.choice(msgs)
    return f"// {msg}: {state['names'].fresh(length=8)}\n"

@register("include")
def gen_include(state: Dict) -> str:
    rng = state["rng"]
    if state.get("included"):
        return ""
    state["included"] = True
    return "require_once 'config.php';\n\n"

@register("var")
def gen_var(state: Dict) -> str:
    rng = state["rng"]
    name = state["names"].fresh(prefix="$")
    ty = rng.choice(PHP_TYPES[:-1])
    val = literal(rng, ty)
    return f"{name} = {val};\n"

@register("echo")
def gen_echo(state: Dict) -> str:
    rng = state["rng"]
    msg = "".join(rng.choice(LETTERS) for _ in range(rng.randint(3,8)))
    return f"echo \"{msg}\\n\";\n"

@register("function")
def gen_function(state: Dict) -> str:
    rng = state["rng"]
    fname = state["names"].fresh(prefix="", length=6)
    # params
    n = rng.randint(0,2)
    params = []
    for _ in range(n):
        p = state["names"].fresh(prefix="$", length=4)
        params.append(p)
    body = ""
    # one echo or return
    if rng.random() < 0.5:
        body = f"    echo \"{fname} called\\n\";\n"
    else:
        ret = literal(rng, rng.choice(PHP_TYPES[:-1]))
        body = f"    return {ret};\n"
    return (
        f"function {fname}({', '.join(params)}) {{\n"
        f"{body}"
        "}\n\n"
    )

@register("class")
def gen_class(state: Dict) -> str:
    rng = state["rng"]
    cname = state["names"].fresh(prefix="", length=6).capitalize()
    # one method
    mname = state["names"].fresh(length=6)
    return (
        f"class {cname} {{\n"
        f"    public function {mname}() {{\n"
        f"        // method {mname}\n"
        f"        echo \"{cname}.{mname}\\n\";\n"
        f"    }}\n"
        f"}}\n\n"
    )

@register("if")
def gen_if(state: Dict) -> str:
    rng = state["rng"]
    v = "$" + state["names"].fresh(length=4)
    cond = literal(rng, "int")
    return (
        f"if ({v} > {cond}) {{\n"
        f"    echo \"{v} is greater\\n\";\n"
        f"}} else {{\n"
        f"    echo \"{v} is not greater\\n\";\n"
        f"}}\n"
    )

@register("loop")
def gen_loop(state: Dict) -> str:
    rng = state["rng"]
    v = "$" + state["names"].fresh(length=4)
    n = rng.randint(2,6)
    return (
        f"for ({v} = 0; {v} < {n}; {v}++) {{\n"
        f"    echo {v} . \"\\n\";\n"
        f"}}\n"
    )

@register("array")
def gen_array(state: Dict) -> str:
    rng = state["rng"]
    v = "$" + state["names"].fresh(length=4)
    size = rng.randint(2,5)
    elems = ", ".join(literal(rng, rng.choice(PHP_TYPES[:-1])) for _ in range(size))
    return f"{v} = array({elems});\n"

def build_php(cfg: PhpConfig) -> str:
    rng = random.Random(cfg.seed)
    names = NameGen(rng)
    state: Dict = {"rng": rng, "names": names}
    parts: List[str] = ["<?php\n\n"]
    lines = parts[0].count("\n")
    kinds, weights = zip(*cfg.weights.items())

    while lines < cfg.loc:
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        chunk = _REGISTRY[kind](state)
        if not chunk:
            continue
        parts.append(chunk)
        lines += chunk.count("\n")

    parts.append("\n?>\n")
    return "".join(parts)

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic PHP code.")
    p.add_argument("loc", nargs="?", type=int, default=200, help="Approx. line count")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out", type=Path, help="Path to save generated .php")
    args = p.parse_args()

    cfg = PhpConfig(loc=args.loc, seed=args.seed)
    code = build_php(cfg)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved generated PHP to {args.out}")
    else:
        sys.stdout.write(code)

if __name__ == "__main__":
    _cli()
