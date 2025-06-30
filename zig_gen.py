#!/usr/bin/env python3
# synthetic_zig.py · v0.1.0
"""
Generate synthetic—yet syntactically valid—Zig source files.

Major features
--------------
* Deterministic output with --seed
* Plugin architecture for new snippet generators
* No hidden global state
* --out to save directly to disk

Usage
-----
python synthetic_zig.py 200
python synthetic_zig.py 300 --seed 42 --out fake.zig
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

__version__ = "0.1.0"

@dataclass(frozen=True, slots=True)
class ZigConfig:
    loc: int = 200
    seed: Optional[int] = None
    weights: Dict[str, float] = field(default_factory=lambda: {
        "comment":     0.10,
        "pkg_import":  0.05,
        "var_decl":    0.25,
        "fn_def":      0.25,
        "struct":      0.15,
        "test":        0.10,
        "comptime":    0.10,
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

class NameGen:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.reserved = set()

    def fresh(self, min_len: int = 3, max_len: int = 8, capital: bool = False) -> str:
        for _ in range(10000):
            length = self.rng.randint(min_len, max_len)
            s = "".join(self.rng.choice(LETTERS) for _ in range(length))
            if capital:
                s = s.capitalize()
            if s not in self.reserved:
                self.reserved.add(s)
                return s
        raise RuntimeError("Identifier space exhausted")

def literal(rng: random.Random):
    t = rng.random()
    if t < 0.4:
        return str(rng.randint(0, 100))
    if t < 0.7:
        return f"{rng.random()*100:.2f}"
    if t < 0.85:
        return "true" if rng.random() < 0.5 else "false"
    # string
    s = "".join(rng.choice(LETTERS) for _ in range(rng.randint(3, 8)))
    return f"\"{s}\""

@register("comment")
def gen_comment(state: Dict) -> str:
    rng = state["rng"]
    tags = ["TODO", "FIXME", "NOTE", "HACK"]
    text = "".join(rng.choice(LETTERS) for _ in range(rng.randint(4, 12)))
    return f"// {rng.choice(tags)}: {text}\n"

@register("pkg_import")
def gen_pkg_import(state: Dict) -> str:
    # Zig typically imports std once
    if state["imported_std"]:
        return ""
    state["imported_std"] = True
    return "const std = @import(\"std\");\n\n"

@register("var_decl")
def gen_var_decl(state: Dict) -> str:
    rng = state["rng"]
    names = state["names"]
    symbols = state["symbols"]

    name = names.fresh()
    symbols["vars"].add(name)
    ty = rng.choice(["i32", "u32", "f64", "bool", "[]const u8"])
    val = literal(rng)
    # for slice-of-u8, wrap into .{}
    if ty == "[]const u8":
        val = f".{{ {val} }}"

    return f"var {name}: {ty} = {val};\n"

@register("fn_def")
def gen_fn_def(state: Dict) -> str:
    rng = state["rng"]
    names = state["names"]
    symbols = state["symbols"]

    name = names.fresh()
    symbols["fns"].add(name)
    ret = rng.choice(["void", "i32", "f64", "bool"])
    n = rng.randint(0, 3)
    params = []
    for _ in range(n):
        pn = names.fresh()
        pt = rng.choice(["i32", "f64", "bool"])
        params.append(f"{pn}: {pt}")
    params_str = ", ".join(params)
    body_lit = literal(rng) if ret != "void" else ""
    lines = [f"fn {name}({params_str}) -> {ret} {{\n"]
    if ret != "void":
        lines.append(f"    return {body_lit};\n")
    lines.append("}\n\n")
    return "".join(lines)

@register("struct")
def gen_struct(state: Dict) -> str:
    rng = state["rng"]
    names = state["names"]
    symbols = state["symbols"]

    cname = names.fresh(capital=True)
    symbols["structs"].add(cname)
    n = rng.randint(1, 3)
    fields = []
    for _ in range(n):
        fn = names.fresh()
        ft = rng.choice(["i32", "f64", "bool"])
        fields.append(f"    {fn}: {ft},")
    return f"const {cname} = struct {{\n" + "\n".join(fields) + "\n};\n\n"

@register("test")
def gen_test(state: Dict) -> str:
    rng = state["rng"]
    names = state["names"]
    symbols = state["symbols"]
    if not symbols["fns"]:
        return ""
    fn = rng.choice(list(symbols["fns"]))
    tn = names.fresh(capital=True)
    lit = literal(rng) if "->" not in fn else literal(rng)
    return (
        f"test \"{tn}\" {{\n"
        f"    const result = {fn}({', '.join('0' for _ in range(0))});\n"
        f"    try std.testing.expect(result == {lit});\n"
        f"}}\n\n"
    )

@register("comptime")
def gen_comptime(state: Dict) -> str:
    rng = state["rng"]
    names = state["names"]
    s = "".join(rng.choice(LETTERS) for _ in range(rng.randint(3, 8)))
    return f"comptime std.debug.print(\"{s}\\n\", .{{}});\n"

def build_zig(cfg: ZigConfig) -> str:
    rng = random.Random(cfg.seed)
    names = NameGen(rng)
    symbols = {"vars": set(), "fns": set(), "structs": set()}
    state = {
        "cfg": cfg,
        "rng": rng,
        "names": names,
        "symbols": symbols,
        "imported_std": False,
    }

    parts: List[str] = ["// Auto-generated Zig code\n\n"]
    lines = parts[0].count("\n")
    kinds, weights = zip(*cfg.weights.items())

    # ensure std import early
    parts.append(gen_pkg_import(state))
    lines += 2

    while lines < cfg.loc:
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        snippet = _REGISTRY[kind](state)
        if not snippet:
            continue
        parts.append(snippet)
        lines += snippet.count("\n")

    # optionally add main if missing
    if "main" not in symbols["fns"]:
        parts.append("pub fn main() void {}\n")
    return "".join(parts)

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate a synthetic Zig source file.")
    p.add_argument("loc", nargs="?", type=int, default=200, help="Approximate line count")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out", type=Path, help="Path to save generated Zig (.zig)")
    args = p.parse_args()

    cfg = ZigConfig(loc=args.loc, seed=args.seed)
    code = build_zig(cfg)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved generated Zig to {args.out}")
    else:
        sys.stdout.write(code)

if __name__ == "__main__":
    _cli()
