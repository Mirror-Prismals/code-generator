#!/usr/bin/env python3
# synthetic_go.py · v0.1.0
"""
Generate synthetic—yet syntactically valid—Go source files.

Major features
--------------
* Deterministic output with --seed
* Plugin architecture for new snippet generators
* No hidden global state
* --out to save directly to disk

Usage
-----
python synthetic_go.py 200
python synthetic_go.py 300 --seed 42 --out fake.go
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

__version__ = "0.1.0"

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class GoConfig:
    loc: int = 200
    seed: Optional[int] = None
    modules: Sequence[str] = ("fmt", "math/rand", "time", "strings")
    weights: Dict[str, float] = field(default_factory=lambda: {
        "comment":    0.05,
        "import":     0.05,
        "var_decl":   0.25,
        "function":   0.25,
        "struct":     0.15,
        "interface":  0.10,
        "slice":      0.10,
        "map":        0.05,
    })
    max_functions: Optional[int] = None
    max_structs:   Optional[int] = None
    max_interfaces: Optional[int] = None

# ──────────────────────────────────────────────────────────────
# Import manager
# ──────────────────────────────────────────────────────────────

class ImportManager:
    def __init__(self, mods: Sequence[str]) -> None:
        self._mods = list(dict.fromkeys(mods))

    def add(self, mod: str) -> None:
        if mod not in self._mods:
            self._mods.append(mod)

    def render(self) -> str:
        if not self._mods:
            return ""
        lines = ["import ("]
        for m in self._mods:
            lines.append(f'    "{m}"')
        lines.append(")\n")
        return "\n".join(lines)

# ──────────────────────────────────────────────────────────────
# Name generator
# ──────────────────────────────────────────────────────────────

class NameGenerator:
    GO_KEYWORDS = {
        "break","default","func","interface","select","case","defer","go",
        "map","struct","chan","else","goto","package","switch","const","fallthrough",
        "if","range","type","continue","for","import","return","var"
    }

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.reserved = set(self.GO_KEYWORDS)

    def fresh(self, *, min_len: int = 3, max_len: int = 8, capital: bool = False) -> str:
        letters = "abcdefghijklmnopqrstuvwxyz"
        for _ in range(10_000):
            length = self.rng.randint(min_len, max_len)
            s = "".join(self.rng.choice(letters) for _ in range(length))
            name = s.capitalize() if capital else s
            if name not in self.reserved:
                self.reserved.add(name)
                return name
        raise RuntimeError("Identifier space exhausted")

# ──────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────

GeneratorFn = Callable[[Dict], str]
_REGISTRY: Dict[str, GeneratorFn] = {}

def register(kind: str) -> Callable[[GeneratorFn], GeneratorFn]:
    def inner(fn: GeneratorFn) -> GeneratorFn:
        if kind in _REGISTRY:
            raise ValueError(f"Duplicate generator: {kind}")
        _REGISTRY[kind] = fn
        return fn
    return inner

# ──────────────────────────────────────────────────────────────
# Literal generator
# ──────────────────────────────────────────────────────────────

def gen_literal(state: Dict, depth: int = 0) -> str:
    rng = state["rng"]
    if depth >= 2 or rng.random() < 0.4:
        t = rng.random()
        if t < 0.4:
            return str(rng.randint(0, 999))
        elif t < 0.7:
            return f"{rng.uniform(0,100):.2f}"
        else:
            s = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(3,6)))
            return f'"{s}"'
    left = gen_literal(state, depth+1)
    right = gen_literal(state, depth+1)
    op = rng.choice(["+", "-", "*", "/"])
    return f"({left}{op}{right})"

# ──────────────────────────────────────────────────────────────
# Generators
# ──────────────────────────────────────────────────────────────

@register("comment")
def gen_comment(state: Dict) -> str:
    rng = state["rng"]
    if rng.random() < 0.5:
        return "// " + rng.choice([
            "TODO: refine this",
            "FIXME: edge cases",
            "NOTE: temporary",
            "HACK: cleanup later"
        ]) + "\n"
    else:
        return "/* " + rng.choice([
            "legacy", "placeholder", "wip", "debug"
        ]) + " */\n"

@register("import")
def gen_import(state: Dict) -> str:
    cfg     = state["cfg"]
    rng     = state["rng"]
    imports = state["imports"]
    mod     = rng.choice(cfg.modules)
    imports.add(mod)
    return ""  # imported at top

@register("var_decl")
def gen_var_decl(state: Dict) -> str:
    rng     = state["rng"]
    names   = state["names"]
    symbols = state["symbols"]

    vname = names.fresh()
    symbols["variables"].add(vname)
    typ = rng.choice(["int", "float64", "string", "bool"])

    lit = gen_literal(state)
    if typ == "bool":
        if rng.random() < 0.5:
            lit = "true"
        else:
            lit = "false"

    return f"var {vname} {typ} = {lit}\n"

@register("function")
def gen_function(state: Dict) -> str:
    cfg      = state["cfg"]
    rng      = state["rng"]
    names    = state["names"]
    symbols  = state["symbols"]

    if cfg.max_functions is not None and len(symbols["functions"]) >= cfg.max_functions:
        return ""
    fname = names.fresh(capital=True)
    symbols["functions"].add(fname)

    # parameters
    n = rng.randint(0,3)
    params = []
    for _ in range(n):
        pname = names.fresh()
        ptype = rng.choice(["int", "float64", "string", "bool"])
        symbols["variables"].add(pname)
        params.append(f"{pname} {ptype}")

    # return type
    rtyp = rng.choice(["int", "float64", "string", "bool", ""])
    ret = f" {rtyp}" if rtyp else ""

    header = f"func {fname}({', '.join(params)}){ret} {{\n"
    body: List[str] = []
    if rtyp:
        body.append(f"    return {gen_literal(state)}\n")
    else:
        body.append("    // no return\n")
    body.append("}\n\n")
    return header + "".join(body)

@register("struct")
def gen_struct(state: Dict) -> str:
    cfg     = state["cfg"]
    rng     = state["rng"]
    names   = state["names"]
    symbols = state["symbols"]

    if cfg.max_structs is not None and len(symbols["structs"]) >= cfg.max_structs:
        return ""
    sname = names.fresh(capital=True)
    symbols["structs"].add(sname)

    # 1–3 fields
    n = rng.randint(1,3)
    fields = []
    for _ in range(n):
        fname = names.fresh(capital=True)
        ftyp  = rng.choice(["int", "float64", "string", "bool"])
        fields.append(f"    {fname} {ftyp}")
    return f"type {sname} struct {{\n" + "\n".join(fields) + "\n}\n\n"

@register("interface")
def gen_interface(state: Dict) -> str:
    cfg     = state["cfg"]
    rng     = state["rng"]
    names   = state["names"]
    symbols = state["symbols"]

    if cfg.max_interfaces is not None and len(symbols["interfaces"]) >= cfg.max_interfaces:
        return ""
    iname = names.fresh(capital=True)
    symbols["interfaces"].add(iname)

    # single method
    mname = names.fresh(capital=True)
    return_type = rng.choice(["int", "float64", "string", "bool", ""])
    ret = f" {return_type}" if return_type else ""
    return (
        f"type {iname} interface {{\n"
        f"    {mname}(){ret}\n"
        f"}}\n\n"
    )

@register("slice")
def gen_slice(state: Dict) -> str:
    rng     = state["rng"]
    names   = state["names"]
    symbols = state["symbols"]

    name = names.fresh()
    symbols["variables"].add(name)
    etyp = rng.choice(["int", "float64", "string"])
    length = rng.randint(1,5)
    return f"{name} := make([]{etyp}, {length})\n"

@register("map")
def gen_map(state: Dict) -> str:
    rng     = state["rng"]
    names   = state["names"]
    symbols = state["symbols"]

    name = names.fresh()
    symbols["variables"].add(name)
    key   = rng.choice(["string", "int"])
    val   = rng.choice(["int", "float64", "string"])
    size  = rng.randint(1,5)
    return f"{name} := make(map[{key}]{val}, {size})\n"

# ──────────────────────────────────────────────────────────────
# Build & CLI
# ──────────────────────────────────────────────────────────────

def build_go(cfg: GoConfig) -> str:
    rng      = random.Random(cfg.seed)
    names    = NameGenerator(rng)
    imports  = ImportManager(cfg.modules)
    symbols  = {
        "variables": set(),
        "functions": set(),
        "structs":   set(),
        "interfaces": set(),
    }
    state = {
        "cfg": cfg,
        "rng": rng,
        "names": names,
        "imports": imports,
        "symbols": symbols,
    }

    parts: List[str] = [
        "package main\n\n",
        imports.render(),
    ]
    lines = sum(p.count("\n") for p in parts)
    kinds, weights = zip(*cfg.weights.items())

    while lines < cfg.loc:
        kind  = rng.choices(kinds, weights=weights, k=1)[0]
        chunk = _REGISTRY[kind](state)
        if not chunk:
            continue
        parts.append(chunk)
        lines += chunk.count("\n")

    # main function
    parts.append("func main() {\n")
    if symbols["functions"]:
        fn = rng.choice(tuple(symbols["functions"]))
        parts.append(f"    {fn}()\n")
    elif symbols["variables"]:
        v = next(iter(symbols["variables"]))
        parts.append(f'    fmt.Println({v})\n')
        imports.add("fmt")
    parts.append("}\n")

    return "".join(parts)


def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate a synthetic Go source file.")
    p.add_argument("loc", nargs="?", type=int, default=200, help="Approx. line count")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--max-funcs", type=int, help="Max functions")
    p.add_argument("--max-structs", type=int, help="Max structs")
    p.add_argument("--max-interfaces", type=int, help="Max interfaces")
    p.add_argument("--out", type=Path, help="Path to save generated code")
    args = p.parse_args()

    cfg = GoConfig(
        loc=args.loc,
        seed=args.seed,
        max_functions=args.max_funcs,
        max_structs=args.max_structs,
        max_interfaces=args.max_interfaces,
    )
    code = build_go(cfg)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved generated Go to {args.out}")
    else:
        sys.stdout.write(code)


if __name__ == "__main__":
    _cli()
