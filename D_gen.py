#!/usr/bin/env python3
# synthetic_d.py · v0.1.1
"""
Generate synthetic—yet syntactically valid—D source files.

Major features
--------------
* Deterministic output with --seed
* Plugin architecture for new snippet generators
* No hidden global state
* --out to save directly to disk

Usage
-----
python synthetic_d.py 200
python synthetic_d.py 300 --seed 42 --out fake.d
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

__version__ = "0.1.1"

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class DConfig:
    loc: int = 200
    seed: Optional[int] = None
    imports: Sequence[str] = ("std.stdio",)
    weights: Dict[str, float] = field(default_factory=lambda: {
        "comment":  0.10,
        "var_decl": 0.30,
        "function": 0.30,
        "class":    0.30,
    })
    max_functions: Optional[int] = None
    max_classes:   Optional[int] = None


# ──────────────────────────────────────────────────────────────
# Import manager
# ──────────────────────────────────────────────────────────────

class ImportManager:
    def __init__(self, modules: Sequence[str]) -> None:
        self._mods = list(dict.fromkeys(modules))
    def add(self, mod: str) -> None:
        if mod not in self._mods:
            self._mods.append(mod)
    def render(self) -> str:
        return "".join(f"import {m};\n" for m in self._mods) + "\n"


# ──────────────────────────────────────────────────────────────
# Name generator
# ──────────────────────────────────────────────────────────────

class NameGenerator:
    D_KEYWORDS = {
        "abstract","alias","align","asm","assert","auto","body","break","case",
        "cast","catch","class","const","continue","debug","default","delegate",
        "delete","deprecated","do","else","enum","export","extern","false","final",
        "finally","for","foreach","foreach_reverse","function","goto","if","immutable",
        "import","in","inout","interface","invariant","is","lazy","macro","mixin",
        "module","new","null","out","override","package","pragma","private",
        "protected","public","pure","ref","return","scope","shared","static",
        "struct","super","switch","synchronized","template","this","throw","true",
        "try","typedef","typeid","typeof","union","unittest","version","void","while",
        "with","__traits"
    }
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.reserved = set(self.D_KEYWORDS)
    def fresh(self, *, min_len: int = 3, max_len: int = 8, capital: bool = False) -> str:
        letters = "abcdefghijklmnopqrstuvwxyz"
        for _ in range(10_000):
            length = self.rng.randint(min_len, max_len)
            name = "".join(self.rng.choice(letters) for _ in range(length))
            if capital:
                name = name.capitalize()
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
# Generators
# ──────────────────────────────────────────────────────────────

@register("comment")
def gen_comment(state: Dict) -> str:
    rng = state["rng"]
    if rng.random() < 0.6:
        tag = rng.choice(["TODO","FIXME","NOTE","HACK"])
        txt = rng.choice(["tweak this","optimize later","check boundary","refactor"])
        return f"// {tag}: {txt}\n"
    else:
        tag = rng.choice(["temporary","legacy","placeholder","wip"])
        return f"/* {tag} */\n"

@register("var_decl")
def gen_var_decl(state: Dict) -> str:
    rng     = state["rng"]
    names   = state["names"]
    imports = state["imports"]
    symbols = state["symbols"]

    ctype = rng.choice(["int","float","double","string","auto"])
    name  = names.fresh()
    symbols["variables"].add(name)

    if ctype in ("int","float","double"):
        # numeric literal
        if rng.random() < 0.5:
            lit = str(rng.randint(0, 999))
        else:
            lit = f"{rng.uniform(0,100):.2f}"
    else:
        # string literal – now correctly quoted
        txt = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(3,7)))
        lit = f'"{txt}"'
    return f"{ctype} {name} = {lit};\n"

@register("function")
def gen_function(state: Dict) -> str:
    cfg     = state["cfg"]
    rng     = state["rng"]
    names   = state["names"]
    imports = state["imports"]
    symbols = state["symbols"]

    if cfg.max_functions is not None and len(symbols["functions"]) >= cfg.max_functions:
        return ""
    ret   = rng.choice(["int","float","double","void","auto"])
    fname = names.fresh()
    symbols["functions"].add(fname)

    # parameters
    n      = rng.randint(0,2)
    params = []
    for _ in range(n):
        ptype = rng.choice(["int","float","double"])
        pname = names.fresh()
        symbols["variables"].add(pname)
        params.append(f"{ptype} {pname}")

    header = f"{ret} {fname}({', '.join(params)}) {{\n"
    body: List[str] = []
    if ret != "void":
        # numeric return
        if rng.random() < 0.5:
            body.append(f"    return {rng.randint(0,999)};\n")
        else:
            body.append(f"    return {rng.uniform(0,100):.2f};\n")
    else:
        # void – use writeln (need std.stdio)
        imports.add("std.stdio")
        body.append(f'    writeln("{fname} called");\n')
    body.append("}\n\n")
    return header + "".join(body)

@register("class")
def gen_class(state: Dict) -> str:
    cfg     = state["cfg"]
    rng     = state["rng"]
    names   = state["names"]
    imports = state["imports"]
    symbols = state["symbols"]

    if cfg.max_classes is not None and len(symbols["classes"]) >= cfg.max_classes:
        return ""
    cname = names.fresh(capital=True)
    symbols["classes"].add(cname)

    mtype = rng.choice(["int","float"])
    mname = names.fresh()
    symbols["variables"].add(mname)

    parts = [
        f"class {cname} {{\npublic:\n",
        f"    this() {{ {mname} = {rng.randint(0,999)}; }}\n",
        f"    {mtype} get{mname.capitalize()}() {{ return {mname}; }}\n",
        "private:\n",
        f"    {mtype} {mname};\n",
        "}\n\n"
    ]
    return "".join(parts)


# ──────────────────────────────────────────────────────────────
# Build & CLI
# ──────────────────────────────────────────────────────────────

def build_d(cfg: DConfig) -> str:
    rng      = random.Random(cfg.seed)
    names    = NameGenerator(rng)
    imports  = ImportManager(cfg.imports)
    symbols  = {"functions": set(), "classes": set(), "variables": set()}
    state    = {
        "cfg": cfg,
        "rng": rng,
        "names": names,
        "imports": imports,
        "symbols": symbols,
    }

    parts: List[str] = [
        "// Auto-generated D file – do not edit\n\n",
        imports.render()
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

    # main()
    parts.append("void main() {\n")
    if symbols["functions"]:
        fn = rng.choice(tuple(symbols["functions"]))
        imports.add("std.stdio")
        parts.append(f"    writeln({fn}());\n")
    if symbols["classes"]:
        cls = rng.choice(tuple(symbols["classes"]))
        var = next(iter(symbols["variables"]))
        imports.add("std.stdio")
        parts.append(f"    auto obj = new {cls}();\n")
        parts.append(f"    writeln(obj.get{var.capitalize()}());\n")
    parts.append("}\n")

    return "".join(parts)


def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate a synthetic D source file.")
    p.add_argument("loc", nargs="?", type=int, default=200, help="Approximate line count")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--max-funcs", type=int, help="Maximum functions")
    p.add_argument("--max-classes", type=int, help="Maximum classes")
    p.add_argument("--out",   type=Path, help="Path to save generated code")
    args = p.parse_args()

    cfg = DConfig(
        loc=args.loc,
        seed=args.seed,
        max_functions=args.max_funcs,
        max_classes=args.max_classes,
    )
    code = build_d(cfg)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved generated D to {args.out}")
    else:
        sys.stdout.write(code)


if __name__ == "__main__":
    _cli()
