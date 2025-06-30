#!/usr/bin/env python3
# synthetic_ts.py · v0.1.0
"""
Generate synthetic—yet syntactically valid—TypeScript source files.

Major features
--------------
* Deterministic output with --seed
* Plugin architecture for new snippet generators
* No hidden global state
* --out to save directly to disk

Usage
-----
python synthetic_ts.py 200
python synthetic_ts.py 300 --seed 42 --out fake.ts
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
class TsConfig:
    loc: int = 200
    seed: Optional[int] = None
    modules: Sequence[str] = ("fs", "path", "http", "url", "util", "events")
    weights: Dict[str, float] = field(default_factory=lambda: {
        "comment":       0.05,
        "import":        0.05,
        "var_decl":      0.25,
        "function":      0.20,
        "arrow_function":0.10,
        "interface":     0.10,
        "class":         0.10,
        "enum":          0.05,
        "export":        0.10,
    })
    max_functions: Optional[int] = None
    max_interfaces: Optional[int] = None
    max_classes: Optional[int] = None

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

class ImportManager:
    def __init__(self) -> None:
        self._map: Dict[str, str] = {}

    def add(self, module: str, alias: str) -> None:
        if module not in self._map:
            self._map[module] = alias

    def render(self) -> str:
        lines = ""
        for module, alias in self._map.items():
            lines += f"import * as {alias} from '{module}';\n"
        return lines + ("\n" if lines else "")


class NameGenerator:
    TS_KEYWORDS = {
        "break","case","catch","class","const","continue","debugger","default",
        "delete","do","else","enum","export","extends","false","finally","for",
        "function","if","import","in","instanceof","new","null","return","super",
        "switch","this","throw","true","try","typeof","var","void","while","with",
        "as","implements","interface","let","package","private","protected",
        "public","static","yield","any","boolean","constructor","declare","get",
        "module","require","number","set","symbol","type","from","of"
    }

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.reserved = set(self.TS_KEYWORDS)

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


def random_string(rng: random.Random, min_len: int = 3, max_len: int = 10) -> str:
    letters = "abcdefghijklmnopqrstuvwxyz"
    length = rng.randint(min_len, max_len)
    s = "".join(rng.choice(letters) for _ in range(length))
    return f'"{s}"'

def gen_literal(state: Dict, depth: int = 0) -> str:
    rng = state["rng"]
    if depth >= 2 or rng.random() < 0.4:
        t = rng.random()
        if t < 0.4:
            return str(rng.randint(0, 999))
        elif t < 0.7:
            return rng.choice(["true", "false"])
        else:
            return random_string(rng)
    left = gen_literal(state, depth + 1)
    right = gen_literal(state, depth + 1)
    op = rng.choice(["+", "-", "*", "/"])
    return f"({left}{op}{right})"


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
    if rng.random() < 0.5:
        tag = rng.choice(["TODO", "FIXME", "NOTE", "HACK", "BUG"])
        txt = rng.choice(["refactor", "optimize", "handle edge case", "cleanup"])
        return f"// {tag}: {txt}\n"
    else:
        tag = rng.choice(["temporary", "legacy", "placeholder", "wip"])
        return f"/* {tag} */\n"

@register("import")
def gen_import(state: Dict) -> str:
    cfg    = state["cfg"]
    rng    = state["rng"]
    names  = state["names"]
    imports= state["imports"]

    module = rng.choice(cfg.modules)
    alias  = names.fresh(capital=True)
    imports.add(module, alias)
    return ""  # rendered at top

@register("var_decl")
def gen_var_decl(state: Dict) -> str:
    rng     = state["rng"]
    names   = state["names"]
    symbols = state["symbols"]

    kind    = rng.choice(["let", "const"])
    name    = names.fresh()
    symbols["variables"].add(name)

    # optional type
    if rng.random() < 0.5:
        t = rng.choice(["number", "string", "boolean", "any"])
        type_ann = f": {t}"
    else:
        type_ann = ""

    val = gen_literal(state)
    return f"{kind} {name}{type_ann} = {val};\n"

@register("function")
def gen_function(state: Dict) -> str:
    cfg     = state["cfg"]
    rng     = state["rng"]
    names   = state["names"]
    symbols = state["symbols"]

    if cfg.max_functions is not None and len(symbols["functions"]) >= cfg.max_functions:
        return ""
    name    = names.fresh()
    symbols["functions"].add(name)

    # params
    n = rng.randint(0, 3)
    params = []
    for _ in range(n):
        pname = names.fresh()
        ptype = rng.choice(["number", "string", "boolean", "any"])
        symbols["variables"].add(pname)
        params.append(f"{pname}: {ptype}")

    ret = rng.choice(["number", "string", "boolean", "void", "any"])
    header = f"function {name}({', '.join(params)}): {ret} {{\n"
    body: List[str] = []
    if ret != "void":
        body.append(f"    return {gen_literal(state)};\n")
    else:
        body.append("    // no return\n")
    body.append("}\n\n")
    return header + "".join(body)

@register("arrow_function")
def gen_arrow(state: Dict) -> str:
    rng     = state["rng"]
    names   = state["names"]
    symbols = state["symbols"]

    name    = names.fresh()
    symbols["functions"].add(name)

    # params
    n = rng.randint(0, 2)
    params = [names.fresh() for _ in range(n)]
    ret    = rng.choice(["number", "string", "boolean", "any"])
    expr   = gen_literal(state)
    return f"const {name} = ({', '.join(params)}): {ret} => {expr};\n"

@register("interface")
def gen_interface(state: Dict) -> str:
    cfg       = state["cfg"]
    rng       = state["rng"]
    names     = state["names"]
    symbols   = state["symbols"]

    if cfg.max_interfaces is not None and len(symbols["interfaces"]) >= cfg.max_interfaces:
        return ""
    name      = names.fresh(capital=True)
    symbols["interfaces"].add(name)

    n = rng.randint(1, 4)
    props = []
    for _ in range(n):
        pname = names.fresh()
        ptype = rng.choice(["number", "string", "boolean", "any"])
        optional = "?" if rng.random() < 0.3 else ""
        props.append(f"    {pname}{optional}: {ptype};")
    return f"interface {name} {{\n" + "\n".join(props) + "\n}\n\n"

@register("class")
def gen_class(state: Dict) -> str:
    cfg     = state["cfg"]
    rng     = state["rng"]
    names   = state["names"]
    symbols = state["symbols"]

    if cfg.max_classes is not None and len(symbols["classes"]) >= cfg.max_classes:
        return ""
    name    = names.fresh(capital=True)
    symbols["classes"].add(name)

    # single constructor prop
    pname = names.fresh()
    ptype = rng.choice(["number", "string", "boolean", "any"])
    symbols["variables"].add(pname)

    parts = [
        f"class {name} {{\n",
        f"    constructor(public {pname}: {ptype}) {{}}\n",
        "}\n\n"
    ]
    return "".join(parts)

@register("enum")
def gen_enum(state: Dict) -> str:
    rng     = state["rng"]
    names   = state["names"]
    symbols = state["symbols"]

    name    = names.fresh(capital=True)
    symbols["enums"].add(name)

    n = rng.randint(2, 5)
    members = [names.fresh(capital=True) for _ in range(n)]
    body = ", ".join(members)
    return f"enum {name} {{ {body} }}\n\n"

@register("export")
def gen_export(state: Dict) -> str:
    rng     = state["rng"]
    symbols = state["symbols"]
    all_syms = (
        list(symbols["functions"]) +
        list(symbols["classes"]) +
        list(symbols["interfaces"]) +
        list(symbols["enums"])
    )
    if not all_syms:
        return ""
    sym = rng.choice(all_syms)
    return f"export {{ {sym} }};\n"

# ──────────────────────────────────────────────────────────────
# Build & CLI
# ──────────────────────────────────────────────────────────────

def build_ts(cfg: TsConfig) -> str:
    rng      = random.Random(cfg.seed)
    names    = NameGenerator(rng)
    imports  = ImportManager()
    symbols  = {
        "variables": set(),
        "functions": set(),
        "classes": set(),
        "interfaces": set(),
        "enums": set(),
    }
    state = {
        "cfg": cfg, "rng": rng, "names": names,
        "imports": imports, "symbols": symbols
    }

    parts: List[str] = [
        "// Auto-generated TypeScript – do not edit\n\n",
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

    # optional invocation
    if symbols["functions"]:
        fn = rng.choice(tuple(symbols["functions"]))
        parts.append(f"\nconsole.log({fn}());\n")

    return "".join(parts)


def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate a synthetic TypeScript file.")
    p.add_argument("loc", nargs="?", type=int, default=200, help="Approx. line count")
    p.add_argument("--seed", type=int, help="Random seed for deterministic output")
    p.add_argument("--max-funcs", type=int, help="Max functions")
    p.add_argument("--max-interfaces", type=int, help="Max interfaces")
    p.add_argument("--max-classes", type=int, help="Max classes")
    p.add_argument("--out", type=Path, help="Path to save generated code")
    args = p.parse_args()

    cfg = TsConfig(
        loc=args.loc,
        seed=args.seed,
        max_functions=args.max_funcs,
        max_interfaces=args.max_interfaces,
        max_classes=args.max_classes,
    )
    code = build_ts(cfg)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved generated TypeScript to {args.out}")
    else:
        sys.stdout.write(code)


if __name__ == "__main__":
    _cli()
