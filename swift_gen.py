#!/usr/bin/env python3
# synthetic_swift.py · v0.1.0
"""
Generate synthetic—yet syntactically valid—Swift source files.

Major features
--------------
* Deterministic output with --seed
* Plugin architecture for new snippet generators
* Tracks defined types for realistic extensions
* No hidden global state
* --out to save directly to disk

Usage
-----
python synthetic_swift.py 200
python synthetic_swift.py 300 --seed 42 --out Fake.swift
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
class SwiftConfig:
    loc: int = 200
    seed: Optional[int] = None
    weights: Dict[str, float] = field(default_factory=lambda: {
        "comment":     0.10,
        "import":      0.05,
        "var_decl":    0.15,
        "function":    0.20,
        "struct":      0.10,
        "class":       0.10,
        "enum":        0.05,
        "protocol":    0.05,
        "extension":   0.10,
        "main_code":   0.10,
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

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

LETTERS = "abcdefghijklmnopqrstuvwxyz"
SWIFT_TYPES = ["Int", "Double", "String", "Bool", "Float"]

def fresh_name(rng: random.Random, length: int = 6, capital: bool = False) -> str:
    s = "".join(rng.choice(LETTERS) for _ in range(length))
    return s.capitalize() if capital else s

def random_literal(rng: random.Random, ty: str) -> str:
    if ty == "String":
        s = "".join(rng.choice(LETTERS) for _ in range(rng.randint(3, 8)))
        return f"\"{s}\""
    if ty == "Bool":
        return rng.choice(["true", "false"])
    if ty in ("Int", "Float", "Double"):
        val = rng.uniform(0, 100)
        if ty == "Int":
            return str(int(val))
        suffix = "f" if ty == "Float" else ""
        return f"{val:.2f}{suffix}"
    return "nil"

# ──────────────────────────────────────────────────────────────
# Generators
# ──────────────────────────────────────────────────────────────

@register("comment")
def gen_comment(state: Dict) -> str:
    rng = state["rng"]
    tags = ["// TODO", "// FIXME", "// NOTE", "// HACK"]
    msg = fresh_name(rng, rng.randint(3, 8))
    return f"{rng.choice(tags)}: {msg}\n"

@register("import")
def gen_import(state: Dict) -> str:
    rng = state["rng"]
    opts = ["Foundation", "SwiftUI", "UIKit"]
    imp = rng.choice(opts)
    if imp in state["imports"]:
        return ""
    state["imports"].add(imp)
    return f"import {imp}\n"

@register("var_decl")
def gen_var_decl(state: Dict) -> str:
    rng = state["rng"]
    isLet = rng.random() < 0.5
    kw = "let" if isLet else "var"
    name = fresh_name(rng)
    ty = rng.choice(SWIFT_TYPES)
    val = random_literal(rng, ty) if rng.random() < 0.7 else ""
    init = f": {ty} = {val}" if val else f": {ty}"
    return f"{kw} {name}{init}\n"

@register("function")
def gen_function(state: Dict) -> str:
    rng = state["rng"]
    name = fresh_name(rng)
    ret = rng.choice(SWIFT_TYPES + ["Void"])
    n = rng.randint(0, 3)
    params = []
    for _ in range(n):
        pname = fresh_name(rng)
        pty = rng.choice(SWIFT_TYPES)
        params.append(f"{pname}: {pty}")
    params_str = ", ".join(params)
    lines = [f"func {name}({params_str}) -> {ret} {{\n"]
    if ret != "Void":
        lit = random_literal(rng, ret)
        lines.append(f"    return {lit}\n")
    else:
        if rng.random() < 0.5:
            lines.append(f"    print(\"{fresh_name(rng,5)}\")\n")
    lines.append("}\n\n")
    return "".join(lines)

@register("struct")
def gen_struct(state: Dict) -> str:
    rng = state["rng"]
    name = fresh_name(rng, capital=True)
    state["types"].add(name)
    n = rng.randint(1, 3)
    fields = []
    for _ in range(n):
        fn = fresh_name(rng)
        ft = rng.choice(SWIFT_TYPES)
        fields.append(f"    var {fn}: {ft}")
    body = "\n".join(fields)
    return f"struct {name} {{\n{body}\n}}\n\n"

@register("class")
def gen_class(state: Dict) -> str:
    rng = state["rng"]
    name = fresh_name(rng, capital=True)
    state["types"].add(name)
    body = [
        f"class {name} {{",
        "    var id: Int",
        "    init(id: Int) {",
        "        self.id = id",
        "    }",
        "    func describe() -> String {",
        f"        return \"{name} #\\(id)\"",
        "    }",
        "}\n"
    ]
    return "\n".join("    " + line if idx>0 else line for idx,line in enumerate(body)) + "\n"

@register("enum")
def gen_enum(state: Dict) -> str:
    rng = state["rng"]
    name = fresh_name(rng, capital=True)
    state["types"].add(name)
    count = rng.randint(2, 5)
    cases = "\n".join(f"    case {fresh_name(rng).lower()}" for _ in range(count))
    return f"enum {name} {{\n{cases}\n}}\n\n"

@register("protocol")
def gen_protocol(state: Dict) -> str:
    rng = state["rng"]
    name = fresh_name(rng, capital=True)
    state["protocols"].add(name)
    return f"protocol {name} {{ func {fresh_name(rng)}() }}\n\n"

@register("extension")
def gen_extension(state: Dict) -> str:
    rng = state["rng"]
    if not state["types"]:
        return ""
    t = rng.choice(list(state["types"]))
    fname = fresh_name(rng)
    return (
        f"extension {t} {{\n"
        f"    func {fname}() {{\n"
        f"        print(\"ext {t}.{fname}\")\n"
        f"    }}\n"
        f"}}\n\n"
    )

@register("main_code")
def gen_main_code(state: Dict) -> str:
    rng = state["rng"]
    # top-level Swift script code
    if rng.random() < 0.5 and state["types"]:
        t = rng.choice(list(state["types"]))
        return f"let obj = {t}(id: {rng.randint(1,10)})\nprint(obj.describe())\n"
    name = fresh_name(rng)
    return f"print(\"Hello, {name}!\")\n"

# ──────────────────────────────────────────────────────────────
# Build & CLI
# ──────────────────────────────────────────────────────────────

def build_swift(cfg: SwiftConfig) -> str:
    rng = random.Random(cfg.seed)
    state = {
        "rng": rng,
        "imports": set(),
        "types": set(),
        "protocols": set(),
    }
    parts: List[str] = ["// Auto-generated Swift code\n\n"]
    lines = parts[0].count("\n")
    kinds, weights = zip(*cfg.weights.items())

    while lines < cfg.loc:
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        snippet = _REGISTRY[kind](state)
        if not snippet:
            continue
        parts.append(snippet)
        lines += snippet.count("\n")

    return "".join(parts)

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate a synthetic Swift file.")
    p.add_argument("loc", nargs="?", type=int, default=200, help="Approx. line count")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out", type=Path, help="Path to save generated .swift")
    args = p.parse_args()

    cfg = SwiftConfig(loc=args.loc, seed=args.seed)
    code = build_swift(cfg)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved generated Swift to {args.out}")
    else:
        sys.stdout.write(code)

if __name__ == "__main__":
    _cli()
