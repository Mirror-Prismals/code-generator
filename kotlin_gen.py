#!/usr/bin/env python3
# synthetic_kotlin.py · v0.1.0
"""
Generate synthetic—yet syntactically valid—Kotlin source files.

Major features
--------------
* Deterministic output with --seed
* Plugin architecture for new snippet generators
* Tracks package/import/main to ensure validity
* --out to save directly to disk

Usage
-----
python synthetic_kotlin.py 200
python synthetic_kotlin.py 300 --seed 42 --out Fake.kt
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
class KotlinConfig:
    loc: int = 200
    seed: Optional[int] = None
    weights: Dict[str, float] = field(default_factory=lambda: {
        "comment":       0.10,
        "package":       0.01,
        "import":        0.05,
        "val_decl":      0.15,
        "fun_decl":      0.20,
        "class_decl":    0.15,
        "data_class":    0.10,
        "interface":     0.05,
        "object_decl":   0.05,
        "main_func":     0.10,
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

def fresh_name(rng: random.Random, length: int = 6, capital: bool = False) -> str:
    s = "".join(rng.choice(LETTERS) for _ in range(length))
    return s.capitalize() if capital else s

def random_literal(rng: random.Random) -> str:
    t = rng.random()
    if t < 0.3:
        return str(rng.randint(0, 100))
    if t < 0.6:
        text = "".join(rng.choice(LETTERS) for _ in range(rng.randint(3,8)))
        return f"\"{text}\""
    return rng.choice(["true", "false"])

# ──────────────────────────────────────────────────────────────
# Generators
# ──────────────────────────────────────────────────────────────

@register("comment")
def gen_comment(state: Dict) -> str:
    rng = state["rng"]
    tags = ["// TODO", "// FIXME", "// NOTE", "// HACK"]
    msg = fresh_name(rng, rng.randint(3, 8))
    return f"{rng.choice(tags)}: {msg}\n"

@register("package")
def gen_package(state: Dict) -> str:
    if state["package_written"]:
        return ""
    state["package_written"] = True
    rng = state["rng"]
    name = fresh_name(rng, 4).lower()
    return f"package com.example.{name}\n\n"

@register("import")
def gen_import(state: Dict) -> str:
    if len(state["imports"]) >= state["max_imports"]:
        return ""
    rng = state["rng"]
    choices = [
        "kotlin.random.Random",
        "kotlin.collections.*",
        "java.time.LocalDate",
        "java.io.File"
    ]
    imp = rng.choice(choices)
    if imp in state["imports"]:
        return ""
    state["imports"].add(imp)
    return f"import {imp}\n"

@register("val_decl")
def gen_val_decl(state: Dict) -> str:
    rng = state["rng"]
    name = fresh_name(rng)
    literal = random_literal(rng)
    return f"val {name} = {literal}\n"

@register("fun_decl")
def gen_fun_decl(state: Dict) -> str:
    rng = state["rng"]
    name = fresh_name(rng)
    params = []
    for _ in range(rng.randint(0, 2)):
        pn = fresh_name(rng)
        pt = rng.choice(["Int","String","Boolean","Double"])
        params.append(f"{pn}: {pt}")
    params_str = ", ".join(params)
    ret = rng.choice(["Int","String","Boolean","Unit","Double"])
    body = ""
    if ret != "Unit":
        body = f" = {random_literal(rng)}"
    return f"fun {name}({params_str}): {ret}{body}\n"

@register("class_decl")
def gen_class_decl(state: Dict) -> str:
    rng = state["rng"]
    name = fresh_name(rng, capital=True)
    props = []
    for _ in range(rng.randint(0, 2)):
        pn = fresh_name(rng)
        pt = rng.choice(["Int","String","Boolean","Double"])
        props.append(f"    val {pn}: {pt}")
    body = "\n".join(props)
    return f"class {name} {{\n{body}\n}}\n\n"

@register("data_class")
def gen_data_class(state: Dict) -> str:
    rng = state["rng"]
    name = fresh_name(rng, capital=True)
    params = []
    for _ in range(rng.randint(1, 3)):
        pn = fresh_name(rng)
        pt = rng.choice(["Int","String","Boolean","Double"])
        params.append(f"val {pn}: {pt}")
    params_str = ", ".join(params)
    return f"data class {name}({params_str})\n\n"

@register("interface")
def gen_interface(state: Dict) -> str:
    rng = state["rng"]
    name = fresh_name(rng, capital=True)
    meth = fresh_name(rng)
    return f"interface {name} {{\n    fun {meth}(): Unit\n}}\n\n"

@register("object_decl")
def gen_object_decl(state: Dict) -> str:
    rng = state["rng"]
    name = fresh_name(rng, capital=True)
    return f"object {name} {{\n    // singleton\n}}\n\n"

@register("main_func")
def gen_main_func(state: Dict) -> str:
    if state["main_written"]:
        return ""
    state["main_written"] = True
    rng = state["rng"]
    lines = ["fun main(args: Array<String>) {"]
    # a few prints or val decls
    for _ in range(rng.randint(1, 3)):
        if rng.random() < 0.5:
            nm = fresh_name(rng)
            lit = random_literal(rng)
            lines.append(f"    val {nm} = {lit}")
        else:
            msg = fresh_name(rng, rng.randint(3,8))
            lines.append(f"    println(\"{msg}\")")
    lines.append("}\n")
    return "\n".join(lines)

# ──────────────────────────────────────────────────────────────
# Build & CLI
# ──────────────────────────────────────────────────────────────

def build_kotlin(cfg: KotlinConfig) -> str:
    rng = random.Random(cfg.seed)
    state = {
        "rng": rng,
        "package_written": False,
        "imports": set(),
        "max_imports": 3,
        "main_written": False,
    }
    parts: List[str] = ["// Auto-generated Kotlin code\n\n"]
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
    p = argparse.ArgumentParser(description="Generate synthetic Kotlin code.")
    p.add_argument("loc", nargs="?", type=int, default=200, help="Approx. line count")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out", type=Path, help="Path to save generated .kt")
    args = p.parse_args()

    cfg = KotlinConfig(loc=args.loc, seed=args.seed)
    code = build_kotlin(cfg)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved generated Kotlin to {args.out}")
    else:
        sys.stdout.write(code)

if __name__ == "__main__":
    _cli()
