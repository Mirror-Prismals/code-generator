#!/usr/bin/env python3
# synthetic_java.py · v0.1.0
"""
Generate synthetic—yet syntactically valid—Java source files.

Major features
--------------
* Deterministic output with --seed
* Plugin architecture for class-body snippet generators
* No hidden global state
* --out to save directly to disk

Usage
-----
python synthetic_java.py 200
python synthetic_java.py 300 --seed 42 --out FakeClass.java
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
class JavaConfig:
    loc: int = 200                       # approx. total lines
    seed: Optional[int] = None
    package_base: str = "com.example"
    weights: Dict[str, float] = field(default_factory=lambda: {
        "comment":      0.10,
        "field":        0.20,
        "method":       0.30,
        "main_method":  0.10,
        "interface":    0.05,
        "enum":         0.05,
        "annotation":   0.05,
        "nested_class": 0.15,
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
JAVA_TYPES = ["int", "long", "double", "float", "boolean", "String"]

def fresh_name(rng: random.Random, length: int = 6) -> str:
    return "".join(rng.choice(LETTERS) for _ in range(length))

def literal(rng: random.Random, ty: str) -> str:
    if ty == "String":
        s = "".join(rng.choice(LETTERS) for _ in range(rng.randint(3,8)))
        return f"\"{s}\""
    if ty == "boolean":
        return rng.choice(["true", "false"])
    if ty in ("float", "double"):
        return f"{rng.uniform(0,100):.2f}"
    # int or long
    n = rng.randint(0, 100)
    return f"{n}{'L' if ty=='long' else ''}"

@register("comment")
def gen_comment(state: Dict) -> str:
    rng = state["rng"]
    txt = fresh_name(rng, rng.randint(4, 12))
    return f"    // {txt}\n"

@register("annotation")
def gen_annotation(state: Dict) -> str:
    rng = state["rng"]
    name = fresh_name(rng, rng.randint(3, 8)).capitalize()
    return f"    @{name}\n"

@register("field")
def gen_field(state: Dict) -> str:
    rng = state["rng"]
    name = fresh_name(rng)
    ty = rng.choice(JAVA_TYPES)
    init = ""
    if rng.random() < 0.5:
        init = " = " + literal(rng, ty)
    return f"    private {ty} {name}{init};\n"

@register("method")
def gen_method(state: Dict) -> str:
    rng = state["rng"]
    name = fresh_name(rng)
    ret = rng.choice(JAVA_TYPES + ["void"])
    # parameters
    n = rng.randint(0, 3)
    params = []
    for _ in range(n):
        pty = rng.choice(JAVA_TYPES)
        pname = fresh_name(rng)
        params.append(f"{pty} {pname}")
    params_str = ", ".join(params)
    lines = []
    lines.append(f"    public {ret} {name}({params_str}) {{\n")
    if ret != "void":
        val = literal(rng, ret)
        lines.append(f"        return {val};\n")
    else:
        if rng.random() < 0.5:
            # println
            msg = fresh_name(rng, rng.randint(3, 8))
            lines.append(f"        System.out.println(\"{msg}\");\n")
    lines.append("    }\n")
    return "".join(lines)

@register("main_method")
def gen_main(state: Dict) -> str:
    if state["main_written"]:
        return ""
    state["main_written"] = True
    rng = state["rng"]
    # generate a few printlns
    lines = []
    lines.append("    public static void main(String[] args) {\n")
    for _ in range(rng.randint(1, 3)):
        msg = fresh_name(rng, rng.randint(4, 12))
        lines.append(f"        System.out.println(\"{msg}\");\n")
    lines.append("    }\n")
    return "".join(lines)

@register("interface")
def gen_interface(state: Dict) -> str:
    rng = state["rng"]
    name = fresh_name(rng, rng.randint(3, 8)).capitalize()
    # one method
    mret = rng.choice(JAVA_TYPES)
    mname = fresh_name(rng)
    return (
        f"    public interface {name} {{\n"
        f"        {mret} {mname}();\n"
        f"    }}\n"
    )

@register("enum")
def gen_enum(state: Dict) -> str:
    rng = state["rng"]
    name = fresh_name(rng, rng.randint(3, 8)).capitalize()
    count = rng.randint(2, 5)
    consts = ", ".join(f"V{rng.randint(1,99)}" for _ in range(count))
    return f"    public enum {name} {{ {consts} }}\n"

@register("nested_class")
def gen_nested_class(state: Dict) -> str:
    rng = state["rng"]
    name = fresh_name(rng, rng.randint(3,8)).capitalize()
    return (
        f"    public static class {name} {{\n"
        f"        public {name}() {{}}\n"
        f"    }}\n"
    )

def build_java(cfg: JavaConfig) -> str:
    rng = random.Random(cfg.seed)
    package = cfg.package_base + "." + fresh_name(rng, 4)
    class_name = fresh_name(rng, 6).capitalize()
    state = {
        "rng": rng,
        "main_written": False
    }

    parts: List[str] = []
    # header
    parts.append(f"// Auto-generated Java class\n\n")
    parts.append(f"package {package};\n\n")
    parts.append("import java.util.*;\n\n")
    parts.append(f"public class {class_name} {{\n")
    lines = sum(p.count("\n") for p in parts)

    kinds, weights = zip(*cfg.weights.items())
    # generate body
    while lines < cfg.loc - 1:
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        snippet = _REGISTRY[kind](state)
        if not snippet:
            continue
        parts.append(snippet)
        lines += snippet.count("\n")

    # ensure main exists
    if not state["main_written"]:
        parts.append(gen_main(state))
    parts.append("}\n")
    return "".join(parts)

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate a synthetic Java class.")
    p.add_argument("loc", nargs="?", type=int, default=200, help="Approximate line count")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out", type=Path, help="Path to save generated .java")
    args = p.parse_args()

    cfg = JavaConfig(loc=args.loc, seed=args.seed)
    code = build_java(cfg)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved generated Java to {args.out}")
    else:
        sys.stdout.write(code)

if __name__ == "__main__":
    _cli()
