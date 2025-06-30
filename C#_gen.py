#!/usr/bin/env python3
# synthetic_csharp.py · v0.1.0
"""
Generate synthetic—yet syntactically valid—C# source files.

Major features
--------------
* Deterministic output with --seed
* Plugin architecture for snippet generators
* No hidden global state
* --out to save directly to disk

Usage
-----
python synthetic_csharp.py 200
python synthetic_csharp.py 300 --seed 42 --out Program.cs
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
class CSharpConfig:
    loc: int = 200                 # approx. total lines
    seed: Optional[int] = None
    namespace_base: str = "Synthetic"

    weights: Dict[str, float] = field(default_factory=lambda: {
        "comment":        0.10,
        "field":          0.15,
        "property":       0.15,
        "method":         0.20,
        "main_method":    0.10,
        "interface":      0.10,
        "enum":           0.10,
        "attribute":      0.10,
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
        self.used: set[str] = set()

    def fresh(self, *, min_len: int = 3, max_len: int = 8, capital: bool = False) -> str:
        for _ in range(10000):
            length = self.rng.randint(min_len, max_len)
            name = "".join(self.rng.choice(LETTERS) for _ in range(length))
            if capital:
                name = name.capitalize()
            if name not in self.used:
                self.used.add(name)
                return name
        raise RuntimeError("Identifier space exhausted")

def literal(rng: random.Random, ty: str) -> str:
    if ty == "string":
        s = "".join(rng.choice(LETTERS) for _ in range(rng.randint(3,8)))
        return f"\"{s}\""
    if ty == "bool":
        return rng.choice(["true", "false"])
    if ty in ("int","long"):
        n = rng.randint(0,100)
        return f"{n}{'L' if ty=='long' else ''}"
    if ty in ("double","float"):
        return f"{rng.uniform(0,100):.2f}{'f' if ty=='float' else ''}"
    # fallback
    return "null"

@register("comment")
def gen_comment(state: Dict) -> str:
    rng = state["rng"]
    tags = ["TODO","FIXME","NOTE","HACK"]
    text = state["names"].fresh(min_len=4, max_len=12)
    return f"        // {rng.choice(tags)}: {text}\n"

@register("field")
def gen_field(state: Dict) -> str:
    rng = state["rng"]
    name = state["names"].fresh(capital=True)
    ty = rng.choice(["int","long","double","float","bool","string"])
    init = ""
    if rng.random() < 0.5:
        init = " = " + literal(rng, ty)
    return f"        public static {ty} {name}{init};\n"

@register("property")
def gen_property(state: Dict) -> str:
    name = state["names"].fresh(capital=True)
    ty = state["rng"].choice(["int","double","bool","string"])
    return f"        public static {ty} {name} {{ get; set; }}\n"

@register("method")
def gen_method(state: Dict) -> str:
    rng = state["rng"]
    name = state["names"].fresh(capital=True)
    ret = rng.choice(["int","long","double","float","bool","string","void"])
    # params
    n = rng.randint(0,3)
    params = []
    for _ in range(n):
        pty = rng.choice(["int","double","bool","string"])
        pname = state["names"].fresh()
        params.append(f"{pty} {pname}")
    body: List[str] = []
    body.append(f"        public static {ret} {name}({', '.join(params)})")
    body.append("        {")
    if ret != "void":
        body.append(f"            return {literal(rng, ret)};")
    else:
        body.append(f"            // no return")
    body.append("        }\n")
    return "\n".join(body) + "\n"

@register("main_method")
def gen_main(state: Dict) -> str:
    if state["main_written"]:
        return ""
    state["main_written"] = True
    rng = state["rng"]
    lines: List[str] = []
    lines.append("        public static void Main(string[] args)")
    lines.append("        {")
    for _ in range(rng.randint(1,3)):
        msg = state["names"].fresh(min_len=4, max_len=12)
        lines.append(f"            System.Console.WriteLine(\"{msg}\");")
    lines.append("        }\n")
    return "\n".join(lines)

@register("interface")
def gen_interface(state: Dict) -> str:
    rng = state["rng"]
    name = "I" + state["names"].fresh(capital=True)
    # one method signature
    ret = rng.choice(["int","void","bool","string"])
    mname = state["names"].fresh(capital=True)
    return (
        f"        public interface {name}\n"
        f"        {{\n"
        f"            {ret} {mname}();\n"
        f"        }}\n\n"
    )

@register("enum")
def gen_enum(state: Dict) -> str:
    rng = state["rng"]
    name = state["names"].fresh(capital=True)
    count = rng.randint(2,5)
    members = ", ".join(f"Val{rng.randint(1,99)}" for _ in range(count))
    return f"        public enum {name} {{ {members} }}\n"

@register("attribute")
def gen_attribute(state: Dict) -> str:
    rng = state["rng"]
    attr = rng.choice(["Serializable","Obsolete","DataContract"])
    return f"        [{attr}]\n"

def build_csharp(cfg: CSharpConfig) -> str:
    rng = random.Random(cfg.seed)
    names = NameGen(rng)
    namespace = cfg.namespace_base + "." + names.fresh(capital=True)
    state = {
        "rng": rng,
        "names": names,
        "main_written": False,
    }

    parts: List[str] = []
    # header
    parts.append("// Auto-generated C# file\n\n")
    # usings
    parts.append("using System;\nusing System.Collections.Generic;\n\n")
    # namespace and class
    parts.append(f"namespace {namespace}\n{{\n")
    parts.append("    public static class Program\n    {\n")
    lines = sum(p.count("\n") for p in parts)

    kinds, weights = zip(*cfg.weights.items())
    while lines < cfg.loc - 2:
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        chunk = _REGISTRY[kind](state)
        if not chunk:
            continue
        parts.append(chunk)
        lines += chunk.count("\n")

    # ensure Main exists
    if not state["main_written"]:
        parts.append(gen_main(state))
    # close class & namespace
    parts.append("    }\n}\n")

    return "".join(parts)

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic C# code.")
    p.add_argument("loc", nargs="?", type=int, default=200, help="Approx. line count")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out", type=Path, help="Path to save generated .cs")
    args = p.parse_args()

    cfg = CSharpConfig(loc=args.loc, seed=args.seed)
    code = build_csharp(cfg)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved generated C# to {args.out}")
    else:
        sys.stdout.write(code)

if __name__ == "__main__":
    _cli()
