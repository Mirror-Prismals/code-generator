#!/usr/bin/env python3
# synthetic_rust.py · v1.1.0
"""
Generate synthetic—yet syntactically valid—Rust source files.

Major features
--------------
* Deterministic output with --seed
* Plugin architecture for new statement generators
* No hidden global state
* --out to save directly to disk

Usage
-----
python synthetic_rust.py 200
python synthetic_rust.py 300 --seed 42 --out fake.rs
"""

from __future__ import annotations
import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

__version__ = "1.1.0"

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

@dataclass(slots=True, frozen=True)
class RustConfig:
    loc: int = 200
    seed: Optional[int] = None
    max_literal_depth: int = 3
    crates: Sequence[str] = field(default_factory=lambda: ["std::collections"])
    macro_comments: Sequence[str] = field(default_factory=lambda: [
        "TODO: optimize this",
        "FIXME: rename fields",
        "Hacky workaround",
        "Left as an exercise to the reader",
    ])
    weights: Dict[str, float] = field(default_factory=lambda: {
        "comment": 0.1,
        "let": 0.3,
        "fn": 0.3,
        "struct": 0.3,
    })
    max_fns: Optional[int] = None
    max_structs: Optional[int] = None


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

class UseManager:
    def __init__(self, crates: Sequence[str]) -> None:
        self._crates: list[str] = list(dict.fromkeys(crates))

    def add(self, crate: str) -> None:
        if crate not in self._crates:
            self._crates.append(crate)

    def render(self) -> str:
        lines = "".join(f"use {crate}::*;\n" for crate in self._crates)
        return lines + ("\n" if lines else "")


class NameGenerator:
    RUST_KEYWORDS = {
        "as", "break", "const", "continue", "crate", "else", "enum", "extern",
        "false", "fn", "for", "if", "impl", "in", "let", "loop", "match",
        "mod", "move", "mut", "pub", "ref", "return", "self", "Self", "static",
        "struct", "super", "trait", "true", "type", "unsafe", "use", "where",
        "while",
    }

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.reserved: set[str] = set(self.RUST_KEYWORDS)

    def fresh_snake(self, *, min_len: int = 3, max_len: int = 10) -> str:
        for _ in range(10_000):
            length = self.rng.randint(min_len, max_len)
            words = [
                "".join(self.rng.choice("abcdefghijklmnopqrstuvwxyz")
                        for _ in range(self.rng.randint(1, 3)))
                for _ in range(max(1, length // 3))
            ]
            name = "_".join(words)
            if name not in self.reserved:
                self.reserved.add(name)
                return name
        raise RuntimeError("Identifier space exhausted")

    def fresh_camel(self, *, min_len: int = 3, max_len: int = 10) -> str:
        for _ in range(10_000):
            raw = "".join(self.rng.choice("abcdefghijklmnopqrstuvwxyz")
                          for _ in range(self.rng.randint(min_len, max_len)))
            name = raw.capitalize()
            if name not in self.reserved:
                self.reserved.add(name)
                return name
        raise RuntimeError("Identifier space exhausted")


# ──────────────────────────────────────────────────────────────
# Context passed to generators
# ──────────────────────────────────────────────────────────────

@dataclass(slots=True)
class Context:
    cfg: RustConfig
    rng: random.Random
    names: NameGenerator
    uses: UseManager
    symbols: Dict[str, set]
    struct_fields: Dict[str, List[str]]  # keep track of each struct's fields

    def literal(self, depth: int = 0) -> str:
        if depth >= self.cfg.max_literal_depth or self.rng.random() < 0.4:
            choice = self.rng.random()
            if choice < 0.4:
                return str(self.rng.randint(0, 9999))
            elif choice < 0.7:
                return "true" if self.rng.random() < 0.5 else "false"
            else:
                txt = "".join(self.rng.choice("abcdefghijklmnopqrstuvwxyz")
                              for _ in range(self.rng.randint(3, 8)))
                return f'"{txt}"'
        lhs = self.literal(depth + 1)
        rhs = self.literal(depth + 1)
        op = self.rng.choice(["+", "-", "*", "/"])
        return f"({lhs} {op} {rhs})"

    def macro_comment(self) -> str:
        return f"// {self.rng.choice(self.cfg.macro_comments)}\n"


# ──────────────────────────────────────────────────────────────
# Generator registry
# ──────────────────────────────────────────────────────────────

GeneratorFn = Callable[[Context], str]
_REGISTRY: Dict[str, GeneratorFn] = {}


def register(kind: str) -> Callable[[GeneratorFn], GeneratorFn]:
    def inner(fn: GeneratorFn) -> GeneratorFn:
        if kind in _REGISTRY:
            raise ValueError(f"Duplicate generator: {kind}")
        _REGISTRY[kind] = fn
        return fn
    return inner


# ──────────────────────────────────────────────────────────────
# Statement generators
# ──────────────────────────────────────────────────────────────

@register("comment")
def gen_comment(ctx: Context) -> str:
    return ctx.macro_comment()

@register("let")
def gen_let(ctx: Context) -> str:
    name = ctx.names.fresh_snake()
    ctx.symbols["variables"].add(name)
    expr = ctx.literal()
    return f"let {name} = {expr};\n"

@register("fn")
def gen_fn(ctx: Context) -> str:
    if ctx.cfg.max_fns is not None and len(ctx.symbols["functions"]) >= ctx.cfg.max_fns:
        return ""
    fname = ctx.names.fresh_snake()
    ctx.symbols["functions"].add(fname)

    # parameters
    n_params = ctx.rng.randint(0, 3)
    params = []
    for _ in range(n_params):
        pname = ctx.names.fresh_snake()
        params.append(f"{pname}: i32")
    params_str = ", ".join(params)

    # optional doc comment + body
    body: List[str] = []
    if ctx.rng.random() < 0.3:
        body.append(f"    /// {ctx.rng.choice(ctx.cfg.macro_comments)}\n")
    ret = ctx.literal()
    if ctx.rng.random() < 0.2:
        body.append(f"    {ctx.macro_comment().rstrip()}\n")
    body.append(f"    {ret}\n")

    signature = f"pub fn {fname}({params_str}) -> i32 {{\n"
    return signature + "".join(body) + "}\n\n"

@register("struct")
def gen_struct(ctx: Context) -> str:
    if ctx.cfg.max_structs is not None and len(ctx.symbols["structs"]) >= ctx.cfg.max_structs:
        return ""
    sname = ctx.names.fresh_camel()
    ctx.symbols["structs"].add(sname)

    n_fields = ctx.rng.randint(1, 3)
    fields: List[str] = []
    field_names: List[str] = []
    for _ in range(n_fields):
        fname = ctx.names.fresh_snake()
        field_names.append(fname)
        fields.append(f"    pub {fname}: i32,")
    ctx.struct_fields[sname] = field_names

    struct_def = f"pub struct {sname} {{\n" + "\n".join(fields) + "\n}\n\n"

    impl = [f"impl {sname} {{\n"]
    impl.append("    pub fn new() -> Self {\n")
    init_pairs = ", ".join(f"{name}: 0" for name in field_names)
    impl.append(f"        {sname} {{ {init_pairs} }}\n    }}\n")
    # getter for first field
    first = field_names[0]
    impl.append(f"    pub fn get_{first}(&self) -> i32 {{ self.{first} }}\n")
    impl.append("}\n\n")

    return struct_def + "".join(impl)


# ──────────────────────────────────────────────────────────────
# Build pipeline
# ──────────────────────────────────────────────────────────────

def build_rust(cfg: RustConfig) -> str:
    rng = random.Random(cfg.seed)
    ctx = Context(
        cfg=cfg,
        rng=rng,
        names=NameGenerator(rng),
        uses=UseManager(cfg.crates),
        symbols={"functions": set(), "structs": set(), "variables": set()},
        struct_fields={}
    )

    parts: List[str] = [
        "// Auto-generated module – do not edit.\n\n",
        ctx.uses.render(),
    ]
    lines = sum(p.count("\n") for p in parts)
    kinds, weights = zip(*cfg.weights.items())

    while lines < cfg.loc:
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        chunk = _REGISTRY[kind](ctx)
        if not chunk:
            continue
        parts.append(chunk)
        lines += chunk.count("\n")

    # main()
    parts.append("fn main() {\n")
    if ctx.symbols["functions"]:
        fn = rng.choice(tuple(ctx.symbols["functions"]))
        parts.append(f"    println!(\"{{}}\", {fn}());\n")
    if ctx.symbols["structs"]:
        st = rng.choice(tuple(ctx.symbols["structs"]))
        fld = ctx.struct_fields[st][0]
        parts.append(f"    let obj = {st}::new();\n")
        parts.append(f"    println!(\"{{}}\", obj.get_{fld}());\n")
    parts.append("}\n")

    return "".join(parts)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate a synthetic Rust module.")
    p.add_argument("loc", nargs="?", type=int, default=200, help="Approximate LOC target")
    p.add_argument("--seed", type=int, help="Random seed for deterministic output")
    p.add_argument("--out", type=Path, help="Path to save the generated code")
    args = p.parse_args()

    cfg = RustConfig(loc=args.loc, seed=args.seed)
    code = build_rust(cfg)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved generated Rust to {args.out}")
    else:
        sys.stdout.write(code)


if __name__ == "__main__":
    _cli()
