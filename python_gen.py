#!/usr/bin/env python3
# synthetic.py · v2.1.0
"""
Generate synthetic—yet syntactically valid—Python source files.

Major features
--------------
* Deterministic output with --seed
* Plugin architecture for new statement generators
* No hidden global state
* Optional compile-time/smoke-test
* --out to save directly to disk

CLI
---
python synthetic.py 400
python synthetic.py 800 --seed 123 --out fake.py
python synthetic.py 600 --skip-compile --no-smoke-test
"""

from __future__ import annotations

import argparse
import importlib.util as import_util
import keyword
import random
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Sequence

__all__ = ["Config", "build"]
__version__ = "2.1.0"

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class Config:
    loc: int = 400
    seed: int | None = None
    max_literal_depth: int = 3
    import_libs: Sequence[str] = ("json", "random", "sys")
    meme_comments: Sequence[str] = (
        "TODO: rewrite this in Rust",
        "FIXME: optimize later",
        "NOTE: legacy workaround – do NOT touch!",
        r"¯\_(ツ)_/¯",
        "left as an exercise for the reader",
        "tfw no unit tests",
    )
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "comment": 0.15,
            "assignment": 0.25,
            "function": 0.30,
            "class": 0.30,
        }
    )
    max_functions: int | None = None
    max_classes: int | None = None


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


class ImportManager:
    def __init__(self, libs: Sequence[str]) -> None:
        self._libs: list[str] = list(dict.fromkeys(libs))

    def add(self, lib: str) -> None:
        if lib not in self._libs:
            self._libs.append(lib)

    def render(self) -> str:
        return "".join(f"import {lib}\n" for lib in self._libs) + "\n"


class NameGenerator:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.reserved: set[str] = set(keyword.kwlist) | set(dir(__builtins__))

    def fresh(self, *, min_len: int = 3, max_len: int = 10, capital: bool = False) -> str:
        for _ in range(10_000):
            name = "".join(self.rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(self.rng.randint(min_len, max_len)))
            if capital:
                name = name.capitalize()
            if name not in self.reserved:
                self.reserved.add(name)
                return name
        raise RuntimeError("Identifier space exhausted")


# ──────────────────────────────────────────────────────────────
# Generator registry
# ──────────────────────────────────────────────────────────────

GeneratorFn = Callable[["Context"], str]
_REGISTRY: Dict[str, GeneratorFn] = {}


def register(kind: str) -> Callable[[GeneratorFn], GeneratorFn]:
    def inner(fn: GeneratorFn) -> GeneratorFn:
        if kind in _REGISTRY:
            raise ValueError(f"Duplicate generator: {kind}")
        _REGISTRY[kind] = fn
        return fn

    return inner


# ──────────────────────────────────────────────────────────────
# Context passed to generators
# ──────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Context:
    cfg: Config
    rng: random.Random
    names: NameGenerator
    imports: ImportManager
    symbols: Dict[str, set]

    # literals & comments ---------------------------------------------------

    def literal(self, depth: int = 0) -> str:
        if depth >= self.cfg.max_literal_depth or self.rng.random() < 0.4:
            return str(self.rng.randint(0, 9999) if self.rng.random() < 0.5 else round(self.rng.uniform(0, 100), 2))
        lhs = self.literal(depth + 1)
        rhs = self.literal(depth + 1)
        op = self.rng.choice(["+", "-", "*", "//", "%", "**"])
        return f"({lhs} {op} {rhs})"

    def meme_comment(self) -> str:
        return f"# {self.rng.choice(self.cfg.meme_comments)}\n"


# ──────────────────────────────────────────────────────────────
# Statement generators
# ──────────────────────────────────────────────────────────────


@register("comment")
def gen_comment(ctx: Context) -> str:
    return ctx.meme_comment()


@register("assignment")
def gen_assignment(ctx: Context) -> str:
    name = ctx.names.fresh()
    ctx.symbols["variables"].add(name)
    return f"{name} = {ctx.literal()}\n"


@register("function")
def gen_function(ctx: Context) -> str:
    if ctx.cfg.max_functions and len(ctx.symbols["functions"]) >= ctx.cfg.max_functions:
        return ""
    fname = ctx.names.fresh()
    params = [ctx.names.fresh() for _ in range(ctx.rng.randint(0, 3))]
    ctx.symbols["functions"].add(fname)

    body: list[str] = []
    if ctx.rng.random() < 0.4:
        body.append(f'    """{ctx.rng.choice(ctx.cfg.meme_comments)}"""\n')
    if ctx.rng.random() < 0.3:
        body.append("    " + ctx.meme_comment())
    return_expr = ctx.rng.choice(params) if params and ctx.rng.random() < 0.6 else ctx.literal()
    body.append(f"    return {return_expr}\n")

    signature = f"def {fname}({', '.join(params)}):\n" if params else f"def {fname}():\n"
    return signature + "".join(body) + "\n"


@register("class")
def gen_class(ctx: Context) -> str:
    if ctx.cfg.max_classes and len(ctx.symbols["classes"]) >= ctx.cfg.max_classes:
        return ""
    cname = ctx.names.fresh(capital=True)
    ctx.symbols["classes"].add(cname)

    def gen_method() -> str:
        mname = ctx.names.fresh()
        ctx.symbols["functions"].add(f"{cname}.{mname}")
        return (
            f"    def {mname}(self):\n"
            f"        {ctx.meme_comment() if ctx.rng.random() < 0.25 else ''}"
            f"        return self._state\n\n"
        )

    methods = "".join(gen_method() for _ in range(ctx.rng.randint(1, 3)))
    return (
        f"class {cname}:\n"
        f'    """Auto-generated synthetic class."""\n\n'
        f"    def __init__(self):\n        self._state = 0\n\n"
        f"{methods}\n"
    )


# ──────────────────────────────────────────────────────────────
# Build pipeline
# ──────────────────────────────────────────────────────────────


def build(cfg: Config) -> str:
    rng = random.Random(cfg.seed)
    ctx = Context(
        cfg=cfg,
        rng=rng,
        names=NameGenerator(rng),
        imports=ImportManager(cfg.import_libs),
        symbols={"functions": set(), "classes": set(), "variables": set()},
    )

    parts: List[str] = [
        '"""\nSynthetic module – auto-generated.\n"""\n\n',
        ctx.imports.render(),
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

    parts.append("if __name__ == '__main__':\n")
    if ctx.symbols["functions"]:
        parts.append(f"    print({rng.choice(tuple(ctx.symbols['functions']))}())\n")
    if ctx.symbols["classes"]:
        cls = rng.choice(tuple(ctx.symbols["classes"]))
        parts.append(f"    obj = {cls}()\n    print(obj.__class__.__name__, obj._state)\n")

    return "".join(parts)


# ──────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────


def smoke_test(code: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "fake_mod.py"
        path.write_text(code, encoding="utf-8")
        spec = import_util.spec_from_file_location("fake_mod", str(path))
        if spec is None or spec.loader is None:
            raise ImportError("Failed to load generated module")
        mod = import_util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]

        callables = [getattr(mod, n) for n in dir(mod) if callable(getattr(mod, n))]
        if callables:
            callables[0]()  # pragma: no cover


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────


def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate a synthetic Python module.")
    p.add_argument("loc", nargs="?", type=int, default=400, help="Approximate LOC target")
    p.add_argument("--seed", type=int, help="Random seed for deterministic output")
    p.add_argument("--max-funcs", type=int, help="Maximum number of functions")
    p.add_argument("--max-classes", type=int, help="Maximum number of classes")
    p.add_argument("--out", type=Path, help="Path to save the generated code")
    p.add_argument("--skip-compile", action="store_true", help="Skip compile() and runtime smoke test")
    p.add_argument("--no-smoke-test", action="store_true", help="Skip runtime smoke test (implies compile)")
    args = p.parse_args()

    cfg = Config(
        loc=args.loc,
        seed=args.seed,
        max_functions=args.max_funcs,
        max_classes=args.max_classes,
    )
    code = build(cfg)

    # compile + smoke-test unless explicitly skipped
    if not args.skip_compile:
        try:
            compile(code, "<generated>", "exec")
            if not args.no_smoke_test:
                smoke_test(code)
        except Exception as exc:
            print("Generation failed:", exc, file=sys.stderr)
            sys.exit(1)

    # Output handling
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved generated module to {args.out}")
    else:
        print(code)


if __name__ == "__main__":
    _cli()
