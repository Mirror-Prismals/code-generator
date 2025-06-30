#!/usr/bin/env python3
# synthetic_shell.py · v0.1.0
"""
Generate synthetic—yet syntactically valid—Bash shell scripts.

Major features
--------------
* Deterministic output with --seed
* Plugin architecture for new snippet generators
* No hidden global state
* --out to save directly to disk

Usage
-----
python synthetic_shell.py 200
python synthetic_shell.py 300 --seed 42 --out fake.sh
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

__version__ = "0.1.0"

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ShellConfig:
    loc: int = 200                 # approx. number of lines
    seed: Optional[int] = None
    weights: Dict[str, float] = field(default_factory=lambda: {
        "comment":    0.10,
        "shebang":    0.05,
        "var_decl":   0.15,
        "echo":       0.10,
        "func_def":   0.15,
        "if_stmt":    0.15,
        "loop":       0.15,
        "pipeline":   0.10,
        "file_op":    0.10,
    })

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
# Helpers
# ──────────────────────────────────────────────────────────────

LETTERS = "abcdefghijklmnopqrstuvwxyz"
def fresh_name(rng: random.Random, length: int = 5) -> str:
    return "".join(rng.choice(LETTERS) for _ in range(length))

def literal(rng: random.Random) -> str:
    if rng.random() < 0.5:
        return str(rng.randint(0, 100))
    else:
        # command substitution
        return f"$(date +%s)"

# ──────────────────────────────────────────────────────────────
# Generators
# ──────────────────────────────────────────────────────────────

@register("shebang")
def gen_shebang(state: Dict) -> str:
    # Only once at top
    if state["shebang_written"]:
        return ""
    state["shebang_written"] = True
    return "#!/usr/bin/env bash\n\n"

@register("comment")
def gen_comment(state: Dict) -> str:
    rng = state["rng"]
    tags = ["TODO", "FIXME", "NOTE", "HACK"]
    text = fresh_name(rng, rng.randint(3, 8))
    return f"# {rng.choice(tags)}: {text}\n"

@register("var_decl")
def gen_var_decl(state: Dict) -> str:
    rng = state["rng"]
    name = fresh_name(rng)
    val = literal(rng)
    return f"{name}={val}\n"

@register("echo")
def gen_echo(state: Dict) -> str:
    rng = state["rng"]
    msgs = ["Hello, world!", "Done.", "Processing...", "All good", "Can't stop"]
    return f'echo "{rng.choice(msgs)}"\n'

@register("func_def")
def gen_func_def(state: Dict) -> str:
    rng = state["rng"]
    name = fresh_name(rng)
    body = []
    # one or two statements inside
    for _ in range(rng.randint(1, 2)):
        stmt_kind = rng.choice(["var_decl", "echo", "pipeline"])
        body.append(_REGISTRY[stmt_kind](state).strip())
    inner = "\n    ".join(body)
    return f"function {name}() {{\n    {inner}\n}}\n\n"

@register("if_stmt")
def gen_if_stmt(state: Dict) -> str:
    rng = state["rng"]
    name = fresh_name(rng)
    # test existence or numeric compare
    if rng.random() < 0.5:
        cond = f"-f /etc/{name}"
    else:
        var = fresh_name(rng)
        cond = f"[ {var} -gt {rng.randint(0,10)} ]"
    then = f'echo "{name} ok"'
    oth  = f'echo "{name} fail"'
    return (
        f"if [ {cond} ]; then\n"
        f"    {then}\n"
        f"else\n"
        f"    {oth}\n"
        f"fi\n"
    )

@register("loop")
def gen_loop(state: Dict) -> str:
    rng = state["rng"]
    name = fresh_name(rng)
    count = rng.randint(2, 5)
    return (
        f"for i in $(seq 1 {count}); do\n"
        f"    echo \"{name}=$i\"\n"
        f"done\n"
    )

@register("pipeline")
def gen_pipeline(state: Dict) -> str:
    rng = state["rng"]
    cmd = rng.choice(["ls", "ps aux", "df -h", "uname -a"])
    pat = fresh_name(rng, rng.randint(2,5))
    return f"{cmd} | grep {pat}\n"

@register("file_op")
def gen_file_op(state: Dict) -> str:
    rng = state["rng"]
    op = rng.choice(["mkdir -p", "touch", "rm -f"])
    name = fresh_name(rng)
    return f"{op} /tmp/{name}\n"

# ──────────────────────────────────────────────────────────────
# Build & CLI
# ──────────────────────────────────────────────────────────────

def build_shell(cfg: ShellConfig) -> str:
    rng = random.Random(cfg.seed)
    state = {
        "cfg": cfg,
        "rng": rng,
        "shebang_written": False,
    }

    parts: List[str] = []
    lines = 0
    kinds, weights = zip(*cfg.weights.items())

    while lines < cfg.loc:
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        chunk = _REGISTRY[kind](state)
        if not chunk:
            continue
        parts.append(chunk)
        lines += chunk.count("\n")

    # ensure shebang at top
    if not state["shebang_written"]:
        parts.insert(0, gen_shebang(state))

    return "".join(parts)

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate a synthetic Bash script.")
    p.add_argument("loc", nargs="?", type=int, default=200, help="Approx. line count")
    p.add_argument("--seed", type=int, help="Random seed for deterministic output")
    p.add_argument("--out", type=Path, help="Path to save generated .sh")
    args = p.parse_args()

    cfg = ShellConfig(loc=args.loc, seed=args.seed)
    code = build_shell(cfg)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved generated shell to {args.out}")
    else:
        sys.stdout.write(code)

if __name__ == "__main__":
    _cli()
