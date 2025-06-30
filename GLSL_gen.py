#!/usr/bin/env python3
# synthetic_glsl.py · v0.1.0
"""
Generate synthetic—yet syntactically valid—GLSL shader files.

Major features
--------------
* Deterministic output with --seed
* Plugin architecture for new snippet generators
* No hidden global state
* Randomly creates either a vertex or fragment shader
* --out to save directly to disk

Usage
-----
python synthetic_glsl.py 200
python synthetic_glsl.py 300 --seed 42 --out fake_vert.vert
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
class GLSLConfig:
    loc: int = 200
    seed: Optional[int] = None
    weights: Dict[str, float] = field(default_factory=lambda: {
        "comment":    0.10,
        "version":    0.05,
        "precision":  0.05,
        "uniform":    0.15,
        "in_var":     0.15,
        "out_var":    0.10,
        "function":   0.20,
        "main":       0.20,
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
def fresh_name(rng: random.Random, length: int = 6) -> str:
    return "".join(rng.choice(LETTERS) for _ in range(length))

@register("comment")
def gen_comment(state: Dict) -> str:
    rng = state["rng"]
    tags = ["// TODO", "// FIXME", "// NOTE", "// HACK"]
    return rng.choice(tags) + f": {fresh_name(rng, 8)}\n"

@register("version")
def gen_version(state: Dict) -> str:
    if state["version_written"]:
        return ""
    state["version_written"] = True
    return "#version 330 core\n\n"

@register("precision")
def gen_precision(state: Dict) -> str:
    # only for fragment shaders
    if state["shader_type"] != "fragment" or state["precision_written"]:
        return ""
    state["precision_written"] = True
    return "precision mediump float;\n\n"

@register("uniform")
def gen_uniform(state: Dict) -> str:
    rng = state["rng"]
    ty = rng.choice(["float","vec2","vec3","vec4","mat4","sampler2D"])
    name = fresh_name(rng)
    state["uniforms"].append(name)
    return f"uniform {ty} {name};\n"

@register("in_var")
def gen_in_var(state: Dict) -> str:
    rng = state["rng"]
    ty = rng.choice(["vec2","vec3","vec4"])
    name = fresh_name(rng)
    state["in_vars"].append(name)
    return f"in {ty} {name};\n"

@register("out_var")
def gen_out_var(state: Dict) -> str:
    rng = state["rng"]
    if state["shader_type"] == "fragment":
        # Fragment output variable
        name = "fragColor"
        ty = "vec4"
    else:
        # Vertex varying
        name = fresh_name(rng)
        ty = rng.choice(["vec2","vec3","vec4"])
    state["out_vars"].append(name)
    return f"out {ty} {name};\n"

@register("function")
def gen_function(state: Dict) -> str:
    rng = state["rng"]
    ret = rng.choice(["float","vec2","vec3","vec4"])
    fname = fresh_name(rng)
    pty = rng.choice(["float","vec2","vec3"])
    pname = fresh_name(rng)
    state["functions"].append(fname)
    expr = rng.choice(["sin","cos","length","normalize"])
    return (
        f"{ret} {fname}({pty} {pname}) {{\n"
        f"    return {expr}({pname});\n"
        f"}}\n"
    )

@register("main")
def gen_main(state: Dict) -> str:
    if state["main_written"]:
        return ""
    state["main_written"] = True
    lines: List[str] = ["\nvoid main() {"]
    if state["shader_type"] == "vertex":
        if state["in_vars"]:
            pos = state["in_vars"][0]
            lines.append(f"    gl_Position = vec4({pos}, 1.0);")
        else:
            lines.append("    gl_Position = vec4(0.0);")
    else:
        outv = state["out_vars"][0] if state["out_vars"] else "fragColor"
        lines.append(f"    {outv} = vec4(1.0);")
    lines.append("}")
    return "\n".join(lines) + "\n"

def build_glsl(cfg: GLSLConfig) -> str:
    rng = random.Random(cfg.seed)
    shader_type = rng.choice(["vertex","fragment"])
    state = {
        "rng": rng,
        "shader_type": shader_type,
        "version_written": False,
        "precision_written": False,
        "main_written": False,
        "uniforms": [],
        "in_vars": [],
        "out_vars": [],
        "functions": [],
    }
    parts: List[str] = [f"// Auto-generated {shader_type} shader\n"]
    lines = parts[0].count("\n")
    kinds, weights = zip(*cfg.weights.items())

    while lines < cfg.loc:
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        chunk = _REGISTRY[kind](state)
        if not chunk:
            continue
        parts.append(chunk)
        lines += chunk.count("\n")

    # ensure version & main exist
    if not state["version_written"]:
        parts.insert(1, gen_version(state))
    if not state["main_written"]:
        parts.append(gen_main(state))

    return "".join(parts)

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic GLSL shaders.")
    p.add_argument("loc", nargs="?", type=int, default=200, help="Approx. line count")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out", type=Path, help="Path to save .vert or .frag")
    args = p.parse_args()

    cfg = GLSLConfig(loc=args.loc, seed=args.seed)
    code = build_glsl(cfg)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved generated shader to {args.out}")
    else:
        sys.stdout.write(code)

if __name__ == "__main__":
    _cli()
