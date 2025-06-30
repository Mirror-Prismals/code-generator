#!/usr/bin/env python3
# synthetic_cuda.py · v0.1.0
"""
Generate synthetic—yet syntactically valid—CUDA C++ source files.

Major features
--------------
* Deterministic output with --seed
* Plugin architecture for new snippet generators
* No hidden global state
* --out to save directly to disk

Usage
-----
python synthetic_cuda.py 200
python synthetic_cuda.py 300 --seed 42 --out fake.cu
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
class CudaConfig:
    loc: int = 200
    seed: Optional[int] = None
    weights: Dict[str, float] = field(default_factory=lambda: {
        "comment":       0.10,
        "include":       0.10,
        "kernel_def":    0.20,
        "device_func":   0.10,
        "host_alloc":    0.10,
        "host_copy":     0.10,
        "kernel_launch": 0.15,
        "loop":          0.10,
        "sync":          0.05,
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

@register("comment")
def gen_comment(state: Dict) -> str:
    rng = state["rng"]
    tags = ["TODO", "FIXME", "NOTE", "HACK"]
    text = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(4,12)))
    return f"// {rng.choice(tags)}: {text}\n"

@register("include")
def gen_include(state: Dict) -> str:
    rng = state["rng"]
    opts = [
        "#include <cuda_runtime.h>",
        "#include <device_launch_parameters.h>",
        "#include <cstdio>",
        "#include <cstdlib>"
    ]
    inc = rng.choice(opts)
    return inc + "\n"

@register("kernel_def")
def gen_kernel_def(state: Dict) -> str:
    rng = state["rng"]
    idx = state["kernel_count"]
    state["kernel_count"] += 1
    t = rng.choice(["float", "int"])
    name = f"kernel{idx}"
    n = rng.randint(1, 4)
    params = ", ".join(f"{t} *d_{rng.choice(['a','b','c','d'])}{i}" for i in range(n))
    body = []
    body.append("    int idx = blockIdx.x * blockDim.x + threadIdx.x;")
    if rng.random() < 0.5:
        body.append("    // simple compute")
        body.append(f"    d_{rng.choice(['a','b','c','d'])}0[idx] = idx;")
    else:
        body.append("    // loop inside kernel")
        body.append("    for(int i=0; i<blockDim.x; ++i) {")
        body.append("        d_a0[idx] += i;")
        body.append("    }")
    return (
        f"__global__ void {name}({params}) {{\n" +
        "\n".join("    " + line for line in body) + "\n}\n\n"
    )

@register("device_func")
def gen_device_func(state: Dict) -> str:
    rng = state["rng"]
    name = f"devFunc{rng.randint(1,10)}"
    return (
        f"__device__ int {name}(int x) {{\n"
        "    return x * x;\n"
        "}\n\n"
    )

@register("host_alloc")
def gen_host_alloc(state: Dict) -> str:
    rng = state["rng"]
    t = rng.choice(["float", "int"])
    name = f"h_{rng.choice(['a','b','c','d'])}{rng.randint(0,3)}"
    size = rng.randint(64, 256)
    return (
        f"{t} *{name};\n"
        f"cudaMalloc(&{name}, sizeof({t}) * {size});\n"
    )

@register("host_copy")
def gen_host_copy(state: Dict) -> str:
    rng = state["rng"]
    src = rng.choice(["h_a0","h_b1","h_c2","h_d3"])
    dst = rng.choice(["d_a0","d_b1","d_c2","d_d3"])
    size = rng.randint(64,256)
    dir_ = rng.choice(["cudaMemcpyHostToDevice","cudaMemcpyDeviceToHost"])
    return f"cudaMemcpy({dst}, {src}, sizeof(*{src}) * {size}, {dir_});\n"

@register("kernel_launch")
def gen_launch(state: Dict) -> str:
    rng = state["rng"]
    idx = rng.randint(0, state["kernel_count"] - 1) if state["kernel_count"] else 0
    name = f"kernel{idx}"
    b = rng.randint(1, 16)
    t = rng.choice([32, 64, 128])
    args = ", ".join(rng.choice(["d_a0","d_b1","d_c2","d_d3"]) for _ in range(rng.randint(1,3)))
    return f"{name}<<<{b},{t}>>>({args});\n"

@register("loop")
def gen_loop(state: Dict) -> str:
    rng = state["rng"]
    cnt = rng.randint(1,5)
    return f"for(int i=0; i<{cnt}; ++i) {{ /* do nothing */ }}\n"

@register("sync")
def gen_sync(state: Dict) -> str:
    return "cudaDeviceSynchronize();\n\n"

def build_cuda(cfg: CudaConfig) -> str:
    rng = random.Random(cfg.seed)
    state = {
        "cfg": cfg,
        "rng": rng,
        "kernel_count": 0
    }
    parts: List[str] = ["// Auto-generated CUDA C++\n\n"]
    lines = parts[0].count("\n")
    kinds, weights = zip(*cfg.weights.items())

    # ensure includes at top
    for _ in range(3):
        parts.append(gen_include(state))
        lines += 1

    # main stub
    parts.append("int main() {\n")
    lines += 1

    while lines < cfg.loc - 3:
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        snippet = _REGISTRY[kind](state)
        if not snippet:
            continue
        # indent host code except comments
        if kind not in ("comment",):
            snippet = "".join("    " + line for line in snippet.splitlines(True))
        parts.append(snippet)
        lines += snippet.count("\n")

    parts.append("    return 0;\n}\n")
    return "".join(parts)

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic CUDA C++ (.cu) file.")
    p.add_argument("loc", nargs="?", type=int, default=200, help="Approximate line count")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out", type=Path, help="Path to save generated .cu")
    args = p.parse_args()

    cfg = CudaConfig(loc=args.loc, seed=args.seed)
    code = build_cuda(cfg)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved generated CUDA to {args.out}")
    else:
        sys.stdout.write(code)

if __name__ == "__main__":
    _cli()
