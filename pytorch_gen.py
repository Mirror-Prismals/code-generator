#!/usr/bin/env python3
# synthetic_pytorch.py · v0.1.0
"""
Generate synthetic—yet syntactically valid—PyTorch code snippets.

Major features
--------------
* Deterministic output with --seed
* Plugin architecture for new snippet generators
* No hidden global state
* --out to save directly to disk

Usage
-----
python synthetic_pytorch.py 200
python synthetic_pytorch.py 300 --seed 42 --out fake_pytorch.py
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
class PTConfig:
    loc: int = 200                  # approximate number of lines
    seed: Optional[int] = None

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

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def fresh_name(rng: random.Random, length: int = 6) -> str:
    return "".join(rng.choice(LETTERS) for _ in range(length))

# ──────────────────────────────────────────────────────────────
# Generators
# ──────────────────────────────────────────────────────────────

@register("comment")
def gen_comment(state: Dict) -> str:
    rng = state["rng"]
    tags = ["# TODO", "# FIXME", "# NOTE", "# HACK"]
    return f"{rng.choice(tags)}: {fresh_name(rng,4)}\n"

@register("import")
def gen_import(state: Dict) -> str:
    if state.get("imports_written"):
        return ""
    state["imports_written"] = True
    return (
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.optim as optim\n"
        "from torch.utils.data import DataLoader, TensorDataset\n\n"
    )

@register("model_class")
def gen_model_class(state: Dict) -> str:
    rng = state["rng"]
    name = "Model" + fresh_name(rng,4)
    state["model_class_written"] = True
    # choose number of layers
    n_layers = rng.randint(1,3)
    layers = []
    in_dim = rng.choice([16,32,64])
    for i in range(n_layers):
        out_dim = rng.choice([16,32,64])
        layers.append(f"        nn.Linear({in_dim}, {out_dim}),")
        layers.append("        nn.ReLU(),")
        in_dim = out_dim
    layers.append(f"        nn.Linear({in_dim}, {rng.choice([1,10,100])})")
    seq = "\n".join(layers)
    return (
        f"class {name}(nn.Module):\n"
        f"    def __init__(self):\n"
        f"        super().__init__()\n"
        f"        self.net = nn.Sequential(\n"
        f"{seq}\n"
        f"        )\n\n"
        f"    def forward(self, x):\n"
        f"        return self.net(x)\n\n"
    )

@register("loss_fn")
def gen_loss_fn(state: Dict) -> str:
    if state.get("loss_fn_written"):
        return ""
    state["loss_fn_written"] = True
    return "loss_fn = nn.MSELoss()\n\n"

@register("optimizer")
def gen_optimizer(state: Dict) -> str:
    if state.get("optimizer_written"):
        return ""
    state["optimizer_written"] = True
    return "optimizer = optim.SGD(model.parameters(), lr=0.01)\n\n"

@register("data_loader")
def gen_data_loader(state: Dict) -> str:
    if state.get("data_loader_written"):
        return ""
    state["data_loader_written"] = True
    return (
        "# dummy dataset\n"
        "x = torch.randn(100, 16)\n"
        "y = torch.randn(100, 1)\n"
        "dataset = TensorDataset(x, y)\n"
        "loader = DataLoader(dataset, batch_size=10, shuffle=True)\n\n"
    )

@register("train_loop")
def gen_train_loop(state: Dict) -> str:
    if state.get("train_loop_written"):
        return ""
    state["train_loop_written"] = True
    return (
        "def train(model, loader, optimizer, loss_fn, epochs=5):\n"
        "    model.train()\n"
        "    for epoch in range(epochs):\n"
        "        total_loss = 0.0\n"
        "        for xb, yb in loader:\n"
        "            pred = model(xb)\n"
        "            loss = loss_fn(pred, yb)\n"
        "            optimizer.zero_grad()\n"
        "            loss.backward()\n"
        "            optimizer.step()\n"
        "            total_loss += loss.item()\n"
        "        print(f\"Epoch {epoch+1}, Loss: {total_loss:.4f}\")\n\n"
    )

@register("tensor_op")
def gen_tensor_op(state: Dict) -> str:
    rng = state["rng"]
    shape = rng.choice([(3,3), (2,4), (4,2)])
    return f"x = torch.randn{shape}\n"

@register("main_block")
def gen_main_block(state: Dict) -> str:
    if state.get("main_written"):
        return ""
    state["main_written"] = True
    return (
        "if __name__ == '__main__':\n"
        "    model = ModelXXXX()\n"
        "    # replace ModelXXXX with actual class name\n"
        "    # e.g. model = ModelABCD()\n"
        "    train(model, loader, optimizer, loss_fn)\n"
    )

# ──────────────────────────────────────────────────────────────
# Build & CLI
# ──────────────────────────────────────────────────────────────

def build_pytorch(cfg: PTConfig) -> str:
    rng = random.Random(cfg.seed)
    state: Dict = {"rng": rng}
    parts: List[str] = []
    lines = 0
    kinds, weights = zip(*cfg.__dict__.get("weights", {
        "comment":       0.05,
        "import":        0.10,
        "model_class":   0.20,
        "loss_fn":       0.10,
        "optimizer":     0.10,
        "data_loader":   0.15,
        "train_loop":    0.15,
        "tensor_op":     0.10,
        "main_block":    0.05,
    }.items()))

    while lines < cfg.loc:
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        gen = _REGISTRY.get(kind)
        if not gen:
            continue
        chunk = gen(state)
        if not chunk:
            continue
        parts.append(chunk)
        lines += chunk.count("\n")

    # fix placeholder in main
    # find model class name
    for p in parts:
        if p.startswith("class ") and "(nn.Module)" in p:
            cls = p.split()[1].split("(")[0]
            parts = [c.replace("ModelXXXX", cls) for c in parts]
            break

    return "".join(parts)

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic PyTorch code.")
    p.add_argument("loc", nargs="?", type=int, default=200, help="Approximate line count")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out", type=Path, help="Path to save generated .py")
    args = p.parse_args()

    cfg = PTConfig(loc=args.loc, seed=args.seed)
    code = build_pytorch(cfg)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved generated PyTorch code to {args.out}")
    else:
        sys.stdout.write(code)

if __name__ == "__main__":
    _cli()
