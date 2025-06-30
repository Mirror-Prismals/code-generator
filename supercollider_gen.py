#!/usr/bin/env python3
# synthetic_supercollider.py · v0.1.2
"""
Generate synthetic SuperCollider scripts with random SynthDefs and Patterns.

Major features
--------------
* Deterministic output with --seed
* Configurable approximate line count (--loc)
* Plugin architecture for snippet generators
* Random SynthDef definitions, Pattern sequences, Routines, and comments
* --out to save directly to disk
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

__version__ = "0.1.2"

@dataclass(frozen=True)
class SCConfig:
    loc: int = 100                  # approximate number of lines
    seed: int | None = None
    out: Path | None = None         # output file path

GeneratorFn = Callable[[Dict], str]
_REGISTRY: Dict[str, GeneratorFn] = {}

def register(kind: str):
    def inner(fn: GeneratorFn) -> GeneratorFn:
        if kind in _REGISTRY:
            raise ValueError(f"Duplicate generator: {kind}")
        _REGISTRY[kind] = fn
        return fn
    return inner

LETTERS = "abcdefghijklmnopqrstuvwxyz"
UGENS    = ["SinOsc.ar", "Saw.ar", "Pulse.ar", "LFSaw.ar", "Mix.ar"]
ENVS     = ["Env.perc", "Env.adsr", "Env.linen"]

class NameGen:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.used = set()
    def fresh(self, prefix: str = "", length: int = 6) -> str:
        for _ in range(1000):
            name = "".join(self.rng.choice(LETTERS) for _ in range(length))
            if name not in self.used:
                self.used.add(name)
                return prefix + name
        raise RuntimeError("Name space exhausted")

@register("comment")
def gen_comment(state: Dict) -> str:
    rng = state["rng"]
    memes = ["// TODO", "// FIXME", "// NOTE", "// HACK", "// ripples", "// shimmer"]
    return f"{rng.choice(memes)}: {state['names'].fresh(length=5)}\n"

@register("boot")
def gen_boot(state: Dict) -> str:
    if state.get("booted"):
        return ""
    state["booted"] = True
    return "Server.default.boot;\n\n"

@register("synthdef")
def gen_synthdef(state: Dict) -> str:
    rng = state["rng"]
    name = state["names"].fresh(prefix="synth_")
    state.setdefault("synths", []).append(name)
    ugen    = rng.choice(UGENS)
    env     = rng.choice(ENVS)
    freq    = rng.choice(["440", "440*2", "MouseX.kr(200,800)", "LFNoise0.kr(200).range(200,800)"])
    amp     = f"{rng.uniform(0.1,0.5):.2f}"
    release = rng.uniform(0.1,1.0)
    return (
        f"SynthDef(\\{name}, {{ |out=0, gate=1|\n"
        f"    var sig = {ugen}({freq}, 0, {amp});\n"
        f"    sig = sig * {env}(0.01, {release}, 1, -4).kr(gate);\n"
        f"    Out.ar(out, sig!2);\n"
        f"}}).add;\n\n"
    )

@register("pattern")
def gen_pattern(state: Dict) -> str:
    rng = state["rng"]
    synths = state.get("synths", [])
    if not synths:
        return ""
    name    = rng.choice(synths)
    pitches = [str(rng.randint(220,880)) for _ in range(rng.randint(4,8))]
    durs    = [str(rng.choice([0.25,0.5,1,1.5,2])) for _ in pitches]
    return (
        f"Pbind(\n"
        f"    \\instrument, \\{name},\n"
        f"    \\freq, Pseq([{', '.join(pitches)}], inf),\n"
        f"    \\dur,  Pseq([{', '.join(durs)}], inf)\n"
        f").play;\n\n"
    )

@register("routine")
def gen_routine(state: Dict) -> str:
    rng  = state["rng"]
    text = state['names'].fresh(length=6)
    func = rng.choice(["Postln", "postln"])
    return (
        "Routine({\n"
        f"    10.do({{ arg i; {func}(\"{text} \" ++ i); }});\n"
        "}).play;\n\n"
    )

@register("pause")
def gen_pause(state: Dict) -> str:
    rng = state["rng"]
    return f"{rng.choice(['0.5.wait;', '1.wait;', '2.wait;'])}\n"

def build_sc(cfg: SCConfig) -> str:
    rng   = random.Random(cfg.seed)
    names = NameGen(rng)
    state: Dict = {"rng": rng, "names": names}
    parts: List[str] = []
    lines = 0
    kinds, weights = zip(*{
        "comment":  0.05,
        "boot":     0.05,
        "synthdef": 0.30,
        "pattern":  0.25,
        "routine":  0.20,
        "pause":    0.15,
    }.items())

    while lines < cfg.loc:
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        snippet = _REGISTRY[kind](state)
        if not snippet:
            continue
        parts.append(snippet)
        lines += snippet.count("\n")

    return "".join(parts)

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic SuperCollider code.")
    p.add_argument("loc", nargs="?", type=int, default=100, help="Approx. line count")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out", type=Path, help="Path to save generated .scd")
    args = p.parse_args()

    cfg = SCConfig(loc=args.loc, seed=args.seed, out=args.out)
    code = build_sc(cfg)

    if cfg.out:
        cfg.out.parent.mkdir(parents=True, exist_ok=True)
        cfg.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved synthetic SuperCollider code to {cfg.out}")
    else:
        sys.stdout.write(code)

if __name__ == "__main__":
    _cli()
