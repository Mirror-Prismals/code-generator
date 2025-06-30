#!/usr/bin/env python3
# synthetic_diagram.py · v0.1.1
"""
Generate synthetic—yet grammatically valid—sentence diagrams
using the typable Rod & Staff style: subject | verb [| object].

Major features
--------------
* Deterministic output with --seed
* Plugin architecture for sentence‐type generators
* No hidden global state
* --out to save directly to disk

Usage
-----
python synthetic_diagram.py 10
python synthetic_diagram.py 20 --seed 42 --out diagrams.txt
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

__version__ = "0.1.1"

@dataclass(frozen=True, slots=True)
class DiagramConfig:
    loc: int = 10                     # Approximate number of diagrams
    seed: Optional[int] = None        # Random seed for reproducibility
    nouns: List[str] = field(default_factory=lambda: [
        "The cat", "The dog", "Birds", "My sister", "The farmer",
        "Children", "The teacher", "Alice", "The car", "The tree",
        "The student", "The rabbit", "My brother", "The bird",
        "The gardener", "The doctor", "The king", "The queen",
        "The knight", "The princess", "The lion", "The tiger",
        "The computer", "The programmer", "The artist", "The musician",
        "The dancer", "The athlete", "The baker", "The chef",
        "The driver", "The painter", "The florist", "The writer",
        "The pilot", "The captain", "The soldier", "The plumber",
    ])
    verbs_intransitive: List[str] = field(default_factory=lambda: [
        "slept", "ran", "jumped", "smiled", "laughed",
        "sang", "sat", "cried", "shone", "fell",
        "danced", "walked", "crawled", "hopped", "flew",
        "swam", "waited", "tiptoed", "trembled", "sighed",
        "sneezed", "coughed", "yawned", "blinked", "blossomed",
        "rustled", "glanced", "nodded", "wandered", "paced",
        "hovered", "wobbled", "whispered", "shouted", "echoed",
        "glowed", "faded", "melted", "sparkled",
    ])
    verbs_transitive: List[str] = field(default_factory=lambda: [
        "chased", "found", "built", "baked", "kicked",
        "drove", "opened", "painted", "wrote", "read",
        "ate", "threw", "caught", "saw", "helped",
        "held", "took", "brought", "sent", "sold",
        "gave", "heard", "taught", "told", "showed",
        "created", "bought", "drank", "fed", "folded",
        "fixed", "moved", "pressed", "pulled", "pushed",
        "wore", "wrote", "sent", "smelled",
    ])
    objects: List[str] = field(default_factory=lambda: [
        "the mouse", "a ball", "a cake", "the tractor",
        "a book", "the fence", "a song", "the letter",
        "shells", "books", "a flower", "the picture",
        "the door", "the window", "a house", "the bag",
        "the toy", "the pen", "the pencil", "the computer",
        "a sandwich", "the meal", "the movie", "the game",
        "a gift", "the apple", "the orange", "the banana",
        "the cup", "the glass", "the bottle", "the chair",
        "the table", "the road", "the box", "the key",
        "the phone", "the camera", "the map", "the brush",
    ])
    weights: Dict[str, float] = field(default_factory=lambda: {
        "intransitive": 0.5,
        "transitive":   0.5,
    })

GeneratorFn = Callable[[Dict], str]
_REGISTRY: Dict[str, GeneratorFn] = {}

def register(kind: str) -> Callable[[GeneratorFn], GeneratorFn]:
    def inner(fn: GeneratorFn) -> GeneratorFn:
        if kind in _REGISTRY:
            raise ValueError(f"Duplicate generator registration: {kind}")
        _REGISTRY[kind] = fn
        return fn
    return inner

class NameGen:
    def __init__(self, rng: random.Random, items: List[str]):
        self.rng = rng
        self.items = items

    def choice(self) -> str:
        return self.rng.choice(self.items)

@register("intransitive")
def gen_intransitive(state: Dict) -> str:
    """Generate an intransitive sentence diagram: subject | verb"""
    subj = state["subj_gen"].choice()
    verb = state["rng"].choice(state["config"].verbs_intransitive)
    return f"{subj} | {verb}."

@register("transitive")
def gen_transitive(state: Dict) -> str:
    """Generate a transitive sentence diagram: subject | verb | object"""
    subj = state["subj_gen"].choice()
    verb = state["rng"].choice(state["config"].verbs_transitive)
    obj  = state["obj_gen"].choice()
    return f"{subj} | {verb} | {obj}."

def build_diagrams(cfg: DiagramConfig) -> str:
    rng = random.Random(cfg.seed)
    subj_gen = NameGen(rng, cfg.nouns)
    obj_gen  = NameGen(rng, cfg.objects)
    state = {
        "rng":       rng,
        "config":    cfg,
        "subj_gen":  subj_gen,
        "obj_gen":   obj_gen,
    }

    output_lines: List[str] = []
    kinds, weights = zip(*cfg.weights.items())
    while len(output_lines) < cfg.loc:
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        diagram = _REGISTRY[kind](state)
        output_lines.append(diagram)
    return "\n".join(output_lines) + "\n"

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic sentence diagrams.")
    p.add_argument("loc", nargs="?", type=int, default=10,
                   help="Approximate number of diagrams to generate")
    p.add_argument("--seed", type=int, help="Random seed for reproducible output")
    p.add_argument("--out", type=Path, help="File path to save diagrams")
    args = p.parse_args()

    cfg = DiagramConfig(loc=args.loc, seed=args.seed)
    text = build_diagrams(cfg)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"✔ Saved {cfg.loc} diagrams to {args.out}")
    else:
        sys.stdout.write(text)

if __name__ == "__main__":
    _cli()
