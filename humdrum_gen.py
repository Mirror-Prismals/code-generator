#!/usr/bin/env python3
# synthetic_humdrum.py · v0.1.0
"""
Generate synthetic—yet well-formed—Humdrum **kern files with random notes.

Major features
--------------
* Deterministic output with --seed
* Configurable number of spines (voices)
* Configurable measures and notes per measure
* Random durations, pitches, accidentals, and octaves
* Well-formed measure bars and end-of-file marker
* --out to save directly to disk

Usage
-----
python synthetic_humdrum.py         # defaults to 2 spines, 4 measures, 8 notes each
python synthetic_humdrum.py --spines 3 --measures 5 --notes 6 --seed 42 --out music.krn
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

__version__ = "0.1.0"

@dataclass(frozen=True)
class HumdrumConfig:
    spines: int = 2
    measures: int = 4
    notes: int = 8
    seed: Optional[int] = None
    out: Path | None = None

DURATIONS = [1, 2, 4, 8, 16]
PITCHES = ["A", "B", "C", "D", "E", "F", "G"]
ACCIDENTALS = ["", "#", "-"]
OCTAVES = [2, 3, 4, 5]

def gen_kern_note(rng: random.Random) -> str:
    dur = rng.choice(DURATIONS)
    pitch = rng.choice(PITCHES)
    acc = rng.choice(ACCIDENTALS)
    octv = rng.choice(OCTAVES)
    # e.g. "4C#4" = quarter-note C-sharp in octave 4
    return f"{dur}{pitch}{acc}{octv}"

def build_humdrum(cfg: HumdrumConfig) -> str:
    rng = random.Random(cfg.seed)
    parts: List[str] = []
    # Header: spine declarations
    header = "\t".join(["**kern"] * cfg.spines)
    parts.append(header + "\n")
    # Generate measures
    for measure in range(1, cfg.measures + 1):
        # notes
        for _ in range(cfg.notes):
            line = "\t".join(gen_kern_note(rng) for _ in range(cfg.spines))
            parts.append(line + "\n")
        # barline
        parts.append("\t".join(["="] * cfg.spines) + "\n")
    # End-of-file
    parts.append("\t".join(["*-"] * cfg.spines) + "\n")
    return "".join(parts)

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate a synthetic Humdrum **kern file.")
    p.add_argument("--spines",    type=int,   default=2, help="Number of **kern spines (voices)")
    p.add_argument("--measures",  type=int,   default=4, help="Number of measures")
    p.add_argument("--notes",     type=int,   default=8, help="Notes per measure")
    p.add_argument("--seed",      type=int,   help="Random seed")
    p.add_argument("--out",       type=Path,  help="Path to save generated .krn")
    args = p.parse_args()

    cfg = HumdrumConfig(
        spines=args.spines,
        measures=args.measures,
        notes=args.notes,
        seed=args.seed,
        out=args.out,
    )
    krn = build_humdrum(cfg)

    if cfg.out:
        cfg.out.parent.mkdir(parents=True, exist_ok=True)
        cfg.out.write_text(krn, encoding="utf-8")
        print(f"✔ Saved synthetic Humdrum file to {cfg.out}")
    else:
        sys.stdout.write(krn)

if __name__ == "__main__":
    _cli()
