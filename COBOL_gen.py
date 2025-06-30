#!/usr/bin/env python3
# synthetic_cobol.py · v0.1.0
"""
Generate synthetic—yet syntactically valid—COBOL source files.

Major features
--------------
* Deterministic output with --seed
* Approximate line-count control (--loc)
* IDENTIFICATION / ENVIRONMENT / DATA / PROCEDURE divisions
* Random PIC variables, literals, MOVE / ADD / IF / PERFORM constructs
* Simple paragraph structure with GO TO / PERFORM
* Writes a runnable program ending with STOP RUN
* --out to save directly to disk

Usage
-----
python synthetic_cobol.py 200
python synthetic_cobol.py 300 --seed 42 --loc 250 --out fake.cob
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

__version__ = "0.1.0"

# ───────────────────────── configuration
@dataclass(frozen=True)
class CobConfig:
    loc : int = 200
    seed: int | None = None
    out : Path | None = None

# ───────────────────────── helpers
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def col8(txt: str) -> str:
    """Place text in COBOL area A/B starting at column 8."""
    return f"       {txt}\n"

def rand_word(rng: random.Random, length: int = 6) -> str:
    return "".join(rng.choice(LETTERS) for _ in range(length))

# ───────────────────────── generator
def build_cobol(cfg: CobConfig) -> str:
    rng = random.Random(cfg.seed)
    lines: List[str] = []

    # IDENTIFICATION DIVISION --------------------------------------------
    lines.append(col8("IDENTIFICATION DIVISION."))
    lines.append(col8(f"PROGRAM-ID. {rand_word(rng,8)}."))
    lines.append(col8("AUTHOR. SYNTHETIC-GEN."))

    # ENVIRONMENT DIVISION ------------------------------------------------
    lines.append(col8("ENVIRONMENT DIVISION."))
    lines.append(col8("CONFIGURATION SECTION."))
    lines.append(col8("SOURCE-COMPUTER.  SYNTHETIC."))
    lines.append(col8("OBJECT-COMPUTER.  SYNTHETIC."))

    # DATA DIVISION / WORKING-STORAGE ------------------------------------
    lines.append(col8("DATA DIVISION."))
    lines.append(col8("WORKING-STORAGE SECTION."))

    var_names = []
    for _ in range(rng.randint(4, 8)):
        name = rand_word(rng, 6)
        pic  = rng.choice(["PIC 9(3).", "PIC 9(5).", "PIC X(10)."])
        lines.append(col8(f"01  {name}          {pic}"))
        var_names.append(name)

    # PROCEDURE DIVISION header
    lines.append(col8("PROCEDURE DIVISION."))
    main_par = "MAIN-PARA"
    lines.append(col8(f"{main_par}."))
    # simple moves / adds
    for name in var_names:
        if rng.random() < .5:
            val = rng.randint(1, 999)
            lines.append(col8(f"    MOVE {val} TO {name}."))
        else:
            src = rng.choice(var_names)
            lines.append(col8(f"    ADD {src} TO {name}."))
    # IF block
    a, b = rng.sample(var_names, 2)
    lines.append(col8(f"    IF {a} > {b}"))
    lines.append(col8("       DISPLAY \"GREATER\""))
    lines.append(col8("    ELSE"))
    lines.append(col8("       DISPLAY \"NOT-GREATER\""))
    lines.append(col8("    END-IF."))
    # PERFORM loop paragraph
    loop_par = "LOOP-PARA"
    idx      = rand_word(rng,4)
    lines.append(col8(f"    MOVE 1 TO {idx}."))
    lines.append(col8(f"    PERFORM VARYING {idx} FROM 1 BY 1 UNTIL {idx} > 5"))
    lines.append(col8(f"        PERFORM {loop_par}"))
    lines.append(col8("    END-PERFORM."))
    lines.append(col8("    STOP RUN."))

    # loop paragraph definition
    lines.append(col8(f"{loop_par}."))
    tgt = rng.choice(var_names)
    lines.append(col8(f"    DISPLAY \"LOOP\", {tgt}."))
    lines.append(col8("    EXIT."))

    # pad / trim to approx target
    while len(lines) < cfg.loc:
        lines.insert(-3, col8(f"* COMMENT {rand_word(rng,5)}"))

    return "".join(lines[:cfg.loc])

# ───────────────────────── CLI
def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic COBOL program.")
    p.add_argument("loc", nargs="?", type=int, default=200,
                   help="Approximate line count")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out",  type=Path, help="Path to save generated .cob")
    args = p.parse_args()

    cfg  = CobConfig(loc=args.loc, seed=args.seed, out=args.out)
    code = build_cobol(cfg)

    if cfg.out:
        cfg.out.parent.mkdir(parents=True, exist_ok=True)
        cfg.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved synthetic COBOL program to {cfg.out}")
    else:
        sys.stdout.write(code)

if __name__ == "__main__":
    _cli()
