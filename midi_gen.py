#!/usr/bin/env python3
# synthetic_midi_text.py · v0.1.0
"""
Generate a textual dump of raw‐timing MIDI data as hex strings.

Major features
--------------
* Deterministic output with --seed
* Configurable number of note events and PPQN
* Manual construction of MIDI header and track chunks
* Variable‐length quantity (VLQ) delta‐time encoding
* Outputs a plain text file containing hex bytes

Usage
-----
python synthetic_midi_text.py 50           # 50 note events
python synthetic_midi_text.py 100 --seed 42 --ppqn 480 --out raw.txt
"""
from __future__ import annotations

import argparse
import random
import struct
from pathlib import Path
from typing import List, Tuple

def encode_varlen(value: int) -> bytes:
    """Encode an integer as a MIDI variable-length quantity."""
    buffer = value & 0x7F
    parts = bytearray()
    while value >> 7:
        value >>= 7
        parts.insert(0, (buffer | 0x80) & 0xFF)
        buffer = value & 0x7F
    parts.insert(0, buffer)
    return bytes(parts)

def make_events(rng: random.Random, count: int, ppqn: int) -> List[Tuple[int, bytes]]:
    """
    Generate (abs_time, raw_event_bytes) for note on/off pairs and end-of-track.
    """
    scale = [60, 62, 64, 65, 67, 69, 71, 72]
    max_time = count * (ppqn // 2)
    events: List[Tuple[int, bytes]] = []
    for _ in range(count):
        start = rng.randint(0, max_time)
        dur = rng.randint(ppqn // 8, ppqn)
        note = rng.choice(scale)
        vel = rng.randint(50, 100)
        events.append((start, bytes([0x90, note, vel])))   # note on, ch0
        events.append((start + dur, bytes([0x80, note, 0]))) # note off, ch0
    # end-of-track
    events.append((max_time + ppqn, b'\xFF\x2F\x00'))
    return sorted(events, key=lambda x: x[0])

def build_midi_raw(count: int, ppqn: int, rng: random.Random) -> bytes:
    # Header chunk: format 1, 1 track
    header = b'MThd' + struct.pack('>IHHH', 6, 1, 1, ppqn)
    # Build track data
    evs = make_events(rng, count, ppqn)
    track_data = bytearray()
    last = 0
    for abs_time, ev in evs:
        delta = abs_time - last
        track_data.extend(encode_varlen(delta))
        track_data.extend(ev)
        last = abs_time
    # Wrap in track chunk
    trk = b'MTrk' + struct.pack('>I', len(track_data)) + track_data
    return header + trk

def midi_to_hex_lines(midi_bytes: bytes, per_line: int = 16) -> List[str]:
    """Convert MIDI bytes into hex strings, chunked per_line bytes per line."""
    lines: List[str] = []
    for i in range(0, len(midi_bytes), per_line):
        chunk = midi_bytes[i:i+per_line]
        line = " ".join(f"{b:02X}" for b in chunk)
        lines.append(line)
    return lines

def main():
    p = argparse.ArgumentParser(description="Generate a textual MIDI hex dump.")
    p.add_argument("events", type=int, help="Number of note events (pairs)")
    p.add_argument("--ppqn", type=int, default=480, help="Pulses per quarter note")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out", type=Path, default=Path("synthetic_raw.txt"),
                   help="Output text filepath")
    args = p.parse_args()

    rng = random.Random(args.seed)
    midi_bytes = build_midi_raw(args.events, args.ppqn, rng)
    hex_lines = midi_to_hex_lines(midi_bytes)

    # Write plain text file
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(hex_lines))
        f.write("\n")

    print(f"✔ Wrote MIDI hex dump ({len(hex_lines)} lines) to {args.out}")

if __name__ == "__main__":
    main()
