#!/usr/bin/env python3
"""Regression for exact gate/down plans derived from deferred expert counts."""

import csv
import subprocess
import sys
import tempfile
from pathlib import Path


repo = Path(__file__).resolve().parents[1]
script = repo / "speed-bench" / "summarize-q4-real-histogram.py"
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    source = root / "tile.csv"
    plan = root / "plan.csv"
    experts = root / "experts.csv"
    frequency = root / "frequency.csv"
    fields = [
        "logical_tier", "physical_device", "sequence", "layer",
        "token_offset", "n_tokens", "ownership", "owner_base",
        "owner_count", "selected_slots", "pair_count", "tile_count",
        "slot_count", "active_experts", "padded_slots", "tile_fill_pct",
        "n_total_expert", "captured_experts",
    ] + [f"expert_{index:03d}_pairs" for index in range(4)]
    row = {
        "logical_tier": 0, "physical_device": 0, "sequence": 0,
        "layer": 3, "token_offset": 0, "n_tokens": 512,
        "ownership": "partner", "owner_base": 128, "owner_count": 4,
        "selected_slots": 32, "pair_count": 32, "tile_count": 4,
        "slot_count": 64, "active_experts": 4, "padded_slots": 32,
        "tile_fill_pct": 50.0, "n_total_expert": 4,
        "captured_experts": 4, "expert_000_pairs": 1,
        "expert_001_pairs": 6, "expert_002_pairs": 9,
        "expert_003_pairs": 16,
    }
    with source.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    subprocess.run(
        [sys.executable, str(script), str(source), str(plan), str(experts),
         str(frequency)],
        check=True,
        capture_output=True,
        text=True,
    )
    with plan.open(newline="", encoding="utf-8") as stream:
        result = next(csv.DictReader(stream))
    expected = {
        "gate_tile8_count": "6",
        "gate_tile8_slots": "48",
        "gate_tile8_padded_slots": "16",
        "gate_adaptive_full8_count": "3",
        "gate_adaptive_tail8_count": "1",
        "gate_adaptive_tail4_count": "2",
        "gate_adaptive_slots": "40",
        "gate_adaptive_padded_slots": "8",
        "down_tile16_count": "4",
        "down_tile16_slots": "64",
        "down_tile16_padded_slots": "32",
        "down_adaptive_full16_count": "1",
        "down_adaptive_tail8_count": "2",
        "down_adaptive_tail4_count": "2",
        "down_adaptive_slots": "40",
        "down_adaptive_padded_slots": "8",
        "gate_grid_tile_capacity": "8",
        "gate_grid_inactive_tiles": "2",
        "down_grid_tile_capacity": "6",
        "down_grid_inactive_tiles": "2",
    }
    for field, value in expected.items():
        if result[field] != value:
            raise SystemExit(f"{field}: expected {value}, got {result[field]}")
    with experts.open(newline="", encoding="utf-8") as stream:
        expert_rows = list(csv.DictReader(stream))
    if [row["global_expert"] for row in expert_rows] != ["128", "129", "130", "131"]:
        raise SystemExit("owner-relative expert numbering was not preserved")

print("q4 real histogram summary regression: OK")
