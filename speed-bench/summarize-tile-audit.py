#!/usr/bin/env python3
"""Aggregate deferred routed-MoE tile records by layer and owner."""

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


if len(sys.argv) != 3:
    raise SystemExit("usage: summarize-tile-audit.py TILE.csv SUMMARY.csv")

source = Path(sys.argv[1])
target = Path(sys.argv[2])
groups: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
with source.open(newline="", encoding="utf-8") as stream:
    for row in csv.DictReader(stream):
        groups[(int(row["layer"]), row["ownership"])].append(row)

fields = [
    "layer",
    "ownership",
    "samples",
    "pairs_mean",
    "tiles_mean",
    "active_experts_mean",
    "padded_slots_mean",
    "tile_fill_pct_mean",
]
rows = []
for (layer, ownership), samples in sorted(groups.items()):
    mean = lambda field: statistics.fmean(float(row[field]) for row in samples)
    rows.append(
        {
            "layer": layer,
            "ownership": ownership,
            "samples": len(samples),
            "pairs_mean": f"{mean('pair_count'):.3f}",
            "tiles_mean": f"{mean('tile_count'):.3f}",
            "active_experts_mean": f"{mean('active_experts'):.3f}",
            "padded_slots_mean": f"{mean('padded_slots'):.3f}",
            "tile_fill_pct_mean": f"{mean('tile_fill_pct'):.3f}",
        }
    )

target.parent.mkdir(parents=True, exist_ok=True)
with target.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(",".join(fields))
for row in rows:
    print(",".join(str(row[field]) for field in fields))
