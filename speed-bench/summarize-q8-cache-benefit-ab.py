#!/usr/bin/env python3
import csv
import statistics
import sys
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


if len(sys.argv) != 3:
    fail("usage: summarize-q8-cache-benefit-ab.py RUNS.tsv OUTPUT_DIR")

runs_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
samples = {}
with runs_path.open(newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        repeat = int(row["repeat"])
        variant = row["variant"]
        with Path(row["csv"]).open(newline="") as cf:
            values = {int(r["ctx_tokens"]): float(r["prefill_tps"])
                      for r in csv.DictReader(cf)}
        samples[(repeat, variant)] = values

repeats = sorted({r for r, _ in samples})
paired = []
for repeat in repeats:
    planner = samples.get((repeat, "benefit-plan"))
    legacy = samples.get((repeat, "first-use"))
    if not planner or not legacy or set(planner) != set(legacy):
        fail(f"repeat {repeat} lacks matched planner/first-use frontiers")
    for ctx in sorted(planner):
        paired.append({
            "repeat": repeat,
            "ctx_tokens": ctx,
            "planner_tps": planner[ctx],
            "first_use_tps": legacy[ctx],
            "planner_over_first_use": planner[ctx] / legacy[ctx],
        })

with (out_dir / "paired-samples.csv").open("w", newline="") as f:
    fields = list(paired[0]) if paired else []
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(paired)

lines = []
for ctx in sorted({r["ctx_tokens"] for r in paired}):
    ratios = [r["planner_over_first_use"] for r in paired if r["ctx_tokens"] == ctx]
    lines.append(f"ctx={ctx} median_planner/first_use={statistics.median(ratios):.5f} "
                 f"min={min(ratios):.5f} max={max(ratios):.5f} n={len(ratios)}")
(out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
