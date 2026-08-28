#!/usr/bin/env python3
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} RUNS.csv OUTPUT.csv")
    runs_path, output_path = map(Path, sys.argv[1:])
    values = defaultdict(list)
    paired = {}
    with runs_path.open(newline="") as handle:
        for run in csv.DictReader(handle):
            with Path(run["csv"]).open(newline="") as bench_handle:
                for row in csv.DictReader(bench_handle):
                    ctx = int(row["ctx_tokens"])
                    tps = float(row["prefill_tps"])
                    key = (run["variant"], ctx)
                    values[key].append(tps)
                    paired[(int(run["repeat"]), run["variant"], ctx)] = tps
    variants = ["control", "indexed8", "down-compact", "both"]
    contexts = sorted({ctx for _, ctx in values})
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "ctx_tokens", "variant", "median_tps", "paired_median_speedup",
            "change_pct", "samples",
        ])
        for ctx in contexts:
            for variant in variants:
                samples = values.get((variant, ctx), [])
                if not samples:
                    continue
                ratios = []
                for repeat in sorted({r for r, v, c in paired if c == ctx}):
                    control = paired.get((repeat, "control", ctx))
                    candidate = paired.get((repeat, variant, ctx))
                    if control and candidate:
                        ratios.append(candidate / control)
                speedup = statistics.median(ratios) if ratios else float("nan")
                writer.writerow([
                    ctx, variant, f"{statistics.median(samples):.3f}",
                    f"{speedup:.6f}", f"{(speedup - 1.0) * 100.0:.3f}",
                    len(samples),
                ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
