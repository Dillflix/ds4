#!/usr/bin/env python3
import csv
import json
import statistics
import sys
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


if len(sys.argv) != 3:
    fail("usage: summarize-q8-t32-fused-ab.py RUNS.tsv OUTPUT_DIR")

runs_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
variants = ("old_local", "new_local", "old_partner", "new_partner")
samples: dict[tuple[int, str], dict[int, float]] = {}
logits_dirs: dict[tuple[int, str], Path] = {}
with runs_path.open(newline="") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        key = (int(row["repeat"]), row["variant"])
        with Path(row["csv"]).open(newline="") as csv_file:
            samples[key] = {
                int(item["ctx_tokens"]): float(item["prefill_tps"])
                for item in csv.DictReader(csv_file)
            }
        logits_dirs[key] = Path(row["logits"])

repeats = sorted({repeat for repeat, _ in samples})
comparisons = (
    ("new_local/old_local", "new_local", "old_local"),
    ("old_partner/old_local", "old_partner", "old_local"),
    ("new_partner/new_local", "new_partner", "new_local"),
    ("new_partner/old_partner", "new_partner", "old_partner"),
    ("new_partner/old_local", "new_partner", "old_local"),
)
paired: list[dict[str, object]] = []
for repeat in repeats:
    for variant in variants:
        if (repeat, variant) not in samples:
            fail(f"repeat {repeat} lacks {variant}")
    reference_frontiers = set(samples[(repeat, "old_local")])
    for variant in variants[1:]:
        if set(samples[(repeat, variant)]) != reference_frontiers:
            fail(f"repeat {repeat} has mismatched {variant} frontiers")
    for name, numerator, denominator in comparisons:
        for context in sorted(reference_frontiers):
            n = samples[(repeat, numerator)][context]
            d = samples[(repeat, denominator)][context]
            paired.append({
                "repeat": repeat,
                "comparison": name,
                "ctx_tokens": context,
                "numerator_tps": n,
                "denominator_tps": d,
                "ratio": n / d,
            })

with (out_dir / "paired-samples.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
    writer.writeheader()
    writer.writerows(paired)


def load_logits(directory: Path) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for path in sorted(directory.glob("frontier_*.logits.json")):
        payload = json.loads(path.read_text())
        values = payload.get("logits")
        if not isinstance(values, list):
            fail(f"missing logits array: {path}")
        result[path.name] = [float(value) for value in values]
    if not result:
        fail(f"no frontier logits in {directory}")
    return result


logit_rows: list[dict[str, object]] = []
for repeat in repeats:
    loaded = {
        variant: load_logits(logits_dirs[(repeat, variant)])
        for variant in variants
    }
    files = set(loaded["old_local"])
    if any(set(loaded[variant]) != files for variant in variants[1:]):
        fail(f"repeat {repeat} has mismatched logit frontiers")
    for name, lhs_name, rhs_name, require_exact in (
        ("old_partner_vs_old_local", "old_partner", "old_local", True),
        ("new_partner_vs_new_local", "new_partner", "new_local", True),
        ("new_local_vs_old_local", "new_local", "old_local", False),
    ):
        for filename in sorted(files):
            lhs = loaded[lhs_name][filename]
            rhs = loaded[rhs_name][filename]
            if len(lhs) != len(rhs):
                fail(f"logit length mismatch: {name} {filename}")
            deltas = [abs(a - b) for a, b in zip(lhs, rhs)]
            exact = lhs == rhs
            logit_rows.append({
                "repeat": repeat,
                "comparison": name,
                "frontier": filename,
                "required_exact": int(require_exact),
                "exact": int(exact),
                "max_abs": max(deltas, default=0.0),
                "mean_abs": statistics.fmean(deltas) if deltas else 0.0,
                "top1_equal": int(max(range(len(lhs)), key=lhs.__getitem__) ==
                                  max(range(len(rhs)), key=rhs.__getitem__)),
            })

with (out_dir / "logit-comparison.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(logit_rows[0]))
    writer.writeheader()
    writer.writerows(logit_rows)

lines: list[str] = []
for name, _, _ in comparisons:
    lines.append(f"[{name}]")
    contexts = sorted({
        int(row["ctx_tokens"])
        for row in paired if row["comparison"] == name
    })
    for context in contexts:
        ratios = [
            float(row["ratio"])
            for row in paired
            if row["comparison"] == name and row["ctx_tokens"] == context
        ]
        lines.append(
            f"ctx={context} median={statistics.median(ratios):.5f} "
            f"min={min(ratios):.5f} max={max(ratios):.5f} n={len(ratios)}"
        )
    lines.append("")
drift = [
    row for row in logit_rows
    if row["comparison"] == "new_local_vs_old_local"
]
lines.append("[new-local numerical drift versus old-local]")
lines.append(f"max_abs={max(float(row['max_abs']) for row in drift):.9g}")
lines.append(f"max_mean_abs={max(float(row['mean_abs']) for row in drift):.9g}")
lines.append(f"top1_equal={sum(int(row['top1_equal']) for row in drift)}/{len(drift)}")
(out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
