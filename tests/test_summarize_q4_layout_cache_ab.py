#!/usr/bin/env python3

import csv
import subprocess
import sys
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "speed-bench" / "summarize-q4-layout-cache-ab.py"
VARIANTS = {
    "baseline-22x21": (22, "0,2,1,3", 100.0),
    "split-21x22": (21, "0,2,1,3", 110.0),
    "swap-22x21": (22, "0,3,1,2", 105.0),
}


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    manifest = root / "runs.tsv"
    fields = [
        "repeat", "slot", "variant", "split", "gpu_devices", "csv", "log",
        "cache_before", "cache_after",
    ]
    manifest_rows = []
    orders = [
        ("baseline-22x21", "split-21x22", "swap-22x21"),
        ("split-21x22", "swap-22x21", "baseline-22x21"),
        ("swap-22x21", "baseline-22x21", "split-21x22"),
    ]
    for repeat, order in enumerate(orders, 1):
        for slot, variant in enumerate(order, 1):
            split, devices, tps = VARIANTS[variant]
            benchmark = root / f"{variant}-r{repeat}.csv"
            benchmark.write_text(
                "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens\n"
                f"2048,2048,{tps + repeat},0\n"
                f"4096,2048,{tps - 10 + repeat},0\n",
                encoding="utf-8",
            )
            before = root / f"{variant}-r{repeat}.before.csv"
            after = root / f"{variant}-r{repeat}.after.csv"
            home = [int(value) for value in devices.split(",")[:2]]
            cache_text = (
                "physical_device,weight_offset,weight_bytes,in_dim,out_dim,fp16_bytes\n"
                f"{home[0]},1,1,1,1,{split * 10}\n"
                f"{home[1]},2,1,1,1,{(43 - split) * 10}\n"
            )
            before.write_text(cache_text, encoding="utf-8")
            after.write_text(cache_text, encoding="utf-8")
            manifest_rows.append({
                "repeat": repeat,
                "slot": slot,
                "variant": variant,
                "split": split,
                "gpu_devices": devices,
                "csv": benchmark,
                "log": root / f"{variant}-r{repeat}.log",
                "cache_before": before,
                "cache_after": after,
            })
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(manifest_rows)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(manifest), str(root), "3"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "measurement_grade=position-balanced" in result.stdout
    assert "variant=split-21x22 median_tps=112.000000" in result.stdout
    with (root / "frontier-summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    split_2k = next(
        row for row in rows
        if row["variant"] == "split-21x22" and row["ctx_tokens"] == "2048"
    )
    assert split_2k["median_paired_speedup_vs_baseline"] == "1.098039216"
    with (root / "cache-summary.csv").open(encoding="utf-8", newline="") as handle:
        cache = list(csv.DictReader(handle))
    swapped_stage1 = next(
        row for row in cache
        if row["variant"] == "swap-22x21" and row["stage"] == "1"
    )
    assert swapped_stage1["home_physical_device"] == "3"

print("q4 layout/cache A/B summarizer: OK")
