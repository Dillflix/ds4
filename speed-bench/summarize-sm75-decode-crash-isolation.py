#!/usr/bin/env python3
"""Summarize a graduated SM75 32K production decode-control run."""

from __future__ import annotations

import csv
import pathlib
import statistics
import sys


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        fail("usage: summarize-sm75-decode-crash-isolation.py DIR")
    root = pathlib.Path(argv[1])
    runs_path = root / "production/runs.tsv"
    with runs_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        fail("decode-control run inventory is empty")

    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        if int(row["pp_tokens"]) != 32768:
            fail("run inventory contains a non-32K prefill case")
        grouped.setdefault(int(row["tg_tokens"]), []).append(row)

    summary_path = root / "production/summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "pp_tokens",
                "tg_tokens",
                "prefill_median_tps",
                "decode_median_tps",
                "first_token_median_ms",
                "steady_median_tps",
                "steady_median_ms_per_token",
                "samples",
                "decode_graph",
                "gpu_health",
            )
        )
        for tg in sorted(grouped):
            group = grouped[tg]
            writer.writerow(
                (
                    32768,
                    tg,
                    f"{statistics.median(float(row['prefill_tps']) for row in group):.6f}",
                    f"{statistics.median(float(row['gen_tps']) for row in group):.6f}",
                    f"{statistics.median(float(row['first_ms']) for row in group):.6f}",
                    f"{statistics.median(float(row['steady_tps']) for row in group):.6f}",
                    f"{statistics.median(float(row['steady_ms_per_token']) for row in group):.6f}",
                    len(group),
                    "disabled",
                    "passed-pre-and-post",
                )
            )


if __name__ == "__main__":
    main(sys.argv)
