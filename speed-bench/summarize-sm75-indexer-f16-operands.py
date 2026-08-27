#!/usr/bin/env python3
"""Summarize paired SM75 WMMA128 inline/F16-materialized timing rounds."""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path


def die(message: str) -> None:
    raise SystemExit(f"error: {message}")


def main() -> int:
    if len(sys.argv) != 2:
        die("usage: summarize-sm75-indexer-f16-operands.py OUTPUT_DIR")
    root = Path(sys.argv[1]).resolve()
    timing_path = root / "timing.csv"
    if not timing_path.is_file() or timing_path.stat().st_size == 0:
        die(f"missing timing input: {timing_path}")
    with timing_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 3:
        die("at least three paired timing rounds are required")

    numeric_fields = [
        "inline_score_ms",
        "materialized_score_ms",
        "q_materialize_ms",
        "materialized_e2e_ms",
        "persistent_k_once_ms",
        "kernel_speedup",
        "e2e_speedup",
    ]
    values: dict[str, list[float]] = {
        field: [float(row[field]) for row in rows] for field in numeric_fields
    }
    orders = {row["order"] for row in rows}
    if not orders.issubset({"inline-first", "materialized-first"}) or len(orders) != 2:
        die("timing rounds must contain both alternating execution orders")
    if any(float(row["inline_score_ms"]) <= 0.0 or
           float(row["materialized_score_ms"]) <= 0.0 or
           float(row["materialized_e2e_ms"]) <= 0.0
           for row in rows):
        die("timing input contains a non-positive duration")

    medians = {field: statistics.median(samples)
               for field, samples in values.items()}
    # DeepSeek-V4 Flash uses ratio-4 indexers in even layers 2..42.  The
    # production 22/21 split therefore places layers 2..20 (10) in stage 0
    # and layers 22..42 (11) in stage 1.
    result = {
        "rounds": len(rows),
        "medians": medians,
        "persistent_f16_k_256k_bytes": 21 * 65536 * 128 * 2,
        "persistent_f16_k_stage0_bytes": 10 * 65536 * 128 * 2,
        "persistent_f16_k_stage1_bytes": 11 * 65536 * 128 * 2,
        "f16_q_live_512_bytes": 512 * 64 * 128 * 2,
        "arithmetic": "bit-exact-required",
    }
    (root / "comparison.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    with (root / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# SM75 indexer FP16-operand experiment\n\n")
        handle.write(
            "The candidate retains the shipping WMMA128 accumulation and "
            "epilogue. It moves the existing `__float2half` Q/K rounding out "
            "of every score tile. Candidate end-to-end time includes one Q "
            "materialization per 512-token microbatch; persistent-K "
            "materialization is reported separately.\n\n"
        )
        handle.write("| Measurement | Paired median |\n|---|---:|\n")
        handle.write(
            f"| Shipping inline WMMA128 | {medians['inline_score_ms']:.6f} ms |\n"
        )
        handle.write(
            f"| Materialized-input kernel | "
            f"{medians['materialized_score_ms']:.6f} ms |\n"
        )
        handle.write(
            f"| Q materialization | {medians['q_materialize_ms']:.6f} ms |\n"
        )
        handle.write(
            f"| Candidate Q-pack + score | "
            f"{medians['materialized_e2e_ms']:.6f} ms |\n"
        )
        handle.write(
            f"| Kernel-only speedup | {medians['kernel_speedup']:.6f}x |\n"
        )
        handle.write(
            f"| End-to-end speedup | {medians['e2e_speedup']:.6f}x |\n"
        )
        handle.write(
            f"| One full 8192-row K conversion | "
            f"{medians['persistent_k_once_ms']:.6f} ms |\n\n"
        )
        handle.write(
            "A production 256K sidecar would consume **336 MiB** total: "
            "160 MiB for the ten stage-0 indexer layers and 176 MiB for the "
            "eleven stage-1 layers. A live 512-token FP16 Q microbatch is "
            "8 MiB per executing tier.\n\n"
        )
        handle.write(
            "This bounded mechanism result is not a production promotion "
            "gate. Advancement still requires full-model exact frontier "
            "logits and an interleaved production A/B.\n"
        )
    print((root / "summary.md").read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
