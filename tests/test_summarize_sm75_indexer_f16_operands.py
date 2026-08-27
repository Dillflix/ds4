#!/usr/bin/env python3
"""Synthetic regression for the SM75 FP16-indexer operand summary."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARIZER = ROOT / "speed-bench" / "summarize-sm75-indexer-f16-operands.py"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="ds4-indexer-f16-summary-") as temp:
        root = Path(temp)
        fields = [
            "round", "order", "inline_score_ms", "materialized_score_ms",
            "q_materialize_ms", "materialized_e2e_ms",
            "persistent_k_once_ms", "kernel_speedup", "e2e_speedup",
        ]
        rows = [
            [1, "inline-first", 10.0, 7.0, 0.5, 7.5, 0.2, 10 / 7, 10 / 7.5],
            [2, "materialized-first", 10.2, 7.2, 0.4, 7.6, 0.2, 10.2 / 7.2, 10.2 / 7.6],
            [3, "inline-first", 9.8, 6.8, 0.6, 7.4, 0.3, 9.8 / 6.8, 9.8 / 7.4],
        ]
        with (root / "timing.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            writer.writerows(rows)
        subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
        result = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
        assert result["rounds"] == 3
        assert result["persistent_f16_k_256k_bytes"] == 336 * 1024 * 1024
        assert result["persistent_f16_k_stage0_bytes"] == 160 * 1024 * 1024
        assert result["persistent_f16_k_stage1_bytes"] == 176 * 1024 * 1024
        assert result["f16_q_live_512_bytes"] == 8 * 1024 * 1024
        assert abs(result["medians"]["inline_score_ms"] - 10.0) < 1e-9
        assert "not a production promotion gate" in (
            root / "summary.md"
        ).read_text(encoding="utf-8")
        (root / "manifest.txt").write_text(
            "candidate=streaming64\n", encoding="utf-8"
        )
        timing = (root / "timing.csv").read_text(encoding="utf-8")
        timing = timing.replace("materialized_score_ms", "candidate_score_ms")
        timing = timing.replace("materialized_e2e_ms", "candidate_e2e_ms")
        (root / "timing.csv").write_text(timing, encoding="utf-8")
        (root / "validation").mkdir()
        with (root / "validation" / "resources.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=(
                "kernel", "REG", "STACK", "SHARED", "LOCAL",
                "four_cta_resource_gate", "reason",
            ))
            writer.writeheader()
            writer.writerow({
                "kernel": "streaming64", "REG": 96, "STACK": 0,
                "SHARED": 8192, "LOCAL": 0,
                "four_cta_resource_gate": "pass", "reason": "none",
            })
        subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
        result = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
        assert result["candidate"] == "streaming64"
        assert abs(result["medians"]["candidate_score_ms"] - 7.0) < 1e-9
        assert result["resource"]["four_cta_resource_gate"] == "pass"
        assert "Streaming FP16 WMMA64" in (
            root / "summary.md"
        ).read_text(encoding="utf-8")
    print("SM75 indexer FP16-operand summarizer test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
