#!/usr/bin/env python3
"""Regression tests for the strict native-Q8 versus 86/86 FP16 scorer."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARIZER = ROOT / "speed-bench" / "summarize-q8-fp16-full-quality.py"


def write_table(
    path: Path, fields: list[str], rows: list[dict[str, object]], delimiter: str = ","
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def score_rows(delta: float = 0.0) -> list[dict[str, object]]:
    rows = []
    for index in range(100):
        rows.append({
            "id": f"case_{index:03d}",
            "target_tokens": 4,
            "nll": 4.0 + 4.0 * delta,
            "avg_nll": 1.0 + delta,
            "first_match": 1,
            "greedy_lcp": 2,
            "api_target_tokens": 4,
            "api_target_mae": 0.1 + delta,
            "api_top1_count": 4,
            "api_top1_match": 3,
            "api_topn_ref": 4,
            "api_topn_hit": 3,
            "api_pair_total": 6,
            "api_pair_agree": 5,
        })
    return rows


def binding_row(
    layer: int, consumer: int, resident: int, partner: int
) -> dict[str, object]:
    return {
        "consumer_device": consumer,
        "resident_device": resident,
        "partner_offload": partner,
        "in_dim": 8192,
        "out_dim": 4096,
        "partner_arithmetic": "f16",
        "weight_offset": 1_000_000 + layer,
        "label": f"tensor:blk.{layer}.attn_output_b.weight",
    }


def non_t256_binding(kind: str, index: int) -> dict[str, object]:
    labels = {
        "attn_kv": "attn_kv",
        "attn_output_a": "attn_output_a",
        "attn_q_a": "attn_q_a",
        "attn_q_b": "attn_q_b",
        "shared_down": "ffn_down_shexp",
        "shared_gate": "ffn_gate_shexp",
        "shared_up": "ffn_up_shexp",
    }
    return {
        "consumer_device": 0,
        "resident_device": 0,
        "partner_offload": 0,
        "in_dim": 1,
        "out_dim": 1,
        "partner_arithmetic": "f16",
        "weight_offset": 2_000_000 + index,
        "label": f"tensor:blk.0.{labels[kind]}.weight.{index}",
    }


def make_fixture(root: Path, quality_delta: float = 0.0) -> None:
    quality = root / "quality"
    quality.mkdir(parents=True)
    (root / "manifest.txt").write_text(
        "gpu_devices=0,3,1,2\n"
        "gpu_vram=auto\n"
        "stage_split=22/21\n"
        "quality_ctx=32769\n"
        "t256_layers=0-42\n"
        "comparison=native-q8-vs-fp16-t256-86-of-86\n"
        "fp16_expected_placement=43-fixed-plus-43-partner\n"
        "fp16_expected_unique_t256_allocations=43\n"
        "fp16_expected_non_t256_bindings=263\n"
        "model_hashing=disabled\n",
        encoding="utf-8",
    )
    (root / "planner-unit.log").write_text(
        "test_engine_mgpu_placement: 1/1 checks passed (0 failed)\n",
        encoding="utf-8",
    )
    (root / "gpu-exactness.log").write_text(
        "q8 partner projection exactness OK (3 classes)\n", encoding="utf-8"
    )
    (quality / "native-q8.log").write_text(
        "score_official: runtime_path=production\n"
        "ds4: CUDA EP forced pipeline split 22/21\n"
        "ds4: CUDA q8 fp16 benefit plan t256-placement=overflow\n",
        encoding="utf-8",
    )
    (quality / "fp16-t256-full.log").write_text(
        "score_official: runtime_path=production\n"
        "ds4: CUDA EP forced pipeline split 22/21\n"
        "ds4: CUDA q8 fp16 benefit plan policy partner-classes=t256 "
        "partner-layers=0-42 home-order=frozen\n"
        "ds4: CUDA q8 fp16 benefit plan materialized 349/473 candidates; "
        "T256-output_b=86/86 partner=43 partner-arithmetic=f16 "
        "t256-placement=all-partner\n"
        "ds4: CUDA q8 partner execution enabled: arithmetic=f16\n",
        encoding="utf-8",
    )

    score_fields = list(score_rows()[0])
    write_table(quality / "native-q8.tsv", score_fields, score_rows(), "\t")
    write_table(
        quality / "fp16-t256-full.tsv", score_fields, score_rows(quality_delta), "\t"
    )

    binding_fields = [
        "consumer_device", "resident_device", "partner_offload", "in_dim",
        "out_dim", "partner_arithmetic", "weight_offset", "label",
    ]
    write_table(quality / "native-q8.bindings.csv", binding_fields, [])
    bindings = []
    for layer in range(43):
        if layer <= 21:
            bindings.append(binding_row(layer, 1, 1, 0))
            bindings.append(binding_row(layer, 0, 1, 1))
        else:
            bindings.append(binding_row(layer, 2, 2, 0))
            bindings.append(binding_row(layer, 3, 2, 1))
    next_index = 0
    for kind, count in {
        "attn_kv": 24,
        "attn_output_a": 90,
        "attn_q_a": 21,
        "attn_q_b": 43,
        "shared_down": 43,
        "shared_gate": 21,
        "shared_up": 21,
    }.items():
        for _ in range(count):
            bindings.append(non_t256_binding(kind, next_index))
            next_index += 1
    write_table(quality / "fp16-t256-full.bindings.csv", binding_fields, bindings)

    audit_fields = [
        "module", "label", "layer", "physical_device", "in_dim", "out_dim",
        "result", "reason",
    ]
    native_audit = []
    full_audit = []
    for _case in range(100):
        for layer in range(43):
            home_device = 0 if layer <= 21 else 3
            common = {
                "module": "attention_output",
                "label": "attn_output_b",
                "layer": layer,
                "in_dim": 8192,
                "out_dim": 4096,
            }
            native_audit.append({
                **common,
                "physical_device": home_device,
                "result": "native_q8",
                "reason": "disabled_by_env",
            })
            full_audit.append({
                **common,
                "physical_device": 1 if layer <= 21 else 2,
                "result": "f16_partner_hit",
                "reason": "nvlink_offload",
            })
    write_table(quality / "native-q8.q8-audit.csv", audit_fields, native_audit)
    write_table(
        quality / "fp16-t256-full.q8-audit.csv", audit_fields, full_audit
    )


class FullFp16QualitySummaryTests(unittest.TestCase):
    def run_summary(self, root: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SUMMARIZER), *args, str(root)]
            if not args
            else [sys.executable, str(SUMMARIZER), *args],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_exact_endpoint_comparison_passes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-fp16-full-pass-") as raw:
            root = Path(raw)
            make_fixture(root)
            result = self.run_summary(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads((root / "quality-comparison.json").read_text())
            self.assertTrue(payload["experiment_integrity"])
            self.assertEqual(payload["coverage"]["native_q8"]["t256_bindings"], 0)
            full = payload["coverage"]["fp16_t256_full"]
            self.assertEqual(full["t256_bindings"], 86)
            self.assertEqual(full["fixed_bindings"], 43)
            self.assertEqual(full["partner_bindings"], 43)
            self.assertEqual(full["unique_t256_allocations"], 43)
            self.assertEqual(full["non_t256_bindings"], 263)
            self.assertTrue(payload["quality"]["predeclared_noninferiority_pass"])
            self.assertIn("86 / 86", (root / "summary.md").read_text())

    def test_partial_85_of_86_candidate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-fp16-full-partial-") as raw:
            root = Path(raw)
            make_fixture(root)
            path = root / "quality/fp16-t256-full.bindings.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            victim = next(
                i for i, row in enumerate(rows)
                if "attn_output_b" in row["label"]
                and row["in_dim"] == "8192"
                and row["out_dim"] == "4096"
            )
            del rows[victim]
            write_table(path, list(rows[0]), rows)
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("85/86", result.stderr)

    def test_native_fp16_contamination_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-native-contaminated-") as raw:
            root = Path(raw)
            make_fixture(root)
            path = root / "quality/native-q8.q8-audit.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["result"] = "f16_hit"
            rows[0]["reason"] = "resident"
            write_table(path, list(rows[0]), rows)
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("contaminated", result.stderr)

    def test_missing_non_t256_cache_entry_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-fp16-full-cache-loss-") as raw:
            root = Path(raw)
            make_fixture(root)
            path = root / "quality/fp16-t256-full.bindings.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            victim = next(i for i, row in enumerate(rows) if "attn_kv" in row["label"])
            del rows[victim]
            write_table(path, list(rows[0]), rows)
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("complete non-T256 cache inventory", result.stderr)

    def test_quality_regression_is_reported_without_invalidating_evidence(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-fp16-full-quality-fail-") as raw:
            root = Path(raw)
            make_fixture(root, quality_delta=0.01)
            result = self.run_summary(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads((root / "quality-comparison.json").read_text())
            self.assertFalse(payload["quality"]["predeclared_noninferiority_pass"])
            self.assertIn("non-inferiority gate: **FAIL**", (root / "summary.md").read_text())

    def test_coverage_only_validates_one_completed_arm(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-fp16-full-coverage-") as raw:
            root = Path(raw)
            make_fixture(root)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SUMMARIZER),
                    "--coverage-only",
                    str(root),
                    "native-q8",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["coverage"]["t256_bindings"], 0)


if __name__ == "__main__":
    unittest.main()
