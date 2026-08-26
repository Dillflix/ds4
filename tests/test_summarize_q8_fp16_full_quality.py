#!/usr/bin/env python3
"""Tests for all-native Q8 versus the complete production FP16 cache."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARIZER = ROOT / "speed-bench/summarize-q8-fp16-full-quality.py"
BINDING_FIELDS = [
    "consumer_device", "resident_device", "partner_offload", "weight_offset",
    "weight_bytes", "in_dim", "out_dim", "fp16_bytes",
    "partner_scratch_tokens", "resident_weight_bytes", "partner_arithmetic",
    "label", "allocation_id", "used_calls", "live",
]
ALLOCATION_FIELDS = [
    "allocation_id", "storage_kind", "half_rounded", "physical_device",
    "weight_offset", "weight_bytes", "in_dim", "out_dim", "resident_bytes",
    "logical_aliases", "live_aliases", "used_calls", "dead_bytes",
    "usage_tracking",
]
DYNAMIC_NON_T256 = {
    "attn_kv": 2, "attn_output_a": 3, "attn_q_b": 1, "shared_down": 2,
}


def write_table(path: Path, fields: list[str], rows: list[dict], delim: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delim)
        writer.writeheader()
        writer.writerows(rows)


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def score_rows(delta: float = 0.0) -> list[dict]:
    return [{
        "id": f"case_{i:03d}", "target_tokens": 4,
        "nll": 4.0 + 4.0 * delta, "avg_nll": 1.0 + delta,
        "first_match": 1, "greedy_lcp": 2, "api_target_tokens": 4,
        "api_target_mae": 0.1 + delta, "api_top1_count": 4,
        "api_top1_match": 3, "api_topn_ref": 4, "api_topn_hit": 3,
        "api_pair_total": 6, "api_pair_agree": 5,
    } for i in range(100)]


def binding(
    aid: int, layer: int, consumer: int, resident: int, partner: int,
    label: str = "attn_output_b", shape: tuple[int, int] = (8192, 4096),
) -> dict:
    in_dim, out_dim = shape
    resident_bytes = in_dim * out_dim * 2
    return {
        "consumer_device": consumer, "resident_device": resident,
        "partner_offload": partner, "weight_offset": 1_000_000 + aid,
        "weight_bytes": 10_000 + aid, "in_dim": in_dim, "out_dim": out_dim,
        "fp16_bytes": resident_bytes, "partner_scratch_tokens": 2048 if partner else 0,
        "resident_weight_bytes": resident_bytes, "partner_arithmetic": "f16",
        "label": f"tensor:blk.{layer}.{label}.weight", "allocation_id": aid,
        "used_calls": 100, "live": 1,
    }


def allocation(row: dict) -> dict:
    return {
        "allocation_id": row["allocation_id"], "storage_kind": "f16",
        "half_rounded": 0, "physical_device": row["resident_device"],
        "weight_offset": row["weight_offset"], "weight_bytes": row["weight_bytes"],
        "in_dim": row["in_dim"], "out_dim": row["out_dim"],
        "resident_bytes": row["resident_weight_bytes"], "logical_aliases": 1,
        "live_aliases": row["live"], "used_calls": row["used_calls"],
        "dead_bytes": 0, "usage_tracking": 1,
    }


def non_t256(kind: str, index: int, aid: int) -> dict:
    labels = {
        "attn_kv": "attn_kv", "attn_output_a": "attn_output_a",
        "attn_q_b": "attn_q_b", "shared_down": "ffn_down_shexp",
    }
    shapes = {
        "attn_kv": (4096, 512), "attn_output_a": (4096, 32768),
        "attn_q_b": (1536, 8192), "shared_down": (2048, 4096),
    }
    return binding(aid, index % 43, 0, 0, 0, f"{labels[kind]}.{index}", shapes[kind])


def make_fixture(root: Path, delta: float = 0.0) -> None:
    quality = root / "quality"
    quality.mkdir(parents=True)
    (root / "manifest.txt").write_text(
        "gpu_devices=0,3,1,2\ngpu_vram=auto\nstage_split=22/21\n"
        "quality_ctx=32769\nt256_layers=0-42\n"
        "comparison=all-native-q8-vs-complete-production-fp16-cache\n"
        "native_expected_expanded_bindings=0\n"
        "fp16_expected_t256_bindings=43/43\n"
        "fp16_expected_placement=43-partner\n"
        "fp16_expected_unique_t256_allocations=43\n"
        "fp16_non_t256_inventory=dynamic-production-policy\n"
        "expanded_weight_liveness=all-bindings-and-allocations-live\n"
        "model_hashing=disabled\n", encoding="utf-8")
    (root / "planner-unit.log").write_text(
        "test: 1/1 checks passed (0 failed)\n", encoding="utf-8")
    (root / "gpu-exactness.log").write_text(
        "q8 partner projection exactness OK (3 classes)\n", encoding="utf-8")
    (quality / "native-q8.log").write_text(
        "score_official: runtime_path=production\n"
        "CUDA EP forced pipeline split 22/21\nt256-placement=overflow\n",
        encoding="utf-8")
    (quality / "production-fp16-cache.log").write_text(
        "score_official: runtime_path=production\n"
        "CUDA EP forced pipeline split 22/21\npartner-classes=t256\n"
        "partner-layers=0-42\nhome-order=frozen\nT256-output_b=43/43\n"
        "partner=43 partner-arithmetic=f16\nt256-placement=all-partner\n"
        "CUDA q8 partner execution enabled:\n", encoding="utf-8")
    fields = list(score_rows()[0])
    write_table(quality / "native-q8.tsv", fields, score_rows(), "\t")
    write_table(quality / "production-fp16-cache.tsv", fields, score_rows(delta), "\t")
    write_table(quality / "native-q8.bindings.csv", BINDING_FIELDS, [])
    write_table(quality / "native-q8.allocations.csv", ALLOCATION_FIELDS, [])
    bindings = [
        binding(layer + 1, layer, 0 if layer <= 21 else 3,
                1 if layer <= 21 else 2, 1)
        for layer in range(43)
    ]
    aid, index = 44, 0
    for kind, count in DYNAMIC_NON_T256.items():
        for _ in range(count):
            bindings.append(non_t256(kind, index, aid))
            aid += 1
            index += 1
    write_table(quality / "production-fp16-cache.bindings.csv", BINDING_FIELDS, bindings)
    write_table(quality / "production-fp16-cache.allocations.csv",
                ALLOCATION_FIELDS, [allocation(row) for row in bindings])
    audit_fields = [
        "module", "label", "layer", "physical_device", "in_dim", "out_dim",
        "result", "reason",
    ]
    native_audit, candidate_audit = [], []
    for _case in range(100):
        for layer in range(43):
            common = {
                "module": "attention_output", "label": "attn_output_b",
                "layer": layer, "in_dim": 8192, "out_dim": 4096,
            }
            native_audit.append({
                **common, "physical_device": 0 if layer <= 21 else 3,
                "result": "native_q8", "reason": "disabled_by_env",
            })
            candidate_audit.append({
                **common, "physical_device": 1 if layer <= 21 else 2,
                "result": "f16_partner_hit", "reason": "nvlink_offload",
            })
    write_table(quality / "native-q8.q8-audit.csv", audit_fields, native_audit)
    write_table(quality / "production-fp16-cache.q8-audit.csv",
                audit_fields, candidate_audit)


class QualitySummaryTests(unittest.TestCase):
    def run_summary(self, root: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, str(SUMMARIZER), str(root)],
                              text=True, capture_output=True, check=False)

    def test_dynamic_inventory_and_live_allocations_pass(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); make_fixture(root)
            result = self.run_summary(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads((root / "quality-comparison.json").read_text())
            full = payload["coverage"]["production_fp16_cache"]
            self.assertEqual(full["t256_bindings"], 43)
            self.assertEqual(full["unique_t256_allocations"], 43)
            self.assertEqual(full["non_t256_class_inventory"], DYNAMIC_NON_T256)
            self.assertEqual(full["expanded_weights"]["dead_bytes"], 0)
            self.assertEqual(len(full["non_t256_descriptor_inventory"]), 8)
            self.assertTrue(payload["quality"]["predeclared_noninferiority_pass"])

    def test_partial_t256_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); make_fixture(root)
            path = root / "quality/production-fp16-cache.bindings.csv"
            rows = read_table(path); del rows[0]
            write_table(path, BINDING_FIELDS, rows)
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("42/43", result.stderr)

    def test_duplicate_t256_allocation_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); make_fixture(root)
            path = root / "quality/production-fp16-cache.bindings.csv"
            rows = read_table(path); rows[1]["allocation_id"] = rows[0]["allocation_id"]
            write_table(path, BINDING_FIELDS, rows)
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("allocation 1 does not match", result.stderr)

    def test_native_contamination_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); make_fixture(root)
            path = root / "quality/native-q8.q8-audit.csv"
            rows = read_table(path); rows[0].update(result="f16_hit", reason="resident")
            write_table(path, list(rows[0]), rows)
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("contaminated", result.stderr)

    def test_unused_binding_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); make_fixture(root)
            path = root / "quality/production-fp16-cache.bindings.csv"
            rows = read_table(path); rows[-1].update(used_calls="0", live="0")
            write_table(path, BINDING_FIELDS, rows)
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("exported but never used", result.stderr)

    def test_dead_allocation_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); make_fixture(root)
            path = root / "quality/production-fp16-cache.allocations.csv"
            rows = read_table(path); rows[-1]["dead_bytes"] = rows[-1]["resident_bytes"]
            write_table(path, ALLOCATION_FIELDS, rows)
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("dead expanded-weight bytes", result.stderr)

    def test_non_t256_partner_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); make_fixture(root)
            bp = root / "quality/production-fp16-cache.bindings.csv"
            ap = root / "quality/production-fp16-cache.allocations.csv"
            bindings = read_table(bp); allocations = read_table(ap)
            bindings[-1].update(partner_offload="1", resident_device="1")
            allocations[-1]["physical_device"] = "1"
            write_table(bp, BINDING_FIELDS, bindings)
            write_table(ap, ALLOCATION_FIELDS, allocations)
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("non-T256 partner binding", result.stderr)

    def test_quality_gate_can_fail_with_valid_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); make_fixture(root, 0.01)
            result = self.run_summary(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads((root / "quality-comparison.json").read_text())
            self.assertFalse(payload["quality"]["predeclared_noninferiority_pass"])

    def test_coverage_only_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); make_fixture(root)
            result = subprocess.run([
                sys.executable, str(SUMMARIZER), "--coverage-only", str(root),
                "production-fp16-cache",
            ], text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["coverage"]["t256_bindings"], 43)


if __name__ == "__main__":
    unittest.main()
