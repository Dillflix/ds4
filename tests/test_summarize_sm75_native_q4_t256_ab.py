#!/usr/bin/env python3

import csv
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARIZER = ROOT / "speed-bench/summarize-sm75-native-q4-t256-ab.py"
VARIANTS = ("standard-all-partner", "native-all-partner")
CONTEXTS = (16384, 32768)
RUN_FIELDS = [
    "repeat", "slot", "variant", "model", "policy", "csv", "log",
    "audit", "bindings", "allocations", "cache_before", "cache_after",
    "logits",
]


def write_table(
    path: Path,
    fields: list[str],
    rows: list[dict[str, object]],
    delimiter: str = ",",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def read_runs(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def t256_binding(layer: int, consumer: int, resident: int) -> dict[str, object]:
    return {
        "consumer_device": consumer,
        "resident_device": resident,
        "partner_offload": int(consumer != resident),
        "weight_offset": 1_000_000 + layer,
        "weight_bytes": 35_651_584,
        "in_dim": 8192,
        "out_dim": 4096,
        "partner_arithmetic": "f16",
        "label": f"tensor:blk.{layer}.attn_output_b.weight",
        "allocation_id": layer + 1,
        "used_calls": 1,
        "live": 1,
    }


def make_fixture(
    root: Path,
    standard_tps: float = 100.0,
    native_tps: float = 120.0,
    repeats: int = 1,
) -> Path:
    audit_fields = [
        "label", "layer", "physical_device", "in_dim", "out_dim",
        "result", "reason",
    ]
    binding_fields = [
        "consumer_device", "resident_device", "partner_offload",
        "weight_offset", "weight_bytes", "in_dim", "out_dim",
        "partner_arithmetic", "label", "allocation_id", "used_calls", "live",
    ]
    allocation_fields = [
        "allocation_id", "storage_kind", "physical_device", "weight_offset",
        "weight_bytes", "in_dim", "out_dim", "resident_bytes",
        "logical_aliases", "live_aliases", "used_calls", "dead_bytes",
        "usage_tracking",
    ]
    records: list[dict[str, object]] = []

    for repeat in range(1, repeats + 1):
        for slot, variant in enumerate(VARIANTS, 1):
            family = variant.split("-", 1)[0]
            stem = root / "runs" / f"{variant}-r{repeat}"
            csv_path = stem.with_suffix(".csv")
            log_path = stem.with_suffix(".log")
            audit_path = stem.with_suffix(".audit.csv")
            bindings_path = stem.with_suffix(".bindings.csv")
            allocations_path = stem.with_suffix(".allocations.csv")
            cache_before = stem.with_suffix(".cache-before.csv")
            cache_after = stem.with_suffix(".cache-after.csv")
            logits = root / "logits" / f"{variant}-r{repeat}"
            logits.mkdir(parents=True, exist_ok=True)

            base_tps = standard_tps if family == "standard" else native_tps
            repeat_scale = 1.0 + 0.002 * (repeat - 1)
            write_table(
                csv_path,
                ["ctx_tokens", "prefill_tokens", "prefill_tps"],
                [
                    {
                        "ctx_tokens": context,
                        "prefill_tokens": 16384,
                        "prefill_tps": base_tps * repeat_scale,
                    }
                    for context in CONTEXTS
                ],
            )

            log_lines = [
                "ds4: CUDA q8 fp16 benefit plan partner-classes=t256 "
                "partner-layers=0-42 t256-placement=all-partner",
                "ds4: CUDA q8 fp16 benefit plan materialized 306/430 candidates; "
                "T256-output_b=43/43 partner=43 partner-arithmetic=f16",
                "ds4: CUDA q8 fp16 partner summary: calls=43",
            ]
            if family == "native":
                log_lines.append(
                    "ds4: SM75 native routed-Q4 layout enabled "
                    "(packed A/W, planner=cost, gate=tile8, down=full-stage)"
                )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

            audit_rows = []
            binding_rows = []
            allocation_rows = []
            for layer in range(43):
                home, peer = (0, 1) if layer <= 21 else (3, 2)
                binding_row = t256_binding(layer, home, peer)
                binding_rows.append(binding_row)
                allocation_rows.append({
                    "allocation_id": binding_row["allocation_id"],
                    "storage_kind": "f16",
                    "physical_device": peer,
                    "weight_offset": binding_row["weight_offset"],
                    "weight_bytes": binding_row["weight_bytes"],
                    "in_dim": 8192,
                    "out_dim": 4096,
                    "resident_bytes": 67_108_864,
                    "logical_aliases": 1,
                    "live_aliases": 1,
                    "used_calls": 1,
                    "dead_bytes": 0,
                    "usage_tracking": 1,
                })
                audit_rows.append({
                    "label": f"blk.{layer}.attn_output_b.weight",
                    "layer": layer,
                    "physical_device": peer,
                    "in_dim": 8192,
                    "out_dim": 4096,
                    "result": "f16_partner_hit",
                    "reason": "nvlink_offload",
                })
            for index in range(7):
                device = (0, 3, 1, 2)[index % 4]
                binding_row = {
                    "consumer_device": device,
                    "resident_device": device,
                    "partner_offload": 0,
                    "weight_offset": 2_000_000 + index,
                    "weight_bytes": 4096 + index,
                    "in_dim": 64,
                    "out_dim": 64,
                    "partner_arithmetic": "f16",
                    "label": f"tensor:non_t256.{index}",
                    "allocation_id": 1000 + index,
                    "used_calls": 1,
                    "live": 1,
                }
                binding_rows.append(binding_row)
                allocation_rows.append({
                    "allocation_id": binding_row["allocation_id"],
                    "storage_kind": "f16",
                    "physical_device": device,
                    "weight_offset": binding_row["weight_offset"],
                    "weight_bytes": binding_row["weight_bytes"],
                    "in_dim": 64,
                    "out_dim": 64,
                    "resident_bytes": 8192,
                    "logical_aliases": 1,
                    "live_aliases": 1,
                    "used_calls": 1,
                    "dead_bytes": 0,
                    "usage_tracking": 1,
                })
            write_table(audit_path, audit_fields, audit_rows)
            write_table(bindings_path, binding_fields, binding_rows)
            write_table(allocations_path, allocation_fields, allocation_rows)
            cache_text = "device,offset,bytes\n0,1024,67108864\n"
            cache_before.write_text(cache_text, encoding="utf-8")
            cache_after.write_text(cache_text, encoding="utf-8")

            raw = struct.pack("<4f", 1.0, -2.0, 3.5, 0.2501)
            for context in CONTEXTS:
                (logits / f"frontier_{context:06d}.logits.f32").write_bytes(raw)

            records.append({
                "repeat": repeat,
                "slot": slot,
                "variant": variant,
                "model": f"/models/{family}.gguf",
                "policy": "all-partner",
                "csv": csv_path,
                "log": log_path,
                "audit": audit_path,
                "bindings": bindings_path,
                "allocations": allocations_path,
                "cache_before": cache_before,
                "cache_after": cache_after,
                "logits": logits,
            })

    runs_path = root / "runs.tsv"
    write_table(runs_path, RUN_FIELDS, records, delimiter="\t")
    return runs_path


class SummarizeSm75NativeQ4T256AbTests(unittest.TestCase):
    def run_summarizer(
        self, root: Path, runs_path: Path
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SUMMARIZER), str(runs_path), str(root / "out")],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_valid_focused_integration_passes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-native-t256-pass-") as raw:
            root = Path(raw)
            result = self.run_summarizer(root, make_fixture(root))
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads((root / "out/acceptance.json").read_text())
            self.assertTrue(payload["accepted"])
            self.assertTrue(payload["gates"]["live_non_t256_inventory_exact"])
            self.assertTrue(
                payload["gates"]["zero_dead_expanded_weight_allocations"]
            )
            self.assertEqual(payload["scope"]["repeats"], 1)
            self.assertEqual(payload["scope"]["variants"], list(VARIANTS))
            self.assertEqual(payload["evidence"]["cross_layout_comparisons"], 2)
            self.assertEqual(payload["evidence"]["repeat_determinism_comparisons"], 0)
            paired = read_csv(root / "out/paired-effects.csv")
            self.assertAlmostEqual(
                float(paired[0]["native_all_partner_over_standard_all_partner"]),
                1.2,
            )
            self.assertIn("Overall: **PASS**", result.stdout)

    def test_wrong_order_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-native-t256-order-") as raw:
            root = Path(raw)
            runs_path = make_fixture(root)
            rows = read_runs(runs_path)
            rows[0]["slot"], rows[1]["slot"] = rows[1]["slot"], rows[0]["slot"]
            write_table(runs_path, RUN_FIELDS, rows, delimiter="\t")
            result = self.run_summarizer(root, runs_path)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("order is", result.stderr)

    def test_nonexact_native_logits_fail_acceptance(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-native-t256-logit-") as raw:
            root = Path(raw)
            runs_path = make_fixture(root)
            target = root / "logits/native-all-partner-r1/frontier_016384.logits.f32"
            target.write_bytes(struct.pack("<4f", 1.0, -2.0, 3.5, 9.0))
            result = self.run_summarizer(root, runs_path)
            self.assertEqual(result.returncode, 1, result.stderr)
            payload = json.loads((root / "out/acceptance.json").read_text())
            self.assertFalse(payload["gates"]["cross_layout_raw_logits_exact"])

    def test_mismatched_frontier_work_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-native-t256-work-") as raw:
            root = Path(raw)
            runs_path = make_fixture(root)
            path = root / "runs/native-all-partner-r1.csv"
            rows = read_csv(path)
            rows[1]["prefill_tokens"] = "8192"
            write_table(path, ["ctx_tokens", "prefill_tokens", "prefill_tps"], rows)
            result = self.run_summarizer(root, runs_path)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("mismatched prefill work", result.stderr)

    def test_incomplete_all_partner_mapping_fails_acceptance(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-native-t256-map-") as raw:
            root = Path(raw)
            runs_path = make_fixture(root)
            path = root / "runs/native-all-partner-r1.bindings.csv"
            rows = read_csv(path)
            rows[1]["resident_device"] = rows[1]["consumer_device"]
            rows[1]["partner_offload"] = "0"
            write_table(path, list(rows[0]), rows)
            result = self.run_summarizer(root, runs_path)
            self.assertEqual(result.returncode, 1, result.stderr)
            payload = json.loads((root / "out/acceptance.json").read_text())
            self.assertFalse(payload["gates"]["policy_and_dispatch_evidence"])

    def test_dead_peer_consumer_binding_fails_acceptance(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-native-t256-dead-peer-") as raw:
            root = Path(raw)
            runs_path = make_fixture(root)
            path = root / "runs/native-all-partner-r1.bindings.csv"
            rows = read_csv(path)
            rows.append(t256_binding(0, 1, 1))
            write_table(path, list(rows[0]), rows)
            result = self.run_summarizer(root, runs_path)
            self.assertEqual(result.returncode, 1, result.stderr)
            payload = json.loads((root / "out/acceptance.json").read_text())
            self.assertFalse(payload["gates"]["policy_and_dispatch_evidence"])

    def test_changed_live_non_t256_descriptor_fails_acceptance(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-native-t256-inventory-") as raw:
            root = Path(raw)
            runs_path = make_fixture(root)
            bindings_path = root / "runs/native-all-partner-r1.bindings.csv"
            bindings = read_csv(bindings_path)
            target = next(row for row in bindings if row["label"] == "tensor:non_t256.0")
            target["weight_offset"] = str(int(target["weight_offset"]) + 64)
            write_table(bindings_path, list(bindings[0]), bindings)
            allocations_path = root / "runs/native-all-partner-r1.allocations.csv"
            allocations = read_csv(allocations_path)
            allocation = next(
                row for row in allocations
                if row["allocation_id"] == target["allocation_id"]
            )
            allocation["weight_offset"] = target["weight_offset"]
            write_table(allocations_path, list(allocations[0]), allocations)
            result = self.run_summarizer(root, runs_path)
            self.assertEqual(result.returncode, 1, result.stderr)
            payload = json.loads((root / "out/acceptance.json").read_text())
            self.assertFalse(payload["gates"]["live_non_t256_inventory_exact"])

    def test_dead_expanded_weight_fails_acceptance(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-native-t256-dead-weight-") as raw:
            root = Path(raw)
            runs_path = make_fixture(root)
            path = root / "runs/native-all-partner-r1.allocations.csv"
            rows = read_csv(path)
            target = next(row for row in rows if row["allocation_id"] == "1000")
            target["used_calls"] = "0"
            target["live_aliases"] = "0"
            target["dead_bytes"] = target["resident_bytes"]
            write_table(path, list(rows[0]), rows)
            result = self.run_summarizer(root, runs_path)
            self.assertEqual(result.returncode, 1, result.stderr)
            payload = json.loads((root / "out/acceptance.json").read_text())
            self.assertFalse(
                payload["gates"]["zero_dead_expanded_weight_allocations"]
            )

    def test_native_regression_fails_acceptance(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-native-t256-regress-") as raw:
            root = Path(raw)
            runs_path = make_fixture(root, standard_tps=100.0, native_tps=99.0)
            result = self.run_summarizer(root, runs_path)
            self.assertEqual(result.returncode, 1, result.stderr)
            payload = json.loads((root / "out/acceptance.json").read_text())
            self.assertFalse(payload["gates"]["native_all_partner_no_regression"])


if __name__ == "__main__":
    unittest.main()
