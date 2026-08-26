#!/usr/bin/env python3
"""Tests for deliberate T256 placement evidence validation."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARIZER = ROOT / "speed-bench" / "summarize-q8-t256-placement.py"
LEGACY_VARIANTS = ("native", "all-local", "balanced", "overflow", "all-partner")
STRICT_VARIANTS = ("native", "all-local", "balanced", "all-partner")


def write(path: Path, fields: list[str], items: list[dict[str, object]], delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(items)


def binding(layer: int, consumer: int, resident: int) -> dict[str, object]:
    weight_offset = (layer + 1) * 4096
    fp16_bytes = 8192 * 4096 * 2
    return {
        "in_dim": 8192,
        "out_dim": 4096,
        "label": f"tensor:blk.{layer}.attn_output_b.weight",
        "consumer_device": consumer,
        "resident_device": resident,
        "partner_offload": int(consumer != resident),
        "partner_arithmetic": "f16",
        "weight_offset": weight_offset,
        "weight_bytes": 4096,
        "fp16_bytes": fp16_bytes,
        "resident_weight_bytes": fp16_bytes,
        "allocation_id": layer + 1,
        "used_calls": 1,
        "live": 1,
    }


PROJECTIONS = (
    ("attn_q_a", "attn_q_a", 4096, 1024),
    ("attn_q_b", "attn_q_b", 1024, 32768),
    ("attn_kv", "attn_kv", 4096, 512),
    ("attn_output_a", "attn_output_a", 4096, 8192),
    ("attn_output_b", "attn_output_b", 8192, 4096),
    ("shared_down", "ffn_down_shexp", 2048, 4096),
    ("shared_gate_up", "ffn_gate_shexp", 4096, 2048),
    ("shared_gate_up", "ffn_up_shexp", 4096, 2048),
)


def complete_bindings(
    variant: str, overflow_partner_layers: set[int]
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for layer in range(43):
        home, peer = (0, 1) if layer <= 21 else (3, 2)
        for ordinal, (_cls, suffix, in_dim, out_dim) in enumerate(PROJECTIONS):
            if variant == "native" and suffix == "attn_output_b":
                continue
            use_partner = suffix == "attn_output_b" and (
                variant == "all-partner"
                or (variant == "balanced" and layer % 2 == 1)
                or (variant == "overflow" and layer in overflow_partner_layers)
            )
            resident = peer if use_partner else home
            source_id = layer * len(PROJECTIONS) + ordinal + 1
            fp16_bytes = in_dim * out_dim * 2
            result.append({
                "in_dim": in_dim,
                "out_dim": out_dim,
                "label": f"tensor:blk.{layer}.{suffix}.weight",
                "consumer_device": home,
                "resident_device": resident,
                "partner_offload": int(use_partner),
                "partner_arithmetic": "f16",
                "weight_offset": source_id * 1048576,
                "weight_bytes": 4096 + ordinal,
                "fp16_bytes": fp16_bytes,
                "resident_weight_bytes": fp16_bytes,
                "allocation_id": source_id,
                "used_calls": 1,
                "live": 1,
            })
    return result


def complete_plan_rows(
    variant: str, overflow_partner_layers: set[int]
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for layer in range(43):
        home, peer = (0, 1) if layer <= 21 else (3, 2)
        for ordinal, (_cls, suffix, in_dim, out_dim) in enumerate(PROJECTIONS):
            source_id = layer * len(PROJECTIONS) + ordinal + 1
            fp16_bytes = in_dim * out_dim * 2
            is_t256 = suffix == "attn_output_b"
            use_partner = is_t256 and (
                variant == "all-partner"
                or (variant == "balanced" and layer % 2 == 1)
                or (variant == "overflow" and layer in overflow_partner_layers)
            )
            if variant == "native" and is_t256:
                resident = -1
                target = home
                fallback = -1
                locked = 0
                status = "unadmitted"
            else:
                resident = peer if use_partner else home
                target = resident
                fallback = (
                    peer
                    if is_t256 and (
                        use_partner
                        or (variant == "overflow" and layer in range(15, 22))
                    )
                    else -1
                )
                locked = int(
                    is_t256 and (
                        variant == "all-partner"
                        or (variant == "balanced" and layer % 2 == 1)
                    )
                )
                status = "partner" if use_partner else "home"
            result.append({
                "sequence": len(result),
                "label": f"tensor:blk.{layer}.{suffix}.weight",
                "consumer_device": home,
                "fallback_device": fallback,
                "target_device": target,
                "placement_locked": locked,
                "resident_device": resident,
                "weight_offset": source_id * 1048576,
                "weight_bytes": 4096 + ordinal,
                "in_dim": in_dim,
                "out_dim": out_dim,
                "fp16_bytes": fp16_bytes,
                "status": status,
            })
    return result


def make_fixture(
    root: Path,
    repeats: int = 5,
    overflow_partner_layers: tuple[int, ...] = tuple(range(15, 22)),
    complete_dense_cache: bool = False,
) -> None:
    overflow_partner_layer_set = set(overflow_partner_layers)
    variants = STRICT_VARIANTS if complete_dense_cache else LEGACY_VARIANTS
    strict_manifest = (
        "ctx_alloc=262273\nnative_control=native-t256-only\n"
        "prefill_attention_head_split=off\n"
        "require_complete_dense_cache=1\n"
        "dense_cache_scope=complete-production-active\n"
        "dense_cache_plan_candidates=344\n"
        "dense_cache_native_live_bindings=301\n"
        "dense_cache_min_free_mib=512\n"
        "dense_cache_inventory=q_a:43,q_b:43,kv:43,output_a:43,output_b:43,shared_down:43,gate_up:86\n"
        if complete_dense_cache else ""
    )
    (root / "manifest.txt").write_text(
        "gpu_devices=0,3,1,2\ngpu_vram=auto\nstage_split=22/21\n"
        f"variants={','.join(variants)}\n"
        "overflow_partner_eligibility=15-21\n" + strict_manifest,
        encoding="utf-8",
    )
    run_rows = []
    for repeat in range(1, repeats + 1):
        for slot in range(len(variants)):
            variant = variants[(slot + repeat - 1) % len(variants)]
            stem = root / "runs" / f"{variant}-r{repeat}"
            csv_path = stem.with_suffix(".csv")
            log_path = stem.with_suffix(".log")
            audit_path = Path(f"{stem}.q8-audit.csv")
            bindings_path = Path(f"{stem}.bindings.csv")
            allocations_path = Path(f"{stem}.allocations.csv")
            plan_path = Path(f"{stem}.plan.csv")
            memory_path = Path(f"{stem}.memory.csv")
            speed = {
                "native": 300, "all-local": 500, "balanced": 490,
                "overflow": 480, "all-partner": 440,
            }[variant]
            write(csv_path, ["ctx_tokens", "prefill_tps"], [
                {"ctx_tokens": 2048, "prefill_tps": speed + repeat},
                {"ctx_tokens": 4096, "prefill_tps": speed - 20 + repeat},
            ])
            placement = "overflow" if variant == "native" else variant
            log_path.write_text(
                "ds4: CUDA EP forced pipeline split 22/21\n"
                f"ds4: benefit plan t256-placement={placement}\n"
                + ("" if variant == "native" else "T256-output_b=43/43\n"),
                encoding="utf-8",
            )
            counts = {
                "native": (0, 0), "all-local": (43, 0),
                "balanced": (22, 21),
                "overflow": (
                    43 - len(overflow_partner_layer_set),
                    len(overflow_partner_layer_set),
                ),
                "all-partner": (0, 43),
            }[variant]
            binding_rows: list[dict[str, object]] = []
            if complete_dense_cache:
                binding_rows = complete_bindings(
                    variant, overflow_partner_layer_set
                )
            elif variant != "native":
                for layer in range(43):
                    home, peer = (0, 1) if layer <= 21 else (3, 2)
                    use_partner = (
                        variant == "all-partner"
                        or (variant == "balanced" and layer % 2 == 1)
                        or (
                            variant == "overflow"
                            and layer in overflow_partner_layer_set
                        )
                    )
                    binding_rows.append(
                        binding(layer, home, peer if use_partner else home)
                    )
            t256_binding_rows = [
                row for row in binding_rows
                if "attn_output_b" in str(row["label"])
            ]
            self_check = (
                sum(row["partner_offload"] == 0 for row in t256_binding_rows),
                sum(row["partner_offload"] == 1 for row in t256_binding_rows),
            )
            if self_check != counts:
                raise AssertionError((variant, self_check, counts))
            write(
                bindings_path,
                [
                    "in_dim", "out_dim", "label", "consumer_device",
                    "resident_device", "partner_offload", "partner_arithmetic",
                    "weight_offset", "weight_bytes", "fp16_bytes",
                    "resident_weight_bytes", "allocation_id", "used_calls", "live",
                ],
                binding_rows,
            )
            allocation_rows = []
            for row in binding_rows:
                allocation_rows.append({
                    "allocation_id": row["allocation_id"],
                    "storage_kind": "f16",
                    "half_rounded": 0,
                    "physical_device": row["resident_device"],
                    "weight_offset": row["weight_offset"],
                    "weight_bytes": row["weight_bytes"],
                    "in_dim": row["in_dim"],
                    "out_dim": row["out_dim"],
                    "resident_bytes": row["resident_weight_bytes"],
                    "logical_aliases": 1,
                    "live_aliases": 1,
                    "used_calls": 1,
                    "dead_bytes": 0,
                    "usage_tracking": 1,
                })
            write(
                allocations_path,
                [
                    "allocation_id", "storage_kind", "half_rounded",
                    "physical_device", "weight_offset", "weight_bytes",
                    "in_dim", "out_dim", "resident_bytes", "logical_aliases",
                    "live_aliases", "used_calls", "dead_bytes",
                    "usage_tracking",
                ],
                allocation_rows,
            )
            if complete_dense_cache:
                cached_by_device = {device: 0 for device in range(4)}
                for row in allocation_rows:
                    cached_by_device[int(row["physical_device"])] += int(
                        row["resident_bytes"]
                    )
                write(
                    memory_path,
                    [
                        "logical_tier", "physical_device", "free_bytes",
                        "total_bytes", "q8_fp16_cached_bytes",
                        "q8_fp16_reserve_bytes",
                    ],
                    [
                        {
                            "logical_tier": tier,
                            "physical_device": device,
                            "free_bytes": 2 * 1024**3,
                            "total_bytes": 48 * 1024**3,
                            "q8_fp16_cached_bytes": cached_by_device[device],
                            "q8_fp16_reserve_bytes": 768 * 1024**2,
                        }
                        for tier, device in enumerate((0, 3, 1, 2))
                    ],
                )
                plan_rows = complete_plan_rows(
                    variant, overflow_partner_layer_set
                )
                write(
                    plan_path,
                    [
                        "sequence", "label", "consumer_device",
                        "fallback_device", "target_device", "placement_locked",
                        "resident_device", "weight_offset", "weight_bytes",
                        "in_dim", "out_dim", "fp16_bytes", "status",
                    ],
                    plan_rows,
                )
            audit_rows = []
            for layer in range(43):
                if variant == "native": result = "native_q8"
                elif variant == "all-local": result = "f16_hit"
                elif variant == "balanced":
                    result = "f16_partner_hit" if layer % 2 else "f16_hit"
                elif variant == "overflow":
                    result = (
                        "f16_partner_hit"
                        if layer in overflow_partner_layer_set
                        else "f16_hit"
                    )
                else: result = "f16_partner_hit"
                audit_rows.append({
                    "module": "attention_output", "label": "attn_output_b",
                    "layer": layer, "in_dim": 8192, "out_dim": 4096,
                    "physical_device": (
                        (1 if layer <= 21 else 2)
                        if result == "f16_partner_hit"
                        else (0 if layer <= 21 else 3)
                    ),
                    "result": result,
                    "reason": {
                        "native_q8": "disabled_t256_by_env",
                        "f16_hit": "resident",
                        "f16_partner_hit": "nvlink_offload",
                    }[result],
                })
            write(
                audit_path,
                [
                    "module", "label", "layer", "physical_device", "in_dim",
                    "out_dim", "result", "reason",
                ],
                audit_rows,
            )
            run_rows.append({
                "repeat": repeat, "slot": slot + 1, "variant": variant,
                "csv": csv_path, "log": log_path, "audit": audit_path,
                "bindings": bindings_path, "allocations": allocations_path,
                **({"plan": plan_path} if complete_dense_cache else {}),
                **({"memory": memory_path} if complete_dense_cache else {}),
            })
    write(
        root / "runs.tsv",
        [
            "repeat", "slot", "variant", "csv", "log", "audit",
            "bindings", "allocations",
            *(["plan"] if complete_dense_cache else []),
            *(["memory"] if complete_dense_cache else []),
        ],
        run_rows,
        "\t",
    )


class T256PlacementSummaryTests(unittest.TestCase):
    def run_summary(self, root: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SUMMARIZER), str(root)],
            capture_output=True, text=True, check=False,
        )

    def test_five_way_counterbalanced_result(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-t256-placement-") as raw:
            root = Path(raw)
            make_fixture(root)
            result = self.run_summary(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads((root / "t256-placement.json").read_text())
            self.assertTrue(payload["experiment_integrity"])
            self.assertTrue(payload["high_confidence_counterbalanced"])
            self.assertEqual(payload["performance"]["all-local"]["2048"]["median_tps"], 503)
            self.assertIn("All partner", (root / "summary.md").read_text())

    def test_single_repeat_is_labeled_screen_only(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-t256-screen-") as raw:
            root = Path(raw)
            make_fixture(root, repeats=1)
            result = self.run_summary(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads((root / "t256-placement.json").read_text())
            self.assertFalse(payload["high_confidence_counterbalanced"])
            self.assertIn("SCREEN ONLY", (root / "summary.md").read_text())

    def test_zero_natural_overflow_is_valid_and_recorded(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-t256-zero-overflow-") as raw:
            root = Path(raw)
            make_fixture(root, repeats=1, overflow_partner_layers=())
            result = self.run_summary(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads((root / "t256-placement.json").read_text())
            self.assertEqual(payload["overflow_observed"], [{
                "repeat": 1,
                "local_bindings": 43,
                "partner_bindings": 0,
                "partner_layers": [],
            }])
            self.assertIn("r1=43/0", (root / "summary.md").read_text())

    def test_ineligible_overflow_partner_layer_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-t256-bad-overflow-layer-") as raw:
            root = Path(raw)
            make_fixture(root, repeats=1, overflow_partner_layers=(14,))
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("ineligible partner layers [14]", result.stderr)

    def test_partial_all_partner_binding_set_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-t256-bad-bindings-") as raw:
            root = Path(raw)
            make_fixture(root, repeats=1)
            path = root / "runs/all-partner-r1.bindings.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                items = list(csv.DictReader(handle))
            write(path, list(items[0]), items[:-1])
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("expected 0+43", result.stderr)

    def test_wrong_balanced_runtime_layers_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-t256-bad-balanced-") as raw:
            root = Path(raw)
            make_fixture(root, repeats=1)
            path = root / "runs/balanced-r1.q8-audit.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                items = list(csv.DictReader(handle))
            items[1]["result"] = "f16_hit"
            write(path, list(items[0]), items)
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("partner layers", result.stderr)

    def test_dead_all_partner_physical_weight_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-t256-dead-allocation-") as raw:
            root = Path(raw)
            make_fixture(root, repeats=1)
            path = root / "runs/all-partner-r1.allocations.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                items = list(csv.DictReader(handle))
            items[0]["live_aliases"] = "0"
            items[0]["used_calls"] = "0"
            items[0]["dead_bytes"] = items[0]["resident_bytes"]
            write(path, list(items[0]), items)
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not one live physical weight", result.stderr)

    def test_complete_dense_cache_inventory_passes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-t256-complete-") as raw:
            root = Path(raw)
            make_fixture(root, repeats=1, complete_dense_cache=True)
            result = self.run_summary(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads((root / "t256-placement.json").read_text())
            self.assertTrue(payload["require_complete_dense_cache"])
            complete = payload["complete_dense_cache"]
            self.assertEqual(complete["plan_candidates_per_arm"], 344)
            self.assertEqual(complete["native_t256_only_live_bindings"], 301)
            self.assertTrue(complete["all_required_candidates_admitted"])
            self.assertEqual(complete["dead_or_unreferenced_allocations"], 0)
            self.assertIn(
                "Complete dense-cache integrity: **PASS**",
                (root / "summary.md").read_text(),
            )

    def test_complete_dense_cache_requires_256k_allocation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-t256-wrong-alloc-") as raw:
            root = Path(raw)
            make_fixture(root, repeats=1, complete_dense_cache=True)
            manifest = root / "manifest.txt"
            manifest.write_text(
                manifest.read_text().replace("ctx_alloc=262273", "ctx_alloc=32769")
            )
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("complete 256K dense-cache experiment", result.stderr)

    def test_unadmitted_complete_candidate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-t256-unadmitted-") as raw:
            root = Path(raw)
            make_fixture(root, repeats=1, complete_dense_cache=True)
            path = root / "runs/all-local-r1.plan.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                items = list(csv.DictReader(handle))
            items[0]["resident_device"] = "-1"
            items[0]["status"] = "unadmitted"
            write(path, list(items[0]), items)
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("dense FP16 plan is incomplete", result.stderr)

    def test_wrong_flash_projection_shape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-t256-wrong-shape-") as raw:
            root = Path(raw)
            make_fixture(root, repeats=1, complete_dense_cache=True)
            path = root / "runs/all-partner-r1.plan.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                items = list(csv.DictReader(handle))
            target = next(row for row in items if ".attn_q_b.weight" in row["label"])
            target["out_dim"] = "8192"
            target["fp16_bytes"] = str(int(target["in_dim"]) * 8192 * 2)
            write(path, list(items[0]), items)
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("shape/FP16 bytes", result.stderr)

    def test_strict_memory_floor_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-t256-low-free-") as raw:
            root = Path(raw)
            make_fixture(root, repeats=1, complete_dense_cache=True)
            path = root / "runs/all-local-r1.memory.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                items = list(csv.DictReader(handle))
            items[0]["free_bytes"] = str(511 * 1024**2)
            write(path, list(items[0]), items)
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("requires at least 512 MiB", result.stderr)

    def test_memory_cache_bytes_must_match_live_allocations(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-t256-memory-drift-") as raw:
            root = Path(raw)
            make_fixture(root, repeats=1, complete_dense_cache=True)
            path = root / "runs/all-partner-r1.memory.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                items = list(csv.DictReader(handle))
            items[0]["q8_fp16_cached_bytes"] = str(
                int(items[0]["q8_fp16_cached_bytes"]) + 64
            )
            write(path, list(items[0]), items)
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("do not match live allocations", result.stderr)

    def test_complete_non_native_descriptor_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-t256-descriptor-drift-") as raw:
            root = Path(raw)
            make_fixture(root, repeats=1, complete_dense_cache=True)
            prefix = root / "runs/all-local-r1"
            for suffix in ("plan.csv", "bindings.csv", "allocations.csv"):
                path = Path(f"{prefix}.{suffix}")
                with path.open(newline="", encoding="utf-8") as handle:
                    items = list(csv.DictReader(handle))
                target = next(
                    row for row in items if "attn_q_a" in row.get("label", "")
                ) if suffix != "allocations.csv" else items[0]
                target["weight_offset"] = str(int(target["weight_offset"]) + 128)
                write(path, list(items[0]), items)
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("identical complete dense plan inventories", result.stderr)

    def test_dead_non_t256_allocation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-t256-dead-dense-") as raw:
            root = Path(raw)
            make_fixture(root, repeats=1, complete_dense_cache=True)
            path = root / "runs/all-partner-r1.allocations.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                items = list(csv.DictReader(handle))
            target = next(
                row for row in items
                if not (row["in_dim"] == "8192" and row["out_dim"] == "4096")
            )
            target["live_aliases"] = "0"
            target["used_calls"] = "0"
            target["dead_bytes"] = target["resident_bytes"]
            write(path, list(items[0]), items)
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("is dead, aliased, or contains dead bytes", result.stderr)

    def test_unreferenced_dense_allocation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-t256-unreferenced-") as raw:
            root = Path(raw)
            make_fixture(root, repeats=1, complete_dense_cache=True)
            path = root / "runs/balanced-r1.allocations.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                items = list(csv.DictReader(handle))
            extra = dict(items[0])
            extra["allocation_id"] = "99999"
            items.append(extra)
            write(path, list(items[0]), items)
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not one-to-one with bindings", result.stderr)

    def test_native_control_must_preserve_non_t256_cache(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-t256-native-all-off-") as raw:
            root = Path(raw)
            make_fixture(root, repeats=1, complete_dense_cache=True)
            plan = root / "runs/native-r1.plan.csv"
            bindings = root / "runs/native-r1.bindings.csv"
            allocations = root / "runs/native-r1.allocations.csv"
            with plan.open(newline="", encoding="utf-8") as handle:
                plan_fields = next(csv.reader(handle))
            with bindings.open(newline="", encoding="utf-8") as handle:
                binding_fields = next(csv.reader(handle))
            with allocations.open(newline="", encoding="utf-8") as handle:
                allocation_fields = next(csv.reader(handle))
            write(plan, plan_fields, [])
            write(bindings, binding_fields, [])
            write(allocations, allocation_fields, [])
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("incomplete production candidate inventory", result.stderr)

    def test_native_non_t256_descriptor_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-t256-native-drift-") as raw:
            root = Path(raw)
            make_fixture(root, repeats=1, complete_dense_cache=True)
            prefix = root / "runs/native-r1"
            for suffix in ("plan.csv", "bindings.csv", "allocations.csv"):
                path = Path(f"{prefix}.{suffix}")
                with path.open(newline="", encoding="utf-8") as handle:
                    items = list(csv.DictReader(handle))
                target = next(
                    row for row in items if "attn_q_a" in row.get("label", "")
                ) if suffix != "allocations.csv" else items[0]
                target["weight_offset"] = str(int(target["weight_offset"]) + 256)
                write(path, list(items[0]), items)
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "identical complete dense plan inventories",
                result.stderr,
            )


if __name__ == "__main__":
    unittest.main()
