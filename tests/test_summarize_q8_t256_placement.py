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
VARIANTS = ("native", "all-local", "balanced", "overflow", "all-partner")


def write(path: Path, fields: list[str], items: list[dict[str, object]], delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(items)


def binding(layer: int, consumer: int, resident: int) -> dict[str, object]:
    return {
        "in_dim": 8192,
        "out_dim": 4096,
        "label": f"tensor:blk.{layer}.attn_output_b.weight",
        "consumer_device": consumer,
        "resident_device": resident,
        "partner_offload": int(consumer != resident),
        "partner_arithmetic": "f16",
    }


def make_fixture(root: Path, repeats: int = 5) -> None:
    (root / "manifest.txt").write_text(
        "gpu_devices=0,3,1,2\ngpu_vram=auto\nstage_split=22/21\n"
        "variants=native,all-local,balanced,overflow,all-partner\n",
        encoding="utf-8",
    )
    run_rows = []
    for repeat in range(1, repeats + 1):
        for slot in range(5):
            variant = VARIANTS[(slot + repeat - 1) % 5]
            stem = root / "runs" / f"{variant}-r{repeat}"
            csv_path = stem.with_suffix(".csv")
            log_path = stem.with_suffix(".log")
            audit_path = Path(f"{stem}.q8-audit.csv")
            bindings_path = Path(f"{stem}.bindings.csv")
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
                + ("" if variant == "native" else "T256-output_b=86/86\n"),
                encoding="utf-8",
            )
            counts = {
                "native": (0, 0), "all-local": (86, 0),
                "balanced": (65, 21), "overflow": (79, 7),
                "all-partner": (43, 43),
            }[variant]
            binding_rows = []
            if variant != "native":
                for layer in range(43):
                    home, peer = (0, 1) if layer <= 21 else (3, 2)
                    binding_rows.append(binding(layer, peer, peer))
                    use_partner = (
                        variant == "all-partner"
                        or (variant == "balanced" and layer % 2 == 1)
                        or (variant == "overflow" and 15 <= layer <= 21)
                    )
                    binding_rows.append(
                        binding(layer, home, peer if use_partner else home)
                    )
            self_check = (
                sum(row["partner_offload"] == 0 for row in binding_rows),
                sum(row["partner_offload"] == 1 for row in binding_rows),
            )
            if self_check != counts:
                raise AssertionError((variant, self_check, counts))
            write(
                bindings_path,
                [
                    "in_dim", "out_dim", "label", "consumer_device",
                    "resident_device", "partner_offload", "partner_arithmetic",
                ],
                binding_rows,
            )
            audit_rows = []
            for layer in range(43):
                if variant == "native": result = "native_q8"
                elif variant == "all-local": result = "f16_hit"
                elif variant == "balanced":
                    result = "f16_partner_hit" if layer % 2 else "f16_hit"
                elif variant == "overflow":
                    result = "f16_partner_hit" if 15 <= layer <= 21 else "f16_hit"
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
                "bindings": bindings_path,
            })
    write(
        root / "runs.tsv",
        ["repeat", "slot", "variant", "csv", "log", "audit", "bindings"],
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
            self.assertIn("expected 43+43", result.stderr)

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


if __name__ == "__main__":
    unittest.main()
