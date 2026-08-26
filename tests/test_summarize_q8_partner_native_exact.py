#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "speed-bench/summarize-q8-partner-native-exact.py"
BINDING_FIELDS = (
    "consumer_device", "resident_device", "partner_offload", "weight_offset",
    "weight_bytes", "in_dim", "out_dim", "fp16_bytes",
    "partner_scratch_tokens", "resident_weight_bytes", "partner_arithmetic",
    "label",
)
AUDIT_FIELDS = (
    "sequence", "module", "label", "layer", "token_offset",
    "physical_device", "weight_offset", "weight_bytes", "in_dim", "out_dim",
    "fp16_bytes", "result", "reason", "cache_bytes_after",
)


def write_csv(path: Path, fields: tuple[str, ...], table: list[dict[str, object]],
              delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(table)


def binding_rows(native: bool) -> list[dict[str, object]]:
    table: list[dict[str, object]] = [{
        "consumer_device": 0, "resident_device": 0, "partner_offload": 0,
        "weight_offset": 100, "weight_bytes": 34, "in_dim": 32,
        "out_dim": 1, "fp16_bytes": 64, "partner_scratch_tokens": 0,
        "resident_weight_bytes": 64, "partner_arithmetic": "f16",
        "label": "tensor:blk.0.attn_kv.weight",
    }]
    if native:
        for layer in range(15, 22):
            table.append({
                "consumer_device": 0, "resident_device": 1,
                "partner_offload": 1, "weight_offset": 1000 + layer,
                "weight_bytes": 35651584, "in_dim": 8192, "out_dim": 4096,
                "fp16_bytes": 67108864, "partner_scratch_tokens": 2048,
                "resident_weight_bytes": 35651584,
                "partner_arithmetic": "native-q8",
                "label": f"tensor:blk.{layer}.attn_output_b.weight",
            })
    return table


def audit_rows(calls_per_layer: int = 1) -> list[dict[str, object]]:
    table: list[dict[str, object]] = []
    sequence = 0
    for _ in range(calls_per_layer):
        for layer in range(15, 22):
            table.append({
                "sequence": sequence, "module": "attention_output",
                "label": "attn_output_b", "layer": layer, "token_offset": 0,
                "physical_device": 1, "weight_offset": 1000 + layer,
                "weight_bytes": 35651584, "in_dim": 8192, "out_dim": 4096,
                "fp16_bytes": 67108864, "result": "native_q8_partner_hit",
                "reason": "exact_sm75_mma", "cache_bytes_after": 1,
            })
            sequence += 1
    return table


class NativeExactSummaryTests(unittest.TestCase):
    def make_fixture(self, root: Path) -> None:
        (root / "quality").mkdir(parents=True)
        (root / "runs").mkdir()
        (root / "logits").mkdir()
        (root / "manifest.txt").write_text(
            "gpu_devices=0,3,1,2\n"
            "gpu_vram=auto\n"
            "stage_split=22/21\n"
            "quality_ctx=32769\n"
            "contexts=16384,32768\n"
            "t256_layers=15-21\n"
            "partner_arithmetic=native-q8\n"
            "home_plan=frozen\n",
            encoding="utf-8",
        )
        quality = "id\ttarget_tokens\tnll\n" + "".join(
            f"case_{index:03d}\t1\t1.0\n" for index in range(100)
        )
        (root / "quality/local.tsv").write_text(quality, encoding="utf-8")
        (root / "quality/native-q8.tsv").write_text(quality, encoding="utf-8")
        write_csv(root / "quality/local.bindings.csv", BINDING_FIELDS,
                  binding_rows(False))
        write_csv(root / "quality/native-q8.bindings.csv", BINDING_FIELDS,
                  binding_rows(True))
        write_csv(root / "quality/native-q8.q8-audit.csv", AUDIT_FIELDS,
                  audit_rows(100))

        run_rows: list[dict[str, object]] = []
        for repeat in range(1, 4):
            order = ("local", "native-q8") if repeat % 2 else (
                "native-q8", "local"
            )
            for slot, variant in enumerate(order, 1):
                stem = f"{variant}-r{repeat}"
                perf = root / f"runs/{stem}.csv"
                write_csv(perf, ("ctx_tokens", "prefill_tokens", "prefill_tps"), [
                    {"ctx_tokens": 16384, "prefill_tokens": 16384,
                     "prefill_tps": 500 if variant == "local" else 525},
                    {"ctx_tokens": 32768, "prefill_tokens": 16384,
                     "prefill_tps": 400 if variant == "local" else 420},
                ])
                bindings = root / f"runs/{stem}.bindings.csv"
                write_csv(bindings, BINDING_FIELDS,
                          binding_rows(variant == "native-q8"))
                audit = root / f"runs/{stem}.q8-audit.csv"
                write_csv(audit, AUDIT_FIELDS,
                          audit_rows(2) if variant == "native-q8" else [])
                logits = root / f"logits/{stem}"
                logits.mkdir()
                for context in (16384, 32768):
                    (logits / f"frontier_{context:06d}.logits.json").write_text(
                        json.dumps({"context": context, "top": [1, 2, 3]}),
                        encoding="utf-8",
                    )
                run_rows.append({
                    "repeat": repeat, "slot": slot, "variant": variant,
                    "csv": perf, "log": root / f"runs/{stem}.log",
                    "audit": audit, "bindings": bindings, "logits": logits,
                })
        write_csv(root / "runs.tsv",
                  ("repeat", "slot", "variant", "csv", "log", "audit",
                   "bindings", "logits"), run_rows, "\t")

    def run_summary(self, root: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(root)], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )

    def test_accepts_byte_exact_native_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_fixture(root)
            result = self.run_summary(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads((root / "native-exact.json").read_text())
            self.assertTrue(payload["quality_byte_exact"])
            self.assertEqual(payload["quality_partner_bindings"], 7)
            self.assertAlmostEqual(payload["performance"][0]["median_ratio"], 1.05)

    def test_rejects_non_native_partner_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_fixture(root)
            path = root / "quality/native-q8.bindings.csv"
            table = binding_rows(True)
            table[-1]["partner_arithmetic"] = "f16"
            write_csv(path, BINDING_FIELDS, table)
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("non-native-Q8", result.stderr)


if __name__ == "__main__":
    unittest.main()
