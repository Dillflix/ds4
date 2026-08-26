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
SUMMARIZER = (
    ROOT / "speed-bench" / "summarize-sm75-native-q4-t256-ab.py"
)
VARIANTS = (
    "standard-local",
    "standard-auto",
    "native-local",
    "native-auto",
)
CONTEXTS = (16384, 32768)
RUN_FIELDS = [
    "repeat", "slot", "variant", "model", "policy", "csv", "log",
    "audit", "bindings", "cache_before", "cache_after", "logits",
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


def make_fixture(
    root: Path,
    tps: dict[str, float] | None = None,
) -> Path:
    if tps is None:
        tps = {
            "standard-local": 100.0,
            "standard-auto": 110.0,
            "native-local": 120.0,
            "native-auto": 132.0,
        }
    audit_fields = [
        "module", "label", "physical_device", "in_dim", "out_dim",
        "result", "reason",
    ]
    binding_fields = ["partner_offload", "in_dim", "out_dim"]
    records: list[dict[str, object]] = []

    # A four-row Latin square: within the block, every slot sees every variant.
    for repeat in range(1, 5):
        order = VARIANTS[-(repeat - 1):] + VARIANTS[:-(repeat - 1)]
        if repeat == 1:
            order = VARIANTS
        for slot, variant in enumerate(order, 1):
            family, policy = variant.split("-", 1)
            stem = root / "runs" / f"{variant}-r{repeat}"
            csv_path = stem.with_suffix(".csv")
            log_path = stem.with_suffix(".log")
            audit_path = stem.with_suffix(".audit.csv")
            bindings_path = stem.with_suffix(".bindings.csv")
            cache_before = stem.with_suffix(".cache-before.csv")
            cache_after = stem.with_suffix(".cache-after.csv")
            logits = root / "logits" / f"{variant}-r{repeat}"
            logits.mkdir(parents=True, exist_ok=True)

            repeat_scale = 1.0 + 0.002 * (repeat - 1)
            write_table(
                csv_path,
                ["ctx_tokens", "prefill_tokens", "prefill_tps"],
                [
                    {
                        "ctx_tokens": context,
                        "prefill_tokens": 16384,
                        "prefill_tps": tps[variant] * repeat_scale,
                    }
                    for context in CONTEXTS
                ],
            )

            log_lines = [
                "ds4: CUDA q8 fp16 benefit plan candidates=100 "
                f"partner-classes={'none' if policy == 'local' else 't256'}"
            ]
            if family == "native":
                log_lines.append(
                    "ds4: SM75 native routed-Q4 layout enabled "
                    "(packed A/W, planner=cost, gate=tile8, down=full-stage)"
                )
            if policy == "auto":
                log_lines.append(
                    "ds4: CUDA q8 fp16 partner summary: calls=8 "
                    "activation=0.10 GiB result=0.10 GiB f16-result-calls=0"
                )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

            audit_rows: list[dict[str, object]] = []
            binding_rows: list[dict[str, object]] = [
                {"partner_offload": 0, "in_dim": 1024, "out_dim": 32768}
            ]
            if policy == "auto":
                audit_rows.append({
                    "module": "attention",
                    "label": "blk.9.attn_output_b.weight",
                    "physical_device": 1,
                    "in_dim": 8192,
                    "out_dim": 4096,
                    "result": "f16_partner_hit",
                    "reason": "nvlink_offload",
                })
                binding_rows.append({
                    "partner_offload": 1,
                    "in_dim": 8192,
                    "out_dim": 4096,
                })
            write_table(audit_path, audit_fields, audit_rows)
            write_table(bindings_path, binding_fields, binding_rows)
            cache_text = "device,offset,bytes\n0,1024,67108864\n"
            cache_before.write_text(cache_text, encoding="utf-8")
            cache_after.write_text(cache_text, encoding="utf-8")

            # Native packing must preserve the result within each Q8 policy.
            # The local and auto policies may legitimately differ from each
            # other because an overflowed T256 projection changes arithmetic.
            values = (1.0, -2.0, 3.5, 0.25) if policy == "local" else (
                1.0, -2.0, 3.5, 0.2501
            )
            raw = struct.pack("<4f", *values)
            for context in CONTEXTS:
                (logits / f"frontier_{context:06d}.logits.f32").write_bytes(raw)

            records.append({
                "repeat": repeat,
                "slot": slot,
                "variant": variant,
                "model": f"/models/{family}.gguf",
                "policy": policy,
                "csv": csv_path,
                "log": log_path,
                "audit": audit_path,
                "bindings": bindings_path,
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

    def test_valid_four_way_ab_passes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-native-t256-pass-") as raw:
            root = Path(raw)
            runs_path = make_fixture(root)

            result = self.run_summarizer(root, runs_path)

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads((root / "out/acceptance.json").read_text())
            self.assertTrue(payload["accepted"])
            self.assertEqual(payload["scope"]["repeats"], 4)
            self.assertEqual(payload["scope"]["contexts"], list(CONTEXTS))
            self.assertFalse(payload["scope"]["official_quality_suite"])
            self.assertTrue(all(payload["gates"].values()))
            self.assertEqual(
                payload["evidence"]["cross_layout_comparisons"], 16
            )
            self.assertEqual(
                payload["evidence"]["repeat_determinism_comparisons"], 24
            )

            paired = read_csv(root / "out/paired-effects.csv")
            self.assertEqual(len(paired), 8)
            self.assertAlmostEqual(
                float(paired[0]["native_auto_over_standard_local"]), 1.32
            )
            self.assertAlmostEqual(float(paired[0]["interaction"]), 1.0)
            context_rows = read_csv(root / "out/context-summary.csv")
            self.assertEqual(len(context_rows), len(CONTEXTS) * 6)
            self.assertIn("Overall: **PASS**", result.stdout)
            self.assertIn(
                "not assumed to be additive or multiplicative", result.stdout
            )
            self.assertIn("quality-isolation gate still applies", result.stdout)

    def test_unbalanced_four_way_order_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="ds4-native-t256-order-fail-"
        ) as raw:
            root = Path(raw)
            runs_path = make_fixture(root)
            rows = read_runs(runs_path)
            fixed_slots = {variant: index + 1 for index, variant in enumerate(VARIANTS)}
            for row in rows:
                row["slot"] = str(fixed_slots[row["variant"]])
            write_table(runs_path, RUN_FIELDS, rows, delimiter="\t")

            result = self.run_summarizer(root, runs_path)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("run order is not counterbalanced", result.stderr)

    def test_nonexact_native_logits_fail_acceptance(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="ds4-native-t256-logit-fail-"
        ) as raw:
            root = Path(raw)
            runs_path = make_fixture(root)
            target = (
                root / "logits/native-auto-r2/"
                "frontier_016384.logits.f32"
            )
            target.write_bytes(struct.pack("<4f", 1.0, -2.0, 3.5, 9.0))

            result = self.run_summarizer(root, runs_path)

            self.assertEqual(result.returncode, 1, result.stderr)
            payload = json.loads((root / "out/acceptance.json").read_text())
            self.assertFalse(payload["accepted"])
            self.assertFalse(
                payload["gates"]["cross_layout_raw_logits_exact"]
            )
            exactness = read_csv(root / "out/exactness.csv")
            self.assertTrue(any(row["exact"] == "0" for row in exactness))

    def test_mismatched_frontier_work_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="ds4-native-t256-work-fail-"
        ) as raw:
            root = Path(raw)
            runs_path = make_fixture(root)
            csv_path = root / "runs/native-local-r3.csv"
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[1]["prefill_tokens"] = "8192"
            write_table(
                csv_path,
                ["ctx_tokens", "prefill_tokens", "prefill_tps"],
                rows,
            )

            result = self.run_summarizer(root, runs_path)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("mismatched prefill work", result.stderr)

    def test_mismatched_frontier_set_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="ds4-native-t256-frontier-fail-"
        ) as raw:
            root = Path(raw)
            runs_path = make_fixture(root)
            csv_path = root / "runs/native-auto-r4.csv"
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            write_table(
                csv_path,
                ["ctx_tokens", "prefill_tokens", "prefill_tps"],
                rows[:-1],
            )

            result = self.run_summarizer(root, runs_path)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("mismatched frontier set", result.stderr)

    def test_combined_regression_fails_even_when_t256_gate_passes(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="ds4-native-t256-regression-"
        ) as raw:
            root = Path(raw)
            runs_path = make_fixture(root, {
                "standard-local": 100.0,
                "standard-auto": 110.0,
                "native-local": 80.0,
                "native-auto": 84.8,
            })

            result = self.run_summarizer(root, runs_path)

            self.assertEqual(result.returncode, 1, result.stderr)
            payload = json.loads((root / "out/acceptance.json").read_text())
            self.assertFalse(payload["accepted"])
            self.assertTrue(
                payload["gates"]["native_auto_over_native_local_16k_32k"]
            )
            self.assertFalse(
                payload["gates"][
                    "native_auto_over_standard_local_no_regression"
                ]
            )
            self.assertTrue(
                payload["gates"]["cross_layout_raw_logits_exact"]
            )


if __name__ == "__main__":
    unittest.main()
