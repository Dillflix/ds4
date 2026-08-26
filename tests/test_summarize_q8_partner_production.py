#!/usr/bin/env python3
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARIZER = ROOT / "speed-bench" / "summarize-q8-partner-production.py"
CONTEXTS = (16384, 32768)


def write_table(path: Path, fieldnames: list[str], rows: list[dict[str, object]],
                delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def quality_rows(nll_delta_per_token: float = 0.0) -> list[dict[str, object]]:
    rows = []
    for index in range(100):
        target_tokens = 4
        avg_nll = 1.0 + nll_delta_per_token
        rows.append({
            "id": f"case_{index:03d}",
            "target_tokens": target_tokens,
            "nll": avg_nll * target_tokens,
            "avg_nll": avg_nll,
            "first_match": 1,
            "greedy_lcp": 4,
        })
    return rows


def make_fixture(
    root: Path,
    quality_delta: float = 0.0,
    local_tps: float = 256.0,
    default_tps: float = 290.0,
) -> None:
    (root / "manifest.txt").parent.mkdir(parents=True, exist_ok=True)
    (root / "manifest.txt").write_text(
        "gpu_devices=0,3,1,2\n"
        "gpu_vram=auto\n"
        "stage_split=22/21\n"
        "ctx_start=16384\n"
        "ctx_max=32768\n"
        "step_mul=2\n"
        "quality_ctx=32769\n"
        "gpu_exactness_test=required-and-enabled\n",
        encoding="utf-8",
    )
    (root / "planner-unit.log").write_text(
        "test_engine_mgpu_placement: 200/200 checks passed (0 failed)\n",
        encoding="utf-8",
    )
    (root / "gpu-exactness.log").write_text(
        "q8 partner projection exactness OK (3 classes)\n",
        encoding="utf-8",
    )
    quality_fields = [
        "id", "target_tokens", "nll", "avg_nll", "first_match", "greedy_lcp",
    ]
    write_table(
        root / "quality/local.tsv", quality_fields, quality_rows(), delimiter="\t"
    )
    write_table(
        root / "quality/default.tsv", quality_fields,
        quality_rows(quality_delta), delimiter="\t",
    )

    audit_fields = [
        "module", "label", "physical_device", "in_dim", "out_dim",
        "result", "reason",
    ]
    write_table(root / "quality/local.q8-audit.csv", audit_fields, [])
    write_table(
        root / "quality/default.q8-audit.csv",
        audit_fields,
        [{
            "module": "attn_output_b",
            "label": "attn_output_b",
            "physical_device": 1,
            "in_dim": 8192,
            "out_dim": 4096,
            "result": "f16_partner_hit",
            "reason": "nvlink_offload",
        }],
    )
    binding_fields = ["partner_offload", "in_dim", "out_dim"]
    write_table(
        root / "quality/local.bindings.csv",
        binding_fields,
        [{"partner_offload": 0, "in_dim": 1024, "out_dim": 32768}],
    )
    write_table(
        root / "quality/default.bindings.csv",
        binding_fields,
        [{"partner_offload": 1, "in_dim": 8192, "out_dim": 4096}],
    )
    (root / "quality/local.log").write_text(
        "score_official: runtime_path=production\n"
        "ds4: CUDA EP forced pipeline split 22/21\n",
        encoding="utf-8",
    )
    (root / "quality/default.log").write_text(
        "score_official: runtime_path=production\n"
        "ds4: CUDA EP forced pipeline split 22/21\n"
        "partner-classes=t256\n"
        "ds4: CUDA q8 fp16 partner summary: calls=1 activation=0.01 GiB "
        "result=0.01 GiB f16-result-calls=0\n",
        encoding="utf-8",
    )

    runs = []
    for repeat in (1, 2, 3):
        order = (("local", local_tps), ("default", default_tps))
        if repeat == 2:
            order = tuple(reversed(order))
        for slot, (variant, tps) in enumerate(order, 1):
            csv_path = root / "performance/runs" / f"{variant}-r{repeat}.csv"
            write_table(
                csv_path,
                ["ctx_tokens", "prefill_tokens", "prefill_tps"],
                [
                    {
                        "ctx_tokens": context,
                        "prefill_tokens": 16384,
                        "prefill_tps": tps,
                    }
                    for context in CONTEXTS
                ],
            )
            runs.append({
                "repeat": repeat,
                "slot": slot,
                "variant": variant,
                "csv": csv_path,
            })

        write_table(
            root / "performance/runs" / f"local-r{repeat}.bindings.csv",
            binding_fields,
            [{"partner_offload": 0, "in_dim": 1024, "out_dim": 32768}],
        )
        write_table(
            root / "performance/runs" / f"default-r{repeat}.bindings.csv",
            binding_fields,
            [
                {"partner_offload": 1, "in_dim": 8192, "out_dim": 4096}
                for _ in range(10)
            ],
        )

    write_table(
        root / "performance/runs.tsv",
        ["repeat", "slot", "variant", "csv"],
        runs,
        delimiter="\t",
    )
    write_table(
        root / "performance/class-evidence.csv",
        ["repeat", "variant", "evidence_status"],
        [
            {"repeat": repeat, "variant": "default", "evidence_status": "ok"}
            for repeat in (1, 2, 3)
        ],
    )
    write_table(
        root / "performance/logit-comparison.csv",
        ["repeat", "variant", "frontier", "top1_equal"],
        [
            {
                "repeat": repeat,
                "variant": "default",
                "frontier": f"frontier_{context:06d}.logits.json",
                "top1_equal": 1,
            }
            for repeat in (1, 2, 3)
            for context in CONTEXTS
        ],
    )
    write_table(
        root / "performance/logit-determinism.csv",
        ["repeat", "variant", "frontier", "exact"],
        [
            {
                "repeat": repeat,
                "variant": variant,
                "frontier": f"frontier_{context:06d}.logits.json",
                "exact": 1,
            }
            for repeat in (2, 3)
            for variant in ("local", "default")
            for context in CONTEXTS
        ],
    )


class SummarizeQ8PartnerProductionTests(unittest.TestCase):
    def run_summarizer(self, root: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SUMMARIZER), str(root)],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_happy_path_passes_all_gates(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-q8-production-pass-") as raw:
            root = Path(raw)
            make_fixture(root)

            result = self.run_summarizer(root)

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads((root / "acceptance.json").read_text())
            self.assertTrue(payload["accepted"])
            self.assertEqual(payload["scope"]["contexts"], [16384, 32768])
            self.assertFalse(payload["scope"]["validates_64k"])
            self.assertTrue(payload["quality"]["pass"])
            self.assertTrue(all(row["pass"] == 1 for row in payload["performance"]))
            self.assertEqual(
                payload["evidence"]["partner_bindings_per_run"], [10, 10, 10]
            )
            self.assertTrue(payload["evidence"]["bindings_t256_only"])
            self.assertTrue(payload["evidence"]["planner_and_gpu_exactness_tests"])
            self.assertTrue(payload["evidence"]["run_order_counterbalanced"])
            self.assertTrue(payload["quality"]["production_policy_evidence"])
            self.assertIn("Overall: **PASS**", (root / "summary.md").read_text())
            self.assertIn(
                "64K is not validated", (root / "summary.md").read_text()
            )

    def test_64k_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="ds4-q8-production-64k-rejected-"
        ) as raw:
            root = Path(raw)
            make_fixture(root)
            manifest = root / "manifest.txt"
            text = manifest.read_text(encoding="utf-8")
            manifest.write_text(
                text.replace("ctx_max=32768", "ctx_max=65536").replace(
                    "quality_ctx=32769", "quality_ctx=65537"
                ),
                encoding="utf-8",
            )

            result = self.run_summarizer(root)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("fixed production validation target", result.stderr)

    def test_quality_regression_fails_acceptance(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="ds4-q8-production-quality-fail-"
        ) as raw:
            root = Path(raw)
            make_fixture(root, quality_delta=0.01)

            result = self.run_summarizer(root)

            self.assertEqual(result.returncode, 1, result.stderr)
            payload = json.loads((root / "acceptance.json").read_text())
            self.assertFalse(payload["accepted"])
            self.assertFalse(payload["quality"]["pass"])
            self.assertGreater(
                payload["quality"]["bootstrap_upper95_delta_nll_per_token"],
                0.002,
            )
            self.assertTrue(all(row["pass"] == 1 for row in payload["performance"]))
            for key in (
                "class_pure", "top1_equal", "repeat_deterministic",
                "bindings_t256_only",
            ):
                self.assertTrue(payload["evidence"][key])
            self.assertEqual(
                payload["evidence"]["partner_bindings_per_run"], [10, 10, 10]
            )
            self.assertIn("Overall: **FAIL**", (root / "summary.md").read_text())

    def test_missing_binding_export_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="ds4-q8-production-missing-binding-"
        ) as raw:
            root = Path(raw)
            make_fixture(root)
            (root / "performance/runs/default-r2.bindings.csv").unlink()

            result = self.run_summarizer(root)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("exactly one local/default export per repeat", result.stderr)

    def test_missing_class_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="ds4-q8-production-missing-evidence-"
        ) as raw:
            root = Path(raw)
            make_fixture(root)
            evidence = root / "performance/class-evidence.csv"
            with evidence.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))[:-1]
            write_table(
                evidence,
                ["repeat", "variant", "evidence_status"],
                rows,
            )

            result = self.run_summarizer(root)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("exactly one default row per repeat", result.stderr)

    def test_fixed_run_order_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="ds4-q8-production-fixed-order-"
        ) as raw:
            root = Path(raw)
            make_fixture(root)
            runs_path = root / "performance/runs.tsv"
            with runs_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            for row in rows:
                row["slot"] = 1 if row["variant"] == "local" else 2
            write_table(
                runs_path,
                ["repeat", "slot", "variant", "csv"],
                rows,
                delimiter="\t",
            )

            result = self.run_summarizer(root)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("run order is not counterbalanced", result.stderr)

    def test_near_threshold_three_repeat_result_requires_five(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="ds4-q8-production-near-threshold-"
        ) as raw:
            root = Path(raw)
            make_fixture(root, local_tps=180.0, default_tps=193.5)

            result = self.run_summarizer(root)

            self.assertEqual(result.returncode, 1, result.stderr)
            payload = json.loads((root / "acceptance.json").read_text())
            self.assertFalse(payload["accepted"])
            self.assertTrue(payload["extend_to_five_repeats"])
            self.assertTrue(all(row["pass"] == 1 for row in payload["performance"]))


if __name__ == "__main__":
    unittest.main()
