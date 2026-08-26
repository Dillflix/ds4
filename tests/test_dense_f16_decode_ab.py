#!/usr/bin/env python3
"""Tests for the primary native-Q8/source-F16 production evidence pass."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARIZER = ROOT / "speed-bench/summarize-dense-f16-decode-ab.py"
RUNNER = ROOT / "speed-bench/cuda-dense-f16-decode-ab.sh"
ARMS = ("native-q8", "source-f16")
FAMILIES = {
    "attn_q_a": ("attn_q_a.weight", (4096, 1024)),
    "attn_q_b": ("attn_q_b.weight", (1024, 32768)),
    "attn_kv": ("attn_kv.weight", (4096, 512)),
    "attn_output_a": ("attn_output_a.weight", (4096, 8192)),
    "attn_output_b": ("attn_output_b.weight", (8192, 4096)),
    "shared_gate": ("ffn_gate_shexp.weight", (4096, 2048)),
    "shared_up": ("ffn_up_shexp.weight", (4096, 2048)),
    "shared_down": ("ffn_down_shexp.weight", (2048, 4096)),
}
AUDIT_FIELDS = [
    "sequence", "module", "label", "layer", "token_offset",
    "physical_device", "weight_offset", "weight_bytes", "weight_type",
    "in_dim", "out_dim", "n_tokens", "executed_input_offset",
    "executed_input_count", "executed_output_offset", "executed_output_count", "backend",
    "placement", "result", "reason",
]
INVENTORY_FIELDS = [
    "arm", "name", "layer", "module", "type_id", "type_name", "dim0",
    "dim1", "elements", "encoded_bytes", "relative_offset", "data_offset",
]
SCORE_FIELDS = [
    "id", "target_tokens", "nll", "avg_nll", "first_match", "greedy_lcp",
    "api_target_tokens", "api_target_mae", "api_top1_count",
    "api_top1_match", "api_topn_ref", "api_topn_hit", "api_pair_total",
    "api_pair_agree",
]
BENCH_FIELDS = [
    "ctx_tokens", "prefill_tokens", "prefill_tps", "gen_tokens", "gen_tps",
    "gen_first_ms", "gen_steady_tokens", "gen_steady_tps", "kvcache_bytes",
]
PEER_LOG = "".join(
    f"ds4: peer access {edge} validated across 4 sizes x 4 iterations\n"
    for edge in ("0->1", "1->0", "2->3", "3->2")
)


def write_table(
    path: Path, fields: list[str], rows: list[dict[str, object]], delimiter: str = ","
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def score_rows(delta: float = 0.0) -> list[dict[str, object]]:
    return [
        {
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
        }
        for index in range(100)
    ]


def audit_rows(arm: str) -> list[dict[str, object]]:
    result = "native_q8" if arm == "native-q8" else "source_f16"
    rows: list[dict[str, object]] = []
    sequence = 0
    for layer in range(43):
        for module, (suffix, dims) in FAMILIES.items():
            fused = arm == "native-q8" and module in {"shared_gate", "shared_up"}
            backend = (
                "sm75_native_q8_fused_shared_mid"
                if fused
                else "sm75_native_q8" if arm == "native-q8" else "cublas_gemm_ex"
            )
            rows.append(
                {
                    "sequence": sequence,
                    "module": module,
                    "label": f"blk.{layer}.{suffix}",
                    "layer": layer,
                    "token_offset": 2048,
                    "physical_device": layer % 4,
                    "weight_offset": 4096 + sequence,
                    "weight_bytes": 1024,
                    "weight_type": "Q8_0" if arm == "native-q8" else "F16",
                    "in_dim": dims[0],
                    "out_dim": dims[1],
                    "executed_input_offset": 0,
                    "executed_input_count": dims[0],
                    "executed_output_offset": 0,
                    "executed_output_count": dims[1],
                    "n_tokens": 1,
                    "backend": backend,
                    "placement": "partner" if fused and layer % 2 else "local",
                    "result": result,
                    "reason": "executed",
                }
            )
            sequence += 1
    return rows


def tensor_inventory_rows(arm: str) -> list[dict[str, object]]:
    inventory_arm = "native-q8-source" if arm == "native-q8" else "source-f16"
    type_id = 8 if arm == "native-q8" else 1
    type_name = "Q8_0" if arm == "native-q8" else "F16"
    return [
        {
            "arm": inventory_arm,
            "name": row["label"],
            "layer": row["layer"],
            "module": row["module"],
            "type_id": type_id,
            "type_name": type_name,
            "dim0": row["in_dim"],
            "dim1": row["out_dim"],
            "elements": int(row["in_dim"]) * int(row["out_dim"]),
            "encoded_bytes": row["weight_bytes"],
            "relative_offset": row["weight_offset"],
            "data_offset": row["weight_offset"],
        }
        for row in audit_rows(arm)
    ]


def bench_rows(arm: str, gen_tokens: int = 16) -> list[dict[str, object]]:
    source = arm == "source-f16"
    return [
        {
            "ctx_tokens": context,
            "prefill_tokens": context if context == 2048 else context // 2,
            "prefill_tps": 330.0 if source else 300.0,
            "gen_tokens": gen_tokens,
            "gen_tps": 12.0 if source else 10.0,
            "gen_first_ms": 90.0 if source else 100.0,
            "gen_steady_tokens": gen_tokens - 1,
            "gen_steady_tps": 12.1 if source else 10.1,
            "kvcache_bytes": context * 1024,
        }
        for context in (2048, 8192, 32768)
    ]


def memory_text() -> str:
    header = (
        "timestamp, index, pci.bus_id, memory.total [MiB], "
        "memory.used [MiB], memory.free [MiB]\n"
    )
    return header + "".join(
        f"2026/08/26 00:00:00.000, {device}, 00000000:0{device}:00.0, "
        "49152, 40000, 9152\n"
        for device in range(4)
    )


def make_fixture(root: Path) -> None:
    q8 = (root / "models/q8.gguf").resolve()
    source = (root / "models/source-f16.gguf").resolve()
    q8.parent.mkdir(parents=True)
    q8.write_bytes(b"q8")
    source.write_bytes(b"f16")

    inventory = {
        "structural_integrity": True,
        "payload_hashing": False,
        "model_q8": {
            "path": str(q8),
            "file_bytes": q8.stat().st_size,
            "target_tensor_count": 344,
            "payload_extents": {"payload_extents_valid": 1},
        },
        "model_source_f16": {
            "path": str(source),
            "file_bytes": source.stat().st_size,
            "target_tensor_count": 344,
            "payload_extents": {"payload_extents_valid": 1},
        },
        "directory_parity": {"kv_metadata_byte_identical": True},
    }
    (root / "inventory").mkdir(parents=True)
    (root / "inventory/comparison.json").write_text(
        json.dumps(inventory), encoding="utf-8"
    )
    write_table(
        root / "inventory/model-q8.csv",
        INVENTORY_FIELDS,
        tensor_inventory_rows("native-q8"),
    )
    write_table(
        root / "inventory/model-source-f16.csv",
        INVENTORY_FIELDS,
        tensor_inventory_rows("source-f16"),
    )

    sidecar = root / "provenance/source-f16-provenance.txt"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(
        "\n".join(
            [
                "git_commit=" + "a" * 40,
                "git_dirty=true",
                "quantizer=/repo/gguf-tools/deepseek4-quantize",
                "quantizer_bytes=12345",
                "quantizer_device=1",
                "quantizer_inode=1001",
                "quantizer_mtime_ns=1760000000000000000",
                "quantizer_sha256=" + "b" * 64,
                "hf_repository=deepseek-ai/DeepSeek-V4-Flash-0731",
                "hf_revision=7872f01b1d1fe23eabc4c98b48bffcef5a386062",
                "hf_snapshot_path_revision_match=true",
                "hf_shard_content_authentication=not_performed",
                "hf_directory=/cache/hf/revision",
                "hf_index=/cache/hf/revision/model.safetensors.index.json",
                "hf_index_bytes=1234",
                "hf_index_device=1",
                "hf_index_inode=1002",
                "hf_index_mtime_epoch=1760000000",
                "hf_index_mtime_ns=1760000000000000000",
                "hf_index_sha256=" + "c" * 64,
                "hf_config=/cache/hf/revision/config.json",
                "hf_config_bytes=5678",
                "hf_config_device=1",
                "hf_config_inode=1003",
                "hf_config_mtime_epoch=1760000000",
                "hf_config_mtime_ns=1760000000000000000",
                "hf_config_sha256=" + "d" * 64,
                "hf_shard_count=1",
                "hf_shard_000_name=model-00001-of-00001.safetensors",
                "hf_shard_000_resolved=/cache/hf/blobs/" + "e" * 64,
                "hf_shard_000_resolved_basename=" + "e" * 64,
                "hf_shard_000_bytes=987654",
                "hf_shard_000_device=1",
                "hf_shard_000_inode=1004",
                "hf_shard_000_mtime_ns=1760000000000000000",
                f"template_model={q8}",
                f"template_model_bytes={q8.stat().st_size}",
                f"template_model_device={q8.stat().st_dev}",
                f"template_model_inode={q8.stat().st_ino}",
                f"template_model_mtime_epoch={int(q8.stat().st_mtime)}",
                f"template_model_mtime_ns={q8.stat().st_mtime_ns}",
                f"source_f16_model={source}",
                f"source_f16_model_bytes={source.stat().st_size}",
                f"source_f16_model_device={source.stat().st_dev}",
                f"source_f16_model_inode={source.stat().st_ino}",
                f"source_f16_model_mtime_epoch={int(source.stat().st_mtime)}",
                f"source_f16_model_mtime_ns={source.stat().st_mtime_ns}",
                "source_f16_selected=344",
                "source_f16_selected_bytes=11362369536",
                "template_q8_selected=344",
                "kv_metadata_byte_identical=true",
                "payload_extents_valid=true",
                "non_target_payload_copy=producer-guaranteed-not-independently-rehashed",
                "model_payload_hashing=disabled",
                "publication_protocol=model-last-rollback-journal-v1",
                "provenance_complete=true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    sidecar_sha = hashlib.sha256(sidecar.read_bytes()).hexdigest()

    (root / "manifest.txt").write_text(
        f"model_q8={q8}\nmodel_source_f16={source}\n"
        f"model_q8_bytes={q8.stat().st_size}\n"
        f"model_source_f16_bytes={source.stat().st_size}\n"
        "source_f16_provenance=provenance/source-f16-provenance.txt\n"
        f"source_f16_provenance_sha256={sidecar_sha}\n"
        "inventory=inventory/comparison.json\n"
        "q8_tensor_inventory=inventory/model-q8.csv\n"
        "source_f16_tensor_inventory=inventory/model-source-f16.csv\n"
        "runs_table=runs.tsv\n"
        "artifact_path_mode=root-relative\n"
        "gpu_devices=0,3,1,2\ngpu_vram=auto\nstage_split=22/21\n"
        "cuda_device_order=PCI_BUS_ID\n"
        "cuda_visible_devices=unset-required\n"
        "nvidia_visible_devices=unset-required\n"
        "required_direct_peer_links=0->1,1->0,2->3,3->2\n"
        "quality_ctx=32769\nprefill_chunk=2048\ncontexts=2048,8192,32768\n"
        "ctx_alloc=32897\ngen_tokens=16\nrepeats=4\n"
        "required_arms=native-q8,source-f16\nrotation=exact-ab-ba\n"
        "q8_derived_diagnostic=not-run-not-required\n"
        "dense_tensor_inventory=344\ndense_audit=coverage-only\n"
        "timed_dense_audit=disabled\nstructural_integrity=required\n"
        "source_provenance=required\n"
        "source_f16_lock=shared-nonblocking-held\n"
        "model_file_identity=bytes-device-inode-mtime\n"
        "hf_shard_content_authentication=not_performed\n"
        "non_target_payload_identity=producer-copy-guarantee-not-independently-rehashed\n"
        "model_payload_hashing=disabled\n",
        encoding="utf-8",
    )

    for name, text in {
        "git-status.txt": " M ds4_cuda.cu\n",
        "git-head.patch": "",
        "untracked-source-files.txt": "",
        "untracked-sources.none": "none\n",
        "toolchain-runtime.txt": "Linux test\nnvcc test\n",
        "binary-sha256.txt": "".join(
            f"{'f' * 64}  {path}\n"
            for path in (
                "./ds4-bench",
                "./gguf-tools/quality-testing/score_official",
                "./tests/test_engine_mgpu_placement",
                "./tests/test_gpu_xdev",
            )
        ),
    }.items():
        (root / f"provenance/{name}").write_text(text, encoding="utf-8")

    for arm_index, arm in enumerate(ARMS):
        write_table(
            root / f"quality/{arm}.tsv",
            SCORE_FIELDS,
            score_rows(arm_index * 0.0001),
            "\t",
        )
        write_table(
            root / f"quality/{arm}.dense-audit.csv", AUDIT_FIELDS, audit_rows(arm)
        )
        (root / f"quality/{arm}.log").write_text(
            "score_official: runtime_path=production\n"
            "ds4: CUDA EP forced pipeline split 22/21\n" + PEER_LOG,
            encoding="utf-8",
        )
        (root / "memory").mkdir(exist_ok=True)
        (root / f"memory/quality-{arm}.csv").write_text(
            memory_text(), encoding="utf-8"
        )

    runs: list[dict[str, object]] = []
    for repeat in range(1, 5):
        offset = (repeat - 1) % 2
        for slot in (1, 2):
            arm = ARMS[((slot - 1) + offset) % 2]
            stem = f"{arm}-r{repeat}"
            csv_rel = f"runs/{stem}.csv"
            log_rel = f"runs/{stem}.log"
            memory_rel = f"memory/{stem}.csv"
            write_table(root / csv_rel, BENCH_FIELDS, bench_rows(arm))
            (root / log_rel).write_text(
                "ds4: CUDA EP forced pipeline split 22/21\n" + PEER_LOG,
                encoding="utf-8",
            )
            (root / memory_rel).write_text(memory_text(), encoding="utf-8")
            runs.append(
                {
                    "repeat": repeat,
                    "slot": slot,
                    "arm": arm,
                    "csv": csv_rel,
                    "log": log_rel,
                    "memory": memory_rel,
                }
            )
    write_table(
        root / "runs.tsv",
        ["repeat", "slot", "arm", "csv", "log", "memory"],
        runs,
        "\t",
    )


class DecodeABSummaryTests(unittest.TestCase):
    def run_summary(self, root: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SUMMARIZER), str(root)],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_complete_primary_evidence_passes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_fixture(root)
            result = self.run_summary(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(
                (root / "dense-f16-decode-ab-results.json").read_text(encoding="utf-8")
            )
            self.assertTrue(payload["primary_experiment_integrity"])
            self.assertEqual(payload["q8_derived_diagnostic"], "not run and not required")
            self.assertTrue(payload["source_provenance"]["producer_git_dirty"])
            self.assertAlmostEqual(
                payload["performance"]["source-f16"]["decode_geomean_ratio_vs_native"],
                1.2,
            )

    def test_archived_summary_does_not_dereference_original_models(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_fixture(root)
            inventory = json.loads(
                (root / "inventory/comparison.json").read_text(encoding="utf-8")
            )
            Path(inventory["model_q8"]["path"]).unlink()
            Path(inventory["model_source_f16"]["path"]).unlink()
            result = self.run_summary(root)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_wrong_native_backend_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_fixture(root)
            path = root / "quality/native-q8.dense-audit.csv"
            rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
            rows[0]["backend"] = "sm75_native_q8_fastish"
            write_table(path, AUDIT_FIELDS, rows)
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("expected one of", result.stderr)

    def test_native_fused_shared_mid_decode_proof_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_fixture(root)
            path = root / "quality/native-q8.dense-audit.csv"
            rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
            gate = next(row for row in rows if row["module"] == "shared_gate")
            gate["backend"] = "sm75_native_q8"
            gate["placement"] = "local"
            write_table(path, AUDIT_FIELDS, rows)
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "fused shared-mid decode rectangle coverage differs", result.stderr
            )

    def test_label_substring_spoof_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_fixture(root)
            path = root / "quality/source-f16.dense-audit.csv"
            rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
            rows[0]["label"] += ".spoof"
            write_table(path, AUDIT_FIELDS, rows)
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("expected exactly", result.stderr)

    def test_wrong_audit_shape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_fixture(root)
            path = root / "quality/source-f16.dense-audit.csv"
            rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
            rows[0]["out_dim"] = "2048"
            write_table(path, AUDIT_FIELDS, rows)
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("has shape=4096x2048", result.stderr)

    def test_wrong_audit_weight_offset_is_rejected_by_inventory_join(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_fixture(root)
            path = root / "quality/native-q8.dense-audit.csv"
            rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
            rows[0]["weight_offset"] = str(int(rows[0]["weight_offset"]) + 32)
            write_table(path, AUDIT_FIELDS, rows)
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("payload identity is offset=", result.stderr)

    def test_wrong_audit_weight_bytes_is_rejected_by_inventory_join(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_fixture(root)
            path = root / "quality/source-f16.dense-audit.csv"
            rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
            rows[0]["weight_bytes"] = str(int(rows[0]["weight_bytes"]) + 1)
            write_table(path, AUDIT_FIELDS, rows)
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("payload identity is offset=", result.stderr)

    def test_missing_tensor_inventory_row_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_fixture(root)
            path = root / "inventory/model-q8.csv"
            rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
            write_table(path, INVENTORY_FIELDS, rows[1:])
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("native-q8 tensor inventory differs: missing=", result.stderr)

    def test_duplicate_tensor_inventory_row_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_fixture(root)
            path = root / "inventory/model-source-f16.csv"
            rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
            rows.append(dict(rows[0]))
            write_table(path, INVENTORY_FIELDS, rows)
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("duplicates inventory tensor", result.stderr)

    def test_partial_execution_rectangle_cannot_prove_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_fixture(root)
            path = root / "quality/source-f16.dense-audit.csv"
            rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
            row = next(
                item
                for item in rows
                if item["layer"] == "0" and item["module"] == "attn_output_b"
            )
            row["executed_input_count"] = str(int(row["in_dim"]) // 2)
            write_table(path, AUDIT_FIELDS, rows)
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("execution-rectangle coverage", result.stderr)

    def test_complementary_execution_rectangles_prove_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_fixture(root)
            path = root / "quality/source-f16.dense-audit.csv"
            rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
            row = next(
                item
                for item in rows
                if item["layer"] == "0" and item["module"] == "attn_output_b"
            )
            half = int(row["in_dim"]) // 2
            row["executed_input_count"] = str(half)
            complement = dict(row)
            complement["sequence"] = "9998"
            complement["executed_input_offset"] = str(half)
            complement["executed_input_count"] = str(int(row["in_dim"]) - half)
            rows.append(complement)
            write_table(path, AUDIT_FIELDS, rows)
            result = self.run_summary(root)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_out_of_bounds_execution_rectangle_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_fixture(root)
            path = root / "quality/native-q8.dense-audit.csv"
            rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
            rows[0]["executed_output_offset"] = rows[0]["out_dim"]
            write_table(path, AUDIT_FIELDS, rows)
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("execution rectangle", result.stderr)

    def test_unknown_audit_module_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_fixture(root)
            path = root / "quality/native-q8.dense-audit.csv"
            rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
            extra = dict(rows[0])
            extra["sequence"] = "9999"
            extra["module"] = "unclassified_dense_projection"
            rows.append(extra)
            write_table(path, AUDIT_FIELDS, rows)
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unknown dense audit module", result.stderr)

    def test_wrong_rotation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_fixture(root)
            path = root / "runs.tsv"
            rows = list(csv.DictReader(path.open(newline="", encoding="utf-8"), delimiter="\t"))
            rows[0]["arm"] = "source-f16"
            write_table(path, ["repeat", "slot", "arm", "csv", "log", "memory"], rows, "\t")
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("expected exact AB/BA", result.stderr)

    def test_missing_direct_peer_marker_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_fixture(root)
            path = root / "runs/native-q8-r1.log"
            path.write_text(
                path.read_text(encoding="utf-8").replace(
                    "ds4: peer access 1->0 validated across 4 sizes x 4 iterations\n",
                    "",
                ),
                encoding="utf-8",
            )
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("peer access 1->0 validated across", result.stderr)

    def test_absolute_artifact_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_fixture(root)
            path = root / "runs.tsv"
            rows = list(csv.DictReader(path.open(newline="", encoding="utf-8"), delimiter="\t"))
            rows[0]["csv"] = str((root / rows[0]["csv"]).resolve())
            write_table(path, ["repeat", "slot", "arm", "csv", "log", "memory"], rows, "\t")
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not a safe root-relative path", result.stderr)

    def test_tampered_source_provenance_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_fixture(root)
            path = root / "provenance/source-f16-provenance.txt"
            path.write_text(path.read_text(encoding="utf-8") + "tampered=true\n", encoding="utf-8")
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("sidecar hash differs", result.stderr)

    def test_overstated_hf_verified_field_is_rejected_even_with_updated_sidecar_hash(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_fixture(root)
            sidecar = root / "provenance/source-f16-provenance.txt"
            sidecar.write_text(
                sidecar.read_text(encoding="utf-8") + "hf_source_verified=true\n",
                encoding="utf-8",
            )
            manifest = root / "manifest.txt"
            new_sha = hashlib.sha256(sidecar.read_bytes()).hexdigest()
            lines = [
                f"source_f16_provenance_sha256={new_sha}"
                if line.startswith("source_f16_provenance_sha256=")
                else line
                for line in manifest.read_text(encoding="utf-8").splitlines()
            ]
            manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("rejected overstated hf_source_verified", result.stderr)

    def test_nonempty_untracked_source_archive_is_accepted_when_exact(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_fixture(root)
            provenance = root / "provenance"
            (provenance / "untracked-sources.none").unlink()
            (provenance / "untracked-source-files.txt").write_text(
                "new-kernel.cu\n", encoding="utf-8"
            )
            source = root / "new-kernel.cu"
            source.write_text("// evidence\n", encoding="utf-8")
            with tarfile.open(provenance / "untracked-sources.tar.gz", "w:gz") as archive:
                archive.add(source, arcname="new-kernel.cu")
            result = self.run_summary(root)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_untracked_none_must_match_empty_list(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_fixture(root)
            (root / "provenance/untracked-source-files.txt").write_text(
                "new-kernel.cu\n", encoding="utf-8"
            )
            result = self.run_summary(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("contradicts a nonempty file list", result.stderr)


class DecodeABRunnerStaticTests(unittest.TestCase):
    def test_runner_is_primary_fixed_fail_closed_and_avoids_model_hashing(self) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn("REPEATS=${REPEATS:-4}", text)
        self.assertIn("REPEATS >= 4 && REPEATS % 2 == 0", text)
        self.assertIn("arms=(native-q8 source-f16)", text)
        self.assertNotIn("MODEL_Q8_DERIVED", text)
        self.assertNotIn("q8-derived-f16", text)
        self.assertIn("DS4_CUDA_NO_Q8_F16_CACHE=1", text)
        self.assertIn("DS4_CUDA_NO_Q8_F16_PARTNER_OFFLOAD=1", text)
        self.assertIn("CUDA_DEVICE_ORDER=PCI_BUS_ID", text)
        self.assertIn("-v CUDA_VISIBLE_DEVICES", text)
        self.assertIn("-v NVIDIA_VISIBLE_DEVICES", text)
        self.assertIn("require_direct_peer_log", text)
        self.assertNotIn("DS4_CUDA_TP_EP_BALANCED_SHARED_MID=0", text)
        self.assertIn('flock --shared --nonblock 8', text)
        self.assertIn("source_f16_model_inode", text)
        self.assertIn("template_model_inode", text)
        self.assertIn("phase=post-run-model-identity", text)
        self.assertIn('--inventory "$tensor_inventory"', text)
        self.assertIn("git diff --binary HEAD", text)
        self.assertIn("untracked-source-files.txt", text)
        self.assertIn("source_f16_provenance_sha256", text)
        self.assertIn("hf_snapshot_path_revision_match", text)
        self.assertIn("hf_shard_content_authentication", text)
        self.assertIn("rejected overstated hf_source_verified", text)
        self.assertNotIn('sha256sum "$MODEL_Q8"', text)
        self.assertNotIn('sha256sum "$MODEL_SOURCE_F16"', text)
        self.assertIn("contexts=2048,8192,32768", text)
        self.assertIn('(( GEN_TOKENS > 0 ))', text)
        timed = text.split("phase=timed-prefill-decode", 1)[1]
        self.assertNotIn("DS4_CUDA_DENSE_EXEC_AUDIT_CSV=", timed)


if __name__ == "__main__":
    unittest.main()
