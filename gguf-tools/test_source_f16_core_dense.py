#!/usr/bin/env python3
"""Exercise the strict 43x8 source-F16 dry-run selector and manifest."""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


LAYERS = 43
SPECS = (
    ("attn_q_a.weight", 4096, 1024),
    ("attn_q_b.weight", 1024, 32768),
    ("attn_kv.weight", 4096, 512),
    ("attn_output_a.weight", 4096, 8192),
    ("attn_output_b.weight", 8192, 4096),
    ("ffn_gate_shexp.weight", 4096, 2048),
    ("ffn_up_shexp.weight", 4096, 2048),
    ("ffn_down_shexp.weight", 2048, 4096),
)
HF_SUFFIXES = (
    "attn.wq_a.weight",
    "attn.wq_b.weight",
    "attn.wkv.weight",
    "attn.wo_a.weight",
    "attn.wo_b.weight",
    "ffn.shared_experts.w1.weight",
    "ffn.shared_experts.w3.weight",
    "ffn.shared_experts.w2.weight",
)

GGUF_TYPE_UINT32 = 4
GGUF_TYPE_STRING = 8
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q4_K = 12


def u32(value: int) -> bytes:
    return struct.pack("<I", value)


def u64(value: int) -> bytes:
    return struct.pack("<Q", value)


def gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return u64(len(encoded)) + encoded


def kv_u32(key: str, value: int) -> bytes:
    return gguf_string(key) + u32(GGUF_TYPE_UINT32) + u32(value)


def kv_string(key: str, value: str) -> bytes:
    return gguf_string(key) + u32(GGUF_TYPE_STRING) + gguf_string(value)


def core_tensors() -> list[tuple[str, int, int, int]]:
    return [
        (f"blk.{layer}.{suffix}", ne0, ne1, GGML_TYPE_Q8_0)
        for layer in range(LAYERS)
        for suffix, ne0, ne1 in SPECS
    ]


def write_metadata_only_gguf(
    path: Path, tensors: list[tuple[str, int, int, int]]
) -> None:
    kvs = (
        kv_u32("general.alignment", 32),
        kv_u32("deepseek4.expert_count", 256),
        kv_string(
            "ds4.routed_expert.q4.layout", "sm75_m8n8k32_native_aw_v1"
        ),
        kv_u32("ds4.routed_expert.q4.layout_version", 1),
        kv_string("quantize.imatrix.file", "preserve-this-value.dat"),
    )
    body = bytearray(b"GGUF" + u32(3) + u64(len(tensors)) + u64(len(kvs)))
    for record in kvs:
        body += record
    for name, ne0, ne1, tensor_type in tensors:
        body += gguf_string(name)
        body += u32(2)
        body += u64(ne0) + u64(ne1)
        body += u32(tensor_type)
        body += u64(0)
    path.write_bytes(body)


def json_object_from_pairs(pairs: list[tuple[str, object]]) -> bytes:
    records = (
        json.dumps(key, separators=(",", ":"))
        + ":"
        + json.dumps(value, separators=(",", ":"))
        for key, value in pairs
    )
    return ("{" + ",".join(records) + "}").encode("utf-8")


def write_fake_hf(
    path: Path,
    *,
    dtype_override: tuple[str, str] | None = None,
    shape_override: tuple[str, list[int]] | None = None,
    missing_index_name: str | None = None,
    duplicate_index_name: str | None = None,
    duplicate_header_name: str | None = None,
    overlap_name: str | None = None,
    negative_offset_name: str | None = None,
    truncate_bytes: int = 0,
    index_shard_name: str | None = None,
) -> None:
    path.mkdir()
    shard_name = "model-00001-of-00001.safetensors"
    header_pairs: list[tuple[str, object]] = []
    index_pairs: list[tuple[str, object]] = []
    offset = 0
    for layer in range(LAYERS):
        for (_suffix, ne0, ne1), hf_suffix in zip(SPECS, HF_SUFFIXES):
            name = f"layers.{layer}.{hf_suffix}"
            dtype = (
                dtype_override[1]
                if dtype_override is not None and name == dtype_override[0]
                else "BF16"
            )
            shape = (
                shape_override[1]
                if shape_override is not None and name == shape_override[0]
                else [ne1, ne0]
            )
            element_bytes = {"F8_E4M3": 1, "BF16": 2, "F16": 2, "F32": 4}.get(
                dtype, 2
            )
            nbytes = shape[0] * shape[1] * element_bytes
            begin = (
                -1
                if name == negative_offset_name
                else (0 if name == overlap_name else offset)
            )
            descriptor = {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [begin, begin + nbytes],
            }
            header_pairs.append((name, descriptor))
            if name != missing_index_name:
                index_pairs.append((name, index_shard_name or shard_name))
            offset += nbytes
    if duplicate_header_name is not None:
        descriptor = next(
            value for name, value in header_pairs if name == duplicate_header_name
        )
        header_pairs.append((duplicate_header_name, descriptor))
    if duplicate_index_name is not None:
        index_pairs.append((duplicate_index_name, shard_name))
    raw_header = json_object_from_pairs(header_pairs)
    with (path / shard_name).open("wb") as handle:
        handle.write(struct.pack("<Q", len(raw_header)))
        handle.write(raw_header)
        # Header validation checks that every declared payload extent exists,
        # but never reads it. Create one sparse logical extent for the fixture.
        handle.seek(8 + len(raw_header) + offset - 1)
        handle.write(b"\0")
        if truncate_bytes:
            handle.truncate(8 + len(raw_header) + offset - truncate_bytes)
    (path / "model.safetensors.index.json").write_text(
        "{\"weight_map\":"
        + json_object_from_pairs(index_pairs).decode("utf-8")
        + "}",
        encoding="utf-8",
    )


def run_dry(
    binary: Path, template: Path, hf_dir: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(binary),
            "--hf",
            str(hf_dir),
            "--template",
            str(template),
            "--source-f16-core-dense",
            "--dry-run",
        ],
        check=False,
        text=True,
        capture_output=True,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expect_failure(
    binary: Path, template: Path, hf_dir: Path, wanted: str
) -> None:
    result = run_dry(binary, template, hf_dir)
    require(result.returncode != 0, f"expected failure for {template.name}")
    require(
        wanted in result.stderr,
        f"missing error {wanted!r}; stderr was:\n{result.stderr}",
    )


def run_producer_recovery_check(root: Path, template: Path) -> None:
    if not sys.platform.startswith("linux"):
        return
    revision = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
    hf_dir = (
        root
        / "models--deepseek-ai--DeepSeek-V4-Flash-0731"
        / "snapshots"
        / revision
    )
    hf_dir.mkdir(parents=True)
    (hf_dir / "model.safetensors.index.json").write_text(
        '{"weight_map":{}}', encoding="utf-8"
    )
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")

    output = root / "recover-output.gguf"
    plan = Path(f"{output}.source-f16-plan.txt")
    prefix = Path(str(output)[: -len(".gguf")] + ".source-f16")
    inventory = Path(f"{prefix}.inventory.json")
    template_inventory = Path(f"{prefix}.template-inventory.csv")
    model_inventory = Path(f"{prefix}.model-inventory.csv")
    provenance = Path(f"{prefix}.provenance.txt")
    finals = (
        output,
        plan,
        inventory,
        template_inventory,
        model_inventory,
        provenance,
    )
    for number, path in enumerate(finals):
        path.write_text(f"new-{number}\n", encoding="utf-8")

    backup = Path(f"{output}.publish-backup")
    backup.mkdir()
    (backup / "originals.list").write_text("0\n1\n5\n", encoding="utf-8")
    (backup / "0").write_text("old-model\n", encoding="utf-8")
    (backup / "1").write_text("old-plan\n", encoding="utf-8")
    (backup / "5").write_text("old-provenance\n", encoding="utf-8")
    committed = Path(f"{output}.publish-committed-backup")
    committed.mkdir()
    (committed / "stale").write_text("stale\n", encoding="utf-8")
    Path(f"{output}.partial").write_text("partial\n", encoding="utf-8")
    Path(f"{output}.partial.1234").write_text("legacy\n", encoding="utf-8")
    validation = Path(f"{output}.validation.partial")
    validation.mkdir()
    (validation / "stale").write_text("stale\n", encoding="utf-8")

    producer = Path(__file__).resolve().parent.parent / "produce-source-f16-core-dense.sh"
    environment = os.environ.copy()
    environment.update({"RECOVER_ONLY": "1", "OVERWRITE": "1"})
    result = subprocess.run(
        ["bash", str(producer), str(hf_dir), str(template), str(output)],
        check=False,
        text=True,
        capture_output=True,
        env=environment,
    )
    require(result.returncode == 0, result.stdout + result.stderr)
    require(output.read_text(encoding="utf-8") == "old-model\n", "model rollback failed")
    require(plan.read_text(encoding="utf-8") == "old-plan\n", "plan rollback failed")
    require(
        provenance.read_text(encoding="utf-8") == "old-provenance\n",
        "provenance rollback failed",
    )
    for path in (inventory, template_inventory, model_inventory):
        require(not path.exists(), f"new-only artifact survived rollback: {path}")
    for path in (
        backup,
        committed,
        Path(f"{output}.partial"),
        Path(f"{output}.partial.1234"),
        validation,
    ):
        require(not path.exists(), f"stale transaction path survived recovery: {path}")


def main() -> int:
    binary = Path(
        sys.argv[1] if len(sys.argv) > 1 else "./deepseek4-quantize"
    ).resolve()
    require(binary.is_file(), f"quantizer binary not found: {binary}")

    tensors = core_tensors()
    require(len(tensors) == 344, "test inventory is not exactly 43x8")
    tensors.append(("blk.0.ffn_gate_exps.weight", 256, 8, GGML_TYPE_Q4_K))

    with tempfile.TemporaryDirectory(prefix="ds4-source-f16-test-") as td:
        root = Path(td)
        hf_dir = root / "hf"
        write_fake_hf(hf_dir)
        valid = root / "valid.gguf"
        write_metadata_only_gguf(valid, tensors)
        run_producer_recovery_check(root / "recovery", valid)
        result = run_dry(binary, valid, hf_dir)
        require(result.returncode == 0, result.stderr)
        manifest = [
            line
            for line in result.stdout.splitlines()
            if line.startswith("source_f16_tensor: ")
        ]
        require(len(manifest) == 344, f"manifest has {len(manifest)} tensors")
        require(
            manifest[0].startswith("source_f16_tensor: blk.0.attn_q_a.weight "),
            "manifest first tensor changed",
        )
        require(
            manifest[-1].startswith(
                "source_f16_tensor: blk.42.ffn_down_shexp.weight "
            ),
            "manifest last tensor changed",
        )
        require("source_f16_selected: 344" in result.stdout, "bad selected count")
        require(
            "source_f16_hf_headers_validated: 344" in result.stdout,
            "HF header preflight did not validate all 344 tensors",
        )
        require("type_changes: 344" in result.stdout, "bad type-change count")
        require(
            "source_f16_selected_bytes: 11362369536" in result.stdout,
            "unexpected selected F16 byte plan",
        )
        require(
            "source_f16_unselected_payloads: verbatim" in result.stdout,
            "copy-through guarantee missing",
        )
        require(
            "source_f16_template_kvs: verbatim" in result.stdout,
            "KV preservation guarantee missing",
        )

        missing = root / "missing.gguf"
        write_metadata_only_gguf(missing, tensors[:-2] + tensors[-1:])
        expect_failure(
            binary, missing, hf_dir,
            "missing required tensor: blk.42.ffn_down_shexp.weight"
        )

        wrong_shape = root / "wrong-shape.gguf"
        malformed = list(tensors)
        name, ne0, ne1, tensor_type = malformed[0]
        malformed[0] = (name, ne0, ne1 + 1, tensor_type)
        write_metadata_only_gguf(wrong_shape, malformed)
        expect_failure(binary, wrong_shape, hf_dir, "has shape [4096, 1025]")

        duplicate = root / "duplicate.gguf"
        write_metadata_only_gguf(duplicate, tensors + [tensors[0]])
        expect_failure(binary, duplicate, hf_dir, "duplicate GGUF tensor name")

        extra_layer = root / "extra-layer.gguf"
        write_metadata_only_gguf(
            extra_layer,
            tensors
            + [("blk.43.attn_q_a.weight", 4096, 1024, GGML_TYPE_F16)],
        )
        expect_failure(binary, extra_layer, hf_dir, "outside Flash layers 0..42")

        first_hf_name = f"layers.0.{HF_SUFFIXES[0]}"
        second_hf_name = f"layers.0.{HF_SUFFIXES[1]}"

        missing_hf = root / "hf-missing-index"
        write_fake_hf(missing_hf, missing_index_name=first_hf_name)
        expect_failure(binary, valid, missing_hf, "HF tensor not found")

        bad_dtype_hf = root / "hf-bad-dtype"
        write_fake_hf(
            bad_dtype_hf, dtype_override=(first_hf_name, "F32")
        )
        expect_failure(binary, valid, bad_dtype_hf, "unsupported HF dtype F32")

        bad_shape_hf = root / "hf-bad-shape"
        write_fake_hf(
            bad_shape_hf,
            shape_override=(first_hf_name, [SPECS[0][2] + 1, SPECS[0][1]]),
        )
        expect_failure(binary, valid, bad_shape_hf, "shape mismatch")

        truncated_hf = root / "hf-truncated"
        write_fake_hf(truncated_hf, truncate_bytes=1)
        expect_failure(binary, valid, truncated_hf, "is truncated")

        duplicate_index_hf = root / "hf-duplicate-index"
        write_fake_hf(
            duplicate_index_hf, duplicate_index_name=first_hf_name
        )
        expect_failure(
            binary, valid, duplicate_index_hf,
            "duplicate safetensors index weight name",
        )

        unsafe_shard_hf = root / "hf-unsafe-shard"
        write_fake_hf(unsafe_shard_hf, index_shard_name="../outside.safetensors")
        expect_failure(
            binary, valid, unsafe_shard_hf,
            "unsafe safetensors shard path in index",
        )

        duplicate_header_hf = root / "hf-duplicate-header"
        write_fake_hf(
            duplicate_header_hf, duplicate_header_name=first_hf_name
        )
        expect_failure(
            binary, valid, duplicate_header_hf,
            "duplicate safetensors shard tensor name",
        )

        overlap_hf = root / "hf-overlap"
        write_fake_hf(overlap_hf, overlap_name=second_hf_name)
        expect_failure(binary, valid, overlap_hf, "payloads overlap")

        negative_offset_hf = root / "hf-negative-offset"
        write_fake_hf(
            negative_offset_hf, negative_offset_name=first_hf_name
        )
        expect_failure(
            binary, valid, negative_offset_hf,
            "negative safetensors data offset",
        )

        # The dry-run must prove every declared safetensors payload extent is
        # present before the producer allocates a 100+ GiB output.  Removing
        # one byte truncates only the final selected tensor.
        shard = hf_dir / "model-00001-of-00001.safetensors"
        with shard.open("r+b") as handle:
            handle.truncate(shard.stat().st_size - 1)
        expect_failure(
            binary,
            valid,
            hf_dir,
            "layers.42.ffn.shared_experts.w2.weight is truncated",
        )

    print("source-F16 core-dense selector tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
