#!/usr/bin/env python3
"""Validate the GGUF inputs for the native-Q8/source-F16 A/B experiment.

Only the GGUF header, metadata, and tensor directory are read.  Tensor payloads
are deliberately neither read nor hashed.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARSER_PATH = ROOT / "gguf-tools/mixed/splice_mixed_expert_layers_gguf.py"
LAYERS = 43
FAMILIES: dict[str, tuple[str, tuple[int, int]]] = {
    "attn_q_a": ("attn_q_a.weight", (4096, 1024)),
    "attn_q_b": ("attn_q_b.weight", (1024, 32768)),
    "attn_kv": ("attn_kv.weight", (4096, 512)),
    "attn_output_a": ("attn_output_a.weight", (4096, 8192)),
    "attn_output_b": ("attn_output_b.weight", (8192, 4096)),
    "shared_gate": ("ffn_gate_shexp.weight", (4096, 2048)),
    "shared_up": ("ffn_up_shexp.weight", (4096, 2048)),
    "shared_down": ("ffn_down_shexp.weight", (2048, 4096)),
}
EXPECTED_TENSORS = LAYERS * len(FAMILIES)
Q8_0 = 8
F16 = 1


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def load_parser():
    spec = importlib.util.spec_from_file_location("ds4_gguf_directory", PARSER_PATH)
    if spec is None or spec.loader is None:
        fail(f"cannot load GGUF parser from {PARSER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def target_specs() -> dict[str, tuple[int, str, tuple[int, int]]]:
    return {
        f"blk.{layer}.{suffix}": (layer, family, dims)
        for layer in range(LAYERS)
        for family, (suffix, dims) in FAMILIES.items()
    }


def validate_model(info, arm: str, expected_type: int, parser) -> list[dict[str, object]]:
    specs = target_specs()
    if len(info.tensor_by_name) != info.tensor_count:
        fail(f"{arm} GGUF contains duplicate tensor names")

    missing = sorted(set(specs) - set(info.tensor_by_name))
    if missing:
        fail(
            f"{arm} GGUF is missing {len(missing)} dense tensors; "
            f"first={missing[0]}"
        )

    rows: list[dict[str, object]] = []
    for name, (layer, family, expected_dims) in sorted(specs.items()):
        tensor = info.tensor_by_name[name]
        if tensor.ggml_type != expected_type:
            fail(
                f"{arm} tensor {name} is {parser.qtype_name(tensor.ggml_type)}, "
                f"expected {parser.qtype_name(expected_type)}"
            )
        if tuple(tensor.dims) != expected_dims:
            fail(
                f"{arm} tensor {name} has shape {tuple(tensor.dims)}, "
                f"expected {expected_dims}"
            )
        elements = expected_dims[0] * expected_dims[1]
        rows.append({
            "arm": arm,
            "name": name,
            "layer": layer,
            "module": family,
            "type_id": tensor.ggml_type,
            "type_name": parser.qtype_name(tensor.ggml_type),
            "dim0": tensor.dims[0],
            "dim1": tensor.dims[1],
            "elements": elements,
            "encoded_bytes": tensor.n_bytes,
            "relative_offset": tensor.rel_offset,
            "data_offset": tensor.data_offset,
        })
    if len(rows) != EXPECTED_TENSORS:
        fail(f"{arm} has {len(rows)} target tensors; expected {EXPECTED_TENSORS}")
    return rows


def validate_payload_extents(info, arm: str) -> dict[str, int]:
    """Reject truncated, overlapping, or misaligned tensor payload ranges.

    This is a stat/directory check only: no tensor payload is read or hashed.
    """
    file_bytes = info.path.stat().st_size
    ordered = sorted(info.tensors, key=lambda tensor: tensor.data_offset)
    previous_end = 0
    maximum_end = 0
    for tensor in ordered:
        if tensor.rel_offset % info.alignment:
            fail(
                f"{arm} tensor {tensor.name} has unaligned relative offset "
                f"{tensor.rel_offset} for alignment {info.alignment}"
            )
        start = tensor.data_offset
        end = start + tensor.n_bytes
        if end < start:
            fail(f"{arm} tensor {tensor.name} payload extent overflows")
        if start < previous_end:
            fail(
                f"{arm} tensor {tensor.name} payload overlaps the previous "
                f"tensor ({start} < {previous_end})"
            )
        if end > file_bytes:
            fail(
                f"{arm} tensor {tensor.name} payload is truncated: "
                f"end={end}, file_bytes={file_bytes}"
            )
        previous_end = end
        maximum_end = max(maximum_end, end)
    return {
        "file_bytes": file_bytes,
        "maximum_payload_end": maximum_end,
        "trailing_bytes": file_bytes - maximum_end,
        "payload_extents_valid": 1,
    }


def validate_directory_parity(q8, source) -> dict[str, int]:
    if q8.version != source.version:
        fail(f"GGUF versions differ: {q8.version} vs {source.version}")
    if q8.alignment != source.alignment:
        fail(f"GGUF alignments differ: {q8.alignment} vs {source.alignment}")
    if q8.kv_count != source.kv_count or q8.kv_blob != source.kv_blob:
        fail(
            "GGUF metadata differs; the source-F16 comparison must preserve "
            "the template KV records byte-for-byte"
        )
    q8_names = set(q8.tensor_by_name)
    source_names = set(source.tensor_by_name)
    if q8_names != source_names:
        missing = sorted(q8_names - source_names)
        extra = sorted(source_names - q8_names)
        fail(
            "GGUF tensor-name directories differ outside the intended type change: "
            f"missing={missing[:1]} extra={extra[:1]}"
        )

    target = set(target_specs())
    changed_target = 0
    for name in sorted(q8_names):
        left = q8.tensor_by_name[name]
        right = source.tensor_by_name[name]
        if tuple(left.dims) != tuple(right.dims):
            fail(f"GGUF shape differs for {name}: {left.dims} vs {right.dims}")
        if name in target:
            if left.ggml_type != right.ggml_type:
                changed_target += 1
        elif left.ggml_type != right.ggml_type:
            fail(
                f"non-target tensor type differs for {name}: "
                f"{left.ggml_type} vs {right.ggml_type}"
            )
    if changed_target != EXPECTED_TENSORS:
        fail(
            f"only {changed_target}/{EXPECTED_TENSORS} target tensor types changed"
        )
    return {
        "tensor_count": len(q8_names),
        "non_target_tensor_count": len(q8_names) - EXPECTED_TENSORS,
        "metadata_records": q8.kv_count,
        "metadata_bytes": len(q8.kv_blob),
        "kv_metadata_byte_identical": True,
        "changed_target_types": changed_target,
        "changed_non_target_types": 0,
    }


def write_inventory(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "arm", "name", "layer", "module", "type_id", "type_name",
        "dim0", "dim1", "elements", "encoded_bytes", "relative_offset",
        "data_offset",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Q8 and source-derived-F16 GGUF tensor directories."
    )
    parser.add_argument("--model-q8", required=True, type=Path)
    parser.add_argument("--model-source-f16", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    for label, path in (
        ("MODEL_Q8", args.model_q8),
        ("MODEL_SOURCE_F16", args.model_source_f16),
    ):
        if not path.is_absolute() or not path.is_file():
            fail(f"{label} must name an existing absolute file: {path}")
    if args.model_q8.resolve() == args.model_source_f16.resolve():
        fail("MODEL_Q8 and MODEL_SOURCE_F16 must be different files")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gguf = load_parser()
    try:
        q8 = gguf.parse_gguf(args.model_q8)
        source = gguf.parse_gguf(args.model_source_f16)
    except (OSError, EOFError, ValueError) as exc:
        fail(str(exc))

    q8_rows = validate_model(q8, "native-q8-source", Q8_0, gguf)
    source_rows = validate_model(source, "source-f16", F16, gguf)
    q8_extents = validate_payload_extents(q8, "native-q8-source")
    source_extents = validate_payload_extents(source, "source-f16")
    parity = validate_directory_parity(q8, source)
    write_inventory(args.out_dir / "model-q8.csv", q8_rows)
    write_inventory(args.out_dir / "model-source-f16.csv", source_rows)

    q8_bytes = sum(int(row["encoded_bytes"]) for row in q8_rows)
    source_bytes = sum(int(row["encoded_bytes"]) for row in source_rows)
    elements = sum(int(row["elements"]) for row in q8_rows)
    summary = {
        "structural_integrity": True,
        "payload_hashing": False,
        "model_q8": {
            "path": str(args.model_q8),
            "file_bytes": args.model_q8.stat().st_size,
            "payload_extents": q8_extents,
            "target_tensor_count": len(q8_rows),
            "target_type_inventory": dict(Counter(str(r["type_name"]) for r in q8_rows)),
            "target_encoded_bytes": q8_bytes,
        },
        "model_source_f16": {
            "path": str(args.model_source_f16),
            "file_bytes": args.model_source_f16.stat().st_size,
            "payload_extents": source_extents,
            "target_tensor_count": len(source_rows),
            "target_type_inventory": dict(Counter(str(r["type_name"]) for r in source_rows)),
            "target_encoded_bytes": source_bytes,
        },
        "target_elements": elements,
        "target_file_byte_delta": source_bytes - q8_bytes,
        "directory_parity": parity,
    }
    (args.out_dir / "comparison.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"validated {EXPECTED_TENSORS} dense tensors: "
        f"Q8_0={q8_bytes} bytes, source F16={source_bytes} bytes, "
        f"delta={source_bytes - q8_bytes} bytes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
