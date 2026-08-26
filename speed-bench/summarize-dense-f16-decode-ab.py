#!/usr/bin/env python3
"""Validate native-Q8 versus source-derived-F16 production evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import tarfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


CASES = 100
LAYERS = 43
ARMS = ("native-q8", "source-f16")
EXPECTED_RESULTS = {"native-q8": "native_q8", "source-f16": "source_f16"}
NATIVE_Q8_BACKENDS = {"sm75_native_q8", "sm75_native_q8_fused_shared_mid"}
SOURCE_F16_BACKEND = "cublas_gemm_ex"
MODULES: dict[str, tuple[str, tuple[int, int]]] = {
    "attn_q_a": ("attn_q_a.weight", (4096, 1024)),
    "attn_q_b": ("attn_q_b.weight", (1024, 32768)),
    "attn_kv": ("attn_kv.weight", (4096, 512)),
    "attn_output_a": ("attn_output_a.weight", (4096, 8192)),
    "attn_output_b": ("attn_output_b.weight", (8192, 4096)),
    "shared_gate": ("ffn_gate_shexp.weight", (4096, 2048)),
    "shared_up": ("ffn_up_shexp.weight", (4096, 2048)),
    "shared_down": ("ffn_down_shexp.weight", (2048, 4096)),
}
EXPECTED_KEYS = {(layer, module) for layer in range(LAYERS) for module in MODULES}
AUDIT_COLUMNS = {
    "sequence", "module", "label", "layer", "token_offset",
    "physical_device", "weight_offset", "weight_bytes", "weight_type",
    "in_dim", "out_dim", "executed_input_offset", "executed_input_count",
    "executed_output_offset", "executed_output_count", "n_tokens", "backend",
    "placement", "result", "reason",
}
INVENTORY_COLUMNS = {
    "arm", "name", "layer", "module", "type_id", "type_name", "dim0",
    "dim1", "elements", "encoded_bytes", "relative_offset", "data_offset",
}
SCORE_COLUMNS = {
    "id", "target_tokens", "nll", "avg_nll", "first_match", "greedy_lcp",
    "api_target_tokens", "api_target_mae", "api_top1_count",
    "api_top1_match", "api_topn_ref", "api_topn_hit", "api_pair_total",
    "api_pair_agree",
}
BENCH_COLUMNS = {
    "ctx_tokens", "prefill_tokens", "prefill_tps", "gen_tokens", "gen_tps",
    "gen_first_ms", "gen_steady_tokens", "gen_steady_tps", "kvcache_bytes",
}
DIRECT_PEER_MARKERS = tuple(
    f"peer access {edge} validated across"
    for edge in ("0->1", "1->0", "2->3", "3->2")
)


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def read_rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if reader.fieldnames is None:
                fail(f"{path} has no header")
            return list(reader)
    except OSError as exc:
        fail(f"cannot read {path}: {exc}")


def read_header(path: Path, delimiter: str = ",") -> set[str]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return set(next(csv.reader(handle, delimiter=delimiter)))
    except (OSError, StopIteration) as exc:
        fail(f"cannot inspect {path}: {exc}")


def require_columns(
    path: Path, rows: list[dict[str, str]], required: set[str], delimiter: str = ","
) -> None:
    fields = set(rows[0]) if rows else read_header(path, delimiter)
    missing = required - fields
    if missing:
        fail(f"{path} lacks columns: {','.join(sorted(missing))}")


def read_key_values(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        for number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key in result:
                fail(f"{path}:{number} duplicates key {key}")
            result[key] = value
    except OSError as exc:
        fail(f"cannot read {path}: {exc}")
    return result


def as_int(row: dict[str, str], key: str, context: str, minimum: int = 0) -> int:
    try:
        value = int(row[key])
    except (KeyError, ValueError) as exc:
        fail(f"{context} has invalid {key}: {row.get(key, '<missing>')}")
        raise AssertionError from exc
    if value < minimum:
        fail(f"{context} has {key}={value}, expected at least {minimum}")
    return value


def type_matches(value: str, arm: str) -> bool:
    normalized = value.strip().lower().replace("-", "_")
    if arm == "source-f16":
        return normalized in {"1", "f16", "float16", "fp16"}
    return normalized in {"8", "q8", "q8_0"}


def read_dense_inventory(
    path: Path, arm: str
) -> dict[tuple[int, str], dict[str, int | str]]:
    if arm not in ARMS:
        fail(f"unknown arm: {arm}")
    rows = read_rows(path)
    require_columns(path, rows, INVENTORY_COLUMNS)
    expected_arm = "native-q8-source" if arm == "native-q8" else "source-f16"
    result: dict[tuple[int, str], dict[str, int | str]] = {}
    for row_number, row in enumerate(rows, 2):
        context = f"{path}:{row_number}"
        module = row.get("module", "")
        if module not in MODULES:
            fail(f"{context} has unknown dense inventory module {module!r}")
        layer = as_int(row, "layer", context)
        if layer >= LAYERS:
            fail(f"{context} has out-of-range layer {layer}")
        suffix, expected_dims = MODULES[module]
        expected_name = f"blk.{layer}.{suffix}"
        if row.get("arm") != expected_arm or row.get("name") != expected_name:
            fail(
                f"{context} does not identify the expected {expected_arm} "
                f"tensor {expected_name}"
            )
        dim0 = as_int(row, "dim0", context, 1)
        dim1 = as_int(row, "dim1", context, 1)
        if (dim0, dim1) != expected_dims:
            fail(
                f"{context} has shape={dim0}x{dim1}; expected "
                f"{expected_dims[0]}x{expected_dims[1]}"
            )
        type_id = as_int(row, "type_id", context)
        if not type_matches(str(type_id), arm) or not type_matches(
            row.get("type_name", ""), arm
        ):
            fail(f"{context} has an unexpected stored weight type")
        elements = as_int(row, "elements", context, 1)
        if elements != dim0 * dim1:
            fail(f"{context} has an inconsistent element count")
        encoded_bytes = as_int(row, "encoded_bytes", context, 1)
        data_offset = as_int(row, "data_offset", context)
        as_int(row, "relative_offset", context)
        key = (layer, module)
        if key in result:
            fail(f"{path} duplicates inventory tensor layer={layer} module={module}")
        result[key] = {
            "name": expected_name,
            "type_id": type_id,
            "dim0": dim0,
            "dim1": dim1,
            "encoded_bytes": encoded_bytes,
            "data_offset": data_offset,
        }
    if set(result) != EXPECTED_KEYS:
        missing = sorted(EXPECTED_KEYS - set(result))
        extra = sorted(set(result) - EXPECTED_KEYS)
        fail(
            f"{arm} tensor inventory differs: missing={missing[:3]} "
            f"extra={extra[:3]}"
        )
    return result


def rectangles_cover_tensor(
    rectangles: list[tuple[int, int, int, int]], in_dim: int, out_dim: int
) -> tuple[bool, tuple[int, int] | None, int]:
    """Return whether the union of executed [input]x[output] rectangles is full.

    Repeated quality cases legitimately emit duplicate and overlapping records,
    so coverage is a union rather than a disjointness assertion. Coordinate
    compression keeps this exact without constructing a multi-billion-cell
    matrix.
    """
    unique = sorted(set(rectangles))
    if (0, in_dim, 0, out_dim) in unique:
        return True, None, len(unique)
    input_edges = {0, in_dim}
    output_edges = {0, out_dim}
    for input_offset, input_count, output_offset, output_count in unique:
        input_edges.update((input_offset, input_offset + input_count))
        output_edges.update((output_offset, output_offset + output_count))
    inputs = sorted(input_edges)
    outputs = sorted(output_edges)
    for input_index in range(len(inputs) - 1):
        input_start, input_end = inputs[input_index : input_index + 2]
        if input_start == input_end:
            continue
        for output_index in range(len(outputs) - 1):
            output_start, output_end = outputs[output_index : output_index + 2]
            if output_start == output_end:
                continue
            if not any(
                rectangle_input <= input_start
                and input_end <= rectangle_input + rectangle_input_count
                and rectangle_output <= output_start
                and output_end <= rectangle_output + rectangle_output_count
                for (
                    rectangle_input,
                    rectangle_input_count,
                    rectangle_output,
                    rectangle_output_count,
                ) in unique
            ):
                return False, (input_start, output_start), len(unique)
    return True, None, len(unique)


def validate_audit(
    path: Path,
    arm: str,
    inventory: dict[tuple[int, str], dict[str, int | str]],
) -> dict[str, object]:
    if arm not in ARMS:
        fail(f"unknown arm: {arm}")
    rows = read_rows(path)
    require_columns(path, rows, AUDIT_COLUMNS)
    if not rows:
        fail(f"{path} contains no dense execution records")

    expected_result = EXPECTED_RESULTS[arm]
    all_keys: set[tuple[int, str]] = set()
    decode_rectangles: defaultdict[
        tuple[int, str], list[tuple[int, int, int, int]]
    ] = defaultdict(list)
    fused_decode_rectangles: defaultdict[
        tuple[int, str], list[tuple[int, int, int, int]]
    ] = defaultdict(list)
    result_counts: Counter[str] = Counter()
    backend_counts: Counter[str] = Counter()
    placement_counts: Counter[str] = Counter()
    decode_counts: Counter[str] = Counter()
    devices: Counter[int] = Counter()
    shapes: Counter[str] = Counter()
    for row_number, row in enumerate(rows, 2):
        module = row.get("module", "")
        if module not in MODULES:
            fail(f"{path}:{row_number} has unknown dense audit module {module!r}")
        context = f"{path}:{row_number} ({module})"
        layer = as_int(row, "layer", context)
        if layer >= LAYERS:
            fail(f"{context} has out-of-range layer {layer}")
        suffix, expected_dims = MODULES[module]
        expected_label = f"blk.{layer}.{suffix}"
        if row.get("label") != expected_label:
            fail(
                f"{context} has label={row.get('label')!r}; "
                f"expected exactly {expected_label!r}"
            )
        key = (layer, module)
        all_keys.add(key)
        if not type_matches(row.get("weight_type", ""), arm):
            fail(f"{context} has unexpected stored weight type {row.get('weight_type')!r}")
        if row.get("result") != expected_result:
            fail(f"{context} has result={row.get('result')!r}; expected {expected_result}")
        backend = row.get("backend", "")
        placement = row.get("placement", "")
        if arm == "native-q8":
            if backend not in NATIVE_Q8_BACKENDS:
                fail(
                    f"{context} has backend={backend!r}; expected one of "
                    f"{sorted(NATIVE_Q8_BACKENDS)}"
                )
            if backend == "sm75_native_q8_fused_shared_mid":
                if module not in {"shared_gate", "shared_up"}:
                    fail(f"{context} uses fused shared-mid proof for the wrong module")
                if placement not in {"local", "partner"}:
                    fail(f"{context} has invalid fused placement={placement!r}")
            elif placement != "local":
                fail(f"{context} generic native Q8 has placement={placement!r}; expected 'local'")
        else:
            if backend != SOURCE_F16_BACKEND:
                fail(f"{context} has backend={backend!r}; expected {SOURCE_F16_BACKEND}")
            if placement != "local":
                fail(f"{context} source F16 has placement={placement!r}; expected 'local'")
        if row.get("reason") != "executed":
            fail(f"{context} has reason={row.get('reason')!r}; expected 'executed'")
        as_int(row, "sequence", context)
        as_int(row, "token_offset", context)
        weight_offset = as_int(row, "weight_offset", context)
        weight_bytes = as_int(row, "weight_bytes", context, 1)
        in_dim = as_int(row, "in_dim", context, 1)
        out_dim = as_int(row, "out_dim", context, 1)
        if (in_dim, out_dim) != expected_dims:
            fail(
                f"{context} has shape={in_dim}x{out_dim}; "
                f"expected {expected_dims[0]}x{expected_dims[1]}"
            )
        expected_payload = inventory.get(key)
        if expected_payload is None:
            fail(f"{context} has no matching tensor-inventory record")
        if (
            weight_offset != expected_payload["data_offset"]
            or weight_bytes != expected_payload["encoded_bytes"]
        ):
            fail(
                f"{context} payload identity is "
                f"offset={weight_offset} bytes={weight_bytes}; expected "
                f"offset={expected_payload['data_offset']} "
                f"bytes={expected_payload['encoded_bytes']}"
            )
        executed_input_offset = as_int(row, "executed_input_offset", context)
        executed_input_count = as_int(row, "executed_input_count", context, 1)
        executed_output_offset = as_int(row, "executed_output_offset", context)
        executed_output_count = as_int(row, "executed_output_count", context, 1)
        if (
            executed_input_offset > in_dim
            or executed_input_count > in_dim - executed_input_offset
            or executed_output_offset > out_dim
            or executed_output_count > out_dim - executed_output_offset
        ):
            fail(
                f"{context} execution rectangle "
                f"input={executed_input_offset}+{executed_input_count}/"
                f"{in_dim} output={executed_output_offset}+"
                f"{executed_output_count}/{out_dim} is out of bounds"
            )
        execution_rectangle = (
            executed_input_offset,
            executed_input_count,
            executed_output_offset,
            executed_output_count,
        )
        n_tokens = as_int(row, "n_tokens", context, 1)
        device = as_int(row, "physical_device", context)
        if device not in {0, 1, 2, 3}:
            fail(f"{context} has unexpected physical device {device}")
        if n_tokens == 1:
            decode_rectangles[key].append(execution_rectangle)
            decode_counts[module] += 1
            if backend == "sm75_native_q8_fused_shared_mid":
                fused_decode_rectangles[key].append(execution_rectangle)
        result_counts[expected_result] += 1
        backend_counts[backend] += 1
        placement_counts[placement] += 1
        devices[device] += 1
        shapes[f"{module}:{in_dim}x{out_dim}"] += 1

    if all_keys != EXPECTED_KEYS:
        missing = sorted(EXPECTED_KEYS - all_keys)
        extra = sorted(all_keys - EXPECTED_KEYS)
        fail(f"{arm} audit target coverage differs: missing={missing[:3]} extra={extra[:3]}")
    complete_decode_keys: set[tuple[int, str]] = set()
    unique_decode_rectangles = 0
    for key in sorted(EXPECTED_KEYS):
        expected_dims = MODULES[key[1]][1]
        covered, first_gap, unique_count = rectangles_cover_tensor(
            decode_rectangles[key], *expected_dims
        )
        unique_decode_rectangles += unique_count
        if not covered:
            fail(
                f"{arm} audit lacks complete one-token execution-rectangle "
                f"coverage for layer={key[0]} module={key[1]}; "
                f"first_gap={first_gap} rectangles={unique_count}"
            )
        complete_decode_keys.add(key)
    complete_fused_keys: set[tuple[int, str]] = set()
    if arm == "native-q8":
        expected_fused = {
            (layer, module)
            for layer in range(LAYERS)
            for module in ("shared_gate", "shared_up")
        }
        for key in sorted(expected_fused):
            covered, _, _ = rectangles_cover_tensor(
                fused_decode_rectangles[key], *MODULES[key[1]][1]
            )
            if covered:
                complete_fused_keys.add(key)
        if complete_fused_keys != expected_fused:
            missing = sorted(expected_fused - complete_fused_keys)
            extra = sorted(complete_fused_keys - expected_fused)
            fail(
                "native-q8 audit fused shared-mid decode rectangle coverage differs: "
                f"missing={missing[:3]} extra={extra[:3]}"
            )
    return {
        "records": sum(result_counts.values()),
        "distinct_target_tensors": len(all_keys),
        "distinct_decode_tensors": len(complete_decode_keys),
        "unique_decode_execution_rectangles": unique_decode_rectangles,
        "results": dict(result_counts),
        "backends": dict(backend_counts),
        "placements": dict(placement_counts),
        "physical_devices": {str(key): value for key, value in sorted(devices.items())},
        "decode_records_by_module": dict(sorted(decode_counts.items())),
        "fused_shared_mid_decode_tensors": len(complete_fused_keys),
        "executed_shapes": dict(sorted(shapes.items())),
        "fallback_records": 0,
    }


def read_scores(path: Path) -> dict[str, dict[str, str]]:
    rows = read_rows(path, "\t")
    require_columns(path, rows, SCORE_COLUMNS, "\t")
    result = {row["id"]: row for row in rows}
    if len(rows) != CASES or len(result) != CASES:
        fail(f"{path} must contain exactly {CASES} unique quality cases")
    for case_id, row in result.items():
        target_tokens = as_int(row, "target_tokens", f"{path}:{case_id}", 1)
        for field in ("nll", "avg_nll", "api_target_mae"):
            try:
                value = float(row[field])
            except (KeyError, ValueError) as exc:
                fail(f"{path}:{case_id} has invalid {field}")
                raise AssertionError from exc
            if not math.isfinite(value):
                fail(f"{path}:{case_id} has non-finite {field}")
        if not math.isclose(
            float(row["nll"]) / target_tokens,
            float(row["avg_nll"]),
            rel_tol=1e-6,
            abs_tol=1e-7,
        ):
            fail(f"{path}:{case_id} has inconsistent nll/avg_nll")
        for numerator, denominator in (
            ("api_top1_match", "api_top1_count"),
            ("api_topn_hit", "api_topn_ref"),
            ("api_pair_agree", "api_pair_total"),
        ):
            num = as_int(row, numerator, f"{path}:{case_id}")
            den = as_int(row, denominator, f"{path}:{case_id}")
            if num > den:
                fail(f"{path}:{case_id} has {numerator}>{denominator}")
    return result


def isum(rows: dict[str, dict[str, str]], key: str) -> int:
    return sum(int(float(row.get(key, "0") or 0)) for row in rows.values())


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def weighted(
    rows: dict[str, dict[str, str]], key: str, count: str
) -> float | None:
    denominator = isum(rows, count)
    if not denominator:
        return None
    return (
        sum(float(row[key]) * int(float(row[count])) for row in rows.values())
        / denominator
    )


def quality_summary(rows: dict[str, dict[str, str]]) -> dict[str, object]:
    tokens = sum(int(row["target_tokens"]) for row in rows.values())
    first = isum(rows, "first_match")
    return {
        "cases": len(rows),
        "target_tokens": tokens,
        "avg_nll": sum(float(row["nll"]) for row in rows.values()) / tokens,
        "first_matches": first,
        "first_match_rate": first / len(rows),
        "avg_greedy_lcp": statistics.fmean(
            float(row["greedy_lcp"]) for row in rows.values()
        ),
        "api_target_mae": weighted(rows, "api_target_mae", "api_target_tokens"),
        "api_top1_rate": ratio(isum(rows, "api_top1_match"), isum(rows, "api_top1_count")),
        "api_topn_recall": ratio(isum(rows, "api_topn_hit"), isum(rows, "api_topn_ref")),
        "api_pair_rate": ratio(isum(rows, "api_pair_agree"), isum(rows, "api_pair_total")),
    }


def bootstrap_nll_delta(
    ids: list[str], native: dict[str, dict[str, str]],
    candidate: dict[str, dict[str, str]], draws: int = 10000,
) -> tuple[float, float, float]:
    rng = random.Random(0x75_F16)
    samples: list[float] = []
    for _ in range(draws):
        delta = 0.0
        tokens = 0
        for _case in ids:
            case_id = ids[rng.randrange(len(ids))]
            delta += float(candidate[case_id]["nll"]) - float(native[case_id]["nll"])
            tokens += int(native[case_id]["target_tokens"])
        samples.append(delta / tokens)
    samples.sort()
    return (
        samples[math.floor(0.025 * draws)],
        samples[math.ceil(0.975 * draws) - 1],
        samples[math.ceil(0.95 * draws) - 1],
    )


def require_direct_peer_markers(path: Path, text: str) -> None:
    for marker in DIRECT_PEER_MARKERS:
        if marker not in text:
            fail(f"{path} lacks runtime marker: {marker}")
    if re.search(r"peer access .* FAILED validation", text):
        fail(f"{path} reports a failed peer validation")


def validate_quality_log(path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        fail(f"cannot read {path}: {exc}")
    for marker in (
        "score_official: runtime_path=production",
        "CUDA EP forced pipeline split 22/21",
    ):
        if marker not in text:
            fail(f"{path} lacks runtime marker: {marker}")
    require_direct_peer_markers(path, text)


def read_bench(path: Path, gen_tokens: int) -> dict[int, dict[str, float]]:
    rows = read_rows(path)
    require_columns(path, rows, BENCH_COLUMNS)
    contexts: dict[int, dict[str, float]] = {}
    for row in rows:
        ctx = as_int(row, "ctx_tokens", str(path), 1)
        if ctx in contexts:
            fail(f"{path} repeats context {ctx}")
        if as_int(row, "gen_tokens", f"{path}:{ctx}") != gen_tokens:
            fail(f"{path}:{ctx} did not generate exactly {gen_tokens} tokens")
        values = {
            "prefill_tps": float(row["prefill_tps"]),
            "gen_tps": float(row["gen_tps"]),
            "gen_first_ms": float(row["gen_first_ms"]),
            "gen_steady_tps": float(row["gen_steady_tps"]),
        }
        if any(
            not math.isfinite(value) or value <= 0
            for value in (values["prefill_tps"], values["gen_tps"], values["gen_first_ms"])
        ):
            fail(f"{path}:{ctx} contains a non-positive/non-finite timing")
        steady_tokens = as_int(row, "gen_steady_tokens", f"{path}:{ctx}")
        if steady_tokens != max(gen_tokens - 1, 0):
            fail(f"{path}:{ctx} has unexpected gen_steady_tokens={steady_tokens}")
        if not math.isfinite(values["gen_steady_tps"]) or (
            steady_tokens > 0 and values["gen_steady_tps"] <= 0
        ):
            fail(f"{path}:{ctx} contains an invalid steady decode timing")
        contexts[ctx] = values
    if set(contexts) != {2048, 8192, 32768}:
        fail(f"{path} contexts are {sorted(contexts)}, expected [2048, 8192, 32768]")
    return contexts


def read_memory(path: Path) -> dict[str, object]:
    rows = read_rows(path)
    if not rows:
        fail(f"memory provenance contains no samples: {path}")
    normalized_fields = {field.split(" [", 1)[0].strip() for field in rows[0]}
    required = {"timestamp", "index", "pci.bus_id", "memory.total", "memory.used", "memory.free"}
    missing = required - normalized_fields
    if missing:
        fail(f"{path} lacks memory fields: {','.join(sorted(missing))}")
    by_device: dict[str, dict[str, object]] = {}
    for row_number, row in enumerate(rows, 2):
        normalized = {
            key.split(" [", 1)[0].strip(): value.strip() for key, value in row.items()
        }
        device = normalized["index"]
        try:
            total = int(normalized["memory.total"])
            used = int(normalized["memory.used"])
            free = int(normalized["memory.free"])
        except ValueError as exc:
            fail(f"{path}:{row_number} has non-integer memory values")
            raise AssertionError from exc
        if total <= 0 or used < 0 or free < 0:
            fail(f"{path}:{row_number} has invalid memory values")
        item = by_device.setdefault(
            device,
            {
                "pci_bus_id": normalized["pci.bus_id"],
                "total_mib": total,
                "peak_used_mib": used,
                "minimum_free_mib": free,
                "samples": 0,
            },
        )
        if item["pci_bus_id"] != normalized["pci.bus_id"] or item["total_mib"] != total:
            fail(f"{path}:{row_number} changes device identity/capacity for GPU {device}")
        item["peak_used_mib"] = max(int(item["peak_used_mib"]), used)
        item["minimum_free_mib"] = min(int(item["minimum_free_mib"]), free)
        item["samples"] = int(item["samples"]) + 1
    if set(by_device) != {"0", "1", "2", "3"}:
        fail(f"{path} memory samples cover GPUs {sorted(by_device)}, expected 0,1,2,3")
    return {"samples": len(rows), "devices": by_device}


def root_relative(root: Path, value: str, context: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        fail(f"{context} is not a safe root-relative path: {value}")
    root_resolved = root.resolve()
    candidate = (root_resolved / relative).resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        fail(f"{context} escapes the evidence root: {value}")
    return candidate


def validate_provenance(root: Path, manifest: dict[str, str]) -> dict[str, object]:
    sidecar_path = root_relative(root, manifest["source_f16_provenance"], "source provenance")
    if not sidecar_path.is_file():
        fail(f"source provenance is missing: {sidecar_path}")
    actual_sha = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    if actual_sha != manifest.get("source_f16_provenance_sha256"):
        fail("source provenance sidecar hash differs from the manifest")
    sidecar_lines = sidecar_path.read_text(encoding="utf-8", errors="strict").splitlines()
    if not sidecar_lines or any(
        re.fullmatch(r"[A-Za-z0-9_]+=.*", line) is None for line in sidecar_lines
    ):
        fail("source provenance contains a malformed record")
    sidecar = read_key_values(sidecar_path)
    expected = {
        "hf_repository": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "hf_revision": "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
        "hf_snapshot_path_revision_match": "true",
        "hf_shard_content_authentication": "not_performed",
        "source_f16_selected": "344",
        "template_q8_selected": "344",
        "kv_metadata_byte_identical": "true",
        "payload_extents_valid": "true",
        "non_target_payload_copy": "producer-guaranteed-not-independently-rehashed",
        "model_payload_hashing": "disabled",
        "publication_protocol": "model-last-rollback-journal-v1",
        "provenance_complete": "true",
    }
    wrong = {key: (sidecar.get(key), value) for key, value in expected.items() if sidecar.get(key) != value}
    if wrong:
        fail(f"source provenance does not satisfy the fixed experiment: {wrong}")
    if "hf_source_verified" in sidecar:
        fail("source provenance uses rejected overstated hf_source_verified")
    for key, length in (
        ("git_commit", 40),
        ("hf_index_sha256", 64),
        ("hf_config_sha256", 64),
        ("quantizer_sha256", 64),
    ):
        if not re.fullmatch(rf"[0-9a-f]{{{length}}}", sidecar.get(key, "")):
            fail(f"source provenance has invalid {key}")
    small_file_bytes: dict[str, int] = {}
    for key in ("quantizer_bytes", "hf_index_bytes", "hf_config_bytes"):
        try:
            small_file_bytes[key] = int(sidecar[key])
        except (KeyError, ValueError) as exc:
            fail(f"source provenance has invalid {key}")
            raise AssertionError from exc
        if small_file_bytes[key] <= 0:
            fail(f"source provenance has non-positive {key}")
    input_identities: dict[str, dict[str, int]] = {}
    for prefix in ("quantizer", "hf_index", "hf_config"):
        values: dict[str, int] = {}
        for suffix in ("device", "inode", "mtime_ns"):
            key = f"{prefix}_{suffix}"
            try:
                values[suffix] = int(sidecar[key])
            except (KeyError, ValueError) as exc:
                fail(f"source provenance has invalid {key}")
                raise AssertionError from exc
        if values["device"] < 0 or values["inode"] <= 0 or values["mtime_ns"] <= 0:
            fail(f"source provenance has invalid {prefix} identity")
        if prefix != "quantizer":
            try:
                values["mtime_epoch"] = int(sidecar[f"{prefix}_mtime_epoch"])
            except (KeyError, ValueError) as exc:
                fail(f"source provenance has invalid {prefix}_mtime_epoch")
                raise AssertionError from exc
            if values["mtime_ns"] // 1_000_000_000 != values["mtime_epoch"]:
                fail(f"source provenance has inconsistent {prefix} mtime")
        input_identities[prefix] = values
    if sidecar.get("git_dirty") not in {"true", "false"}:
        fail("source provenance has invalid git_dirty")
    for key in ("quantizer", "hf_directory", "hf_index", "hf_config"):
        if not sidecar.get(key):
            fail(f"source provenance has empty/missing {key}")
    file_identities: dict[str, dict[str, int]] = {}
    for prefix in ("template_model", "source_f16_model"):
        values: dict[str, int] = {}
        for suffix in ("device", "inode", "mtime_epoch", "mtime_ns"):
            key = f"{prefix}_{suffix}"
            try:
                values[suffix] = int(sidecar[key])
            except (KeyError, ValueError) as exc:
                fail(f"source provenance has invalid {key}")
                raise AssertionError from exc
        if (
            values["device"] < 0
            or values["inode"] <= 0
            or values["mtime_epoch"] <= 0
            or values["mtime_ns"] // 1_000_000_000 != values["mtime_epoch"]
        ):
            fail(f"source provenance has invalid {prefix} file identity")
        file_identities[prefix] = values
    if sidecar.get("source_f16_selected_bytes") != "11362369536":
        fail("source provenance has invalid source_f16_selected_bytes")
    try:
        shard_count = int(sidecar["hf_shard_count"])
    except (KeyError, ValueError) as exc:
        fail("source provenance has invalid hf_shard_count")
        raise AssertionError from exc
    if shard_count <= 0:
        fail("source provenance has no HF shards")
    for shard in range(shard_count):
        for suffix in ("name", "resolved", "resolved_basename"):
            if not sidecar.get(f"hf_shard_{shard:03d}_{suffix}"):
                fail(f"source provenance lacks shard {shard} {suffix}")
        name = PurePosixPath(sidecar[f"hf_shard_{shard:03d}_name"])
        resolved = PurePosixPath(sidecar[f"hf_shard_{shard:03d}_resolved"])
        if name.is_absolute() or ".." in name.parts:
            fail(f"source provenance has unsafe shard {shard} name")
        if not resolved.is_absolute():
            fail(f"source provenance has non-absolute shard {shard} resolved path")
        if sidecar[f"hf_shard_{shard:03d}_resolved_basename"] != resolved.name:
            fail(f"source provenance has inconsistent shard {shard} resolved basename")
        try:
            size = int(sidecar[f"hf_shard_{shard:03d}_bytes"])
        except (KeyError, ValueError) as exc:
            fail(f"source provenance has invalid shard {shard} size")
            raise AssertionError from exc
        if size <= 0:
            fail(f"source provenance has non-positive shard {shard} size")
        for suffix in ("device", "inode", "mtime_ns"):
            key = f"hf_shard_{shard:03d}_{suffix}"
            try:
                value = int(sidecar[key])
            except (KeyError, ValueError) as exc:
                fail(f"source provenance has invalid shard {shard} {suffix}")
                raise AssertionError from exc
            invalid = value < 0 if suffix == "device" else value <= 0
            if invalid:
                fail(f"source provenance has invalid shard {shard} {suffix}")
    return {
        "sidecar": manifest["source_f16_provenance"],
        "sidecar_sha256": actual_sha,
        "hf_repository": sidecar["hf_repository"],
        "hf_revision": sidecar["hf_revision"],
        "hf_snapshot_path_revision_match": True,
        "hf_shards": shard_count,
        "hf_shard_content_authentication": sidecar["hf_shard_content_authentication"],
        "producer_git_commit": sidecar["git_commit"],
        "producer_git_dirty": sidecar["git_dirty"] == "true",
        "quantizer": sidecar["quantizer"],
        "quantizer_bytes": small_file_bytes["quantizer_bytes"],
        "hf_index_bytes": small_file_bytes["hf_index_bytes"],
        "hf_config_bytes": small_file_bytes["hf_config_bytes"],
        "quantizer_sha256": sidecar["quantizer_sha256"],
        "producer_input_identities": input_identities,
        "hf_shard_identity_records": shard_count,
        "file_identities_at_publication": file_identities,
        "publication_protocol": sidecar["publication_protocol"],
        "model_payload_hashing": False,
        "non_target_payload_identity": sidecar["non_target_payload_copy"],
    }


def validate_reproducibility(root: Path) -> dict[str, object]:
    required = (
        "provenance/git-status.txt",
        "provenance/git-head.patch",
        "provenance/untracked-source-files.txt",
        "provenance/toolchain-runtime.txt",
        "provenance/binary-sha256.txt",
    )
    for value in required:
        path = root_relative(root, value, "reproducibility artifact")
        if not path.is_file():
            fail(f"reproducibility artifact is missing: {value}")
    for value in (
        "provenance/toolchain-runtime.txt",
        "provenance/binary-sha256.txt",
    ):
        if root_relative(root, value, "reproducibility artifact").stat().st_size == 0:
            fail(f"reproducibility artifact is empty: {value}")
    untracked_archive = root / "provenance/untracked-sources.tar.gz"
    untracked_none = root / "provenance/untracked-sources.none"
    if untracked_archive.is_file() == untracked_none.is_file():
        fail("reproducibility must contain exactly one untracked-source archive/none marker")
    source_list_path = root / "provenance/untracked-source-files.txt"
    source_names = source_list_path.read_text(encoding="utf-8").splitlines()
    if any(not name or name.startswith("-") or PurePosixPath(name).is_absolute()
           or ".." in PurePosixPath(name).parts
           for name in source_names):
        fail("untracked source list contains an unsafe path")
    if len(source_names) != len(set(source_names)):
        fail("untracked source list contains duplicate paths")
    if untracked_archive.is_file():
        if untracked_archive.stat().st_size == 0 or not source_names:
            fail("untracked source archive/list is empty")
        try:
            with tarfile.open(untracked_archive, "r:gz") as archive:
                members = archive.getmembers()
        except (OSError, tarfile.TarError) as exc:
            fail(f"cannot inspect untracked source archive: {exc}")
        member_names = [member.name.removeprefix("./") for member in members]
        if any(
            not member.isfile()
            or name.startswith("-")
            or PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts
            for member, name in zip(members, member_names)
        ):
            fail("untracked source archive contains an unsafe/non-file entry")
        if set(member_names) != set(source_names) or len(member_names) != len(source_names):
            fail("untracked source archive does not exactly match its file list")
    else:
        if source_names:
            fail("untracked source none marker contradicts a nonempty file list")
        if untracked_none.read_text(encoding="utf-8").strip() != "none":
            fail("untracked source none marker is invalid")
    hashes = root_relative(root, "provenance/binary-sha256.txt", "binary hashes").read_text(
        encoding="utf-8"
    ).splitlines()
    expected_binaries = {
        "./ds4-bench",
        "./gguf-tools/quality-testing/score_official",
        "./tests/test_engine_mgpu_placement",
        "./tests/test_gpu_xdev",
    }
    parsed_hashes: dict[str, str] = {}
    for line in hashes:
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None or match.group(2) in parsed_hashes:
            fail("binary-sha256.txt contains an invalid/duplicate record")
        parsed_hashes[match.group(2)] = match.group(1)
    if set(parsed_hashes) != expected_binaries:
        fail("binary-sha256.txt does not contain the four expected binary hashes")
    return {
        "git_patch": "provenance/git-head.patch",
        "untracked_sources": (
            "provenance/untracked-sources.tar.gz"
            if untracked_archive.is_file()
            else "none"
        ),
        "toolchain_runtime": "provenance/toolchain-runtime.txt",
        "binary_hashes": "provenance/binary-sha256.txt",
    }


def fnum(value: object, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def summarize(root: Path) -> dict[str, object]:
    manifest = read_key_values(root / "manifest.txt")
    expected_manifest = {
        "gpu_devices": "0,3,1,2",
        "gpu_vram": "auto",
        "stage_split": "22/21",
        "cuda_device_order": "PCI_BUS_ID",
        "cuda_visible_devices": "unset-required",
        "nvidia_visible_devices": "unset-required",
        "required_direct_peer_links": "0->1,1->0,2->3,3->2",
        "quality_ctx": "32769",
        "prefill_chunk": "2048",
        "contexts": "2048,8192,32768",
        "required_arms": "native-q8,source-f16",
        "rotation": "exact-ab-ba",
        "q8_derived_diagnostic": "not-run-not-required",
        "artifact_path_mode": "root-relative",
        "q8_tensor_inventory": "inventory/model-q8.csv",
        "source_f16_tensor_inventory": "inventory/model-source-f16.csv",
        "dense_tensor_inventory": "344",
        "timed_dense_audit": "disabled",
        "structural_integrity": "required",
        "source_provenance": "required",
        "source_f16_lock": "shared-nonblocking-held",
        "model_file_identity": "bytes-device-inode-mtime",
        "hf_shard_content_authentication": "not_performed",
        "non_target_payload_identity": "producer-copy-guarantee-not-independently-rehashed",
        "model_payload_hashing": "disabled",
    }
    wrong = {
        key: (manifest.get(key), value)
        for key, value in expected_manifest.items()
        if manifest.get(key) != value
    }
    if wrong:
        fail(f"manifest does not describe the fixed two-arm experiment: {wrong}")
    try:
        gen_tokens = int(manifest["gen_tokens"])
        repeats = int(manifest["repeats"])
        ctx_alloc = int(manifest["ctx_alloc"])
    except (KeyError, ValueError) as exc:
        fail("manifest has invalid gen_tokens/repeats/ctx_alloc")
        raise AssertionError from exc
    if (
        gen_tokens <= 0
        or repeats < 4
        or repeats % 2
        or ctx_alloc < 32768 + gen_tokens + 1
    ):
        fail("experiment requires positive generation and an even repeat count of at least 4")

    inventory_path = root_relative(root, manifest["inventory"], "inventory")
    try:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read model inventory: {exc}")
    if not inventory.get("structural_integrity") or inventory.get("payload_hashing") is not False:
        fail("model structural inventory is invalid")
    for key in ("model_q8", "model_source_f16"):
        model = inventory.get(key, {})
        if model.get("target_tensor_count") != 344:
            fail(f"inventory {key} does not contain 344 target tensors")
        if model.get("payload_extents", {}).get("payload_extents_valid") != 1:
            fail(f"inventory {key} does not validate payload extents")
    if inventory.get("directory_parity", {}).get("kv_metadata_byte_identical") is not True:
        fail("inventory does not establish raw GGUF KV metadata identity")
    if inventory["model_q8"]["path"] != manifest.get("model_q8"):
        fail("inventory MODEL_Q8 differs from manifest")
    if inventory["model_source_f16"]["path"] != manifest.get("model_source_f16"):
        fail("inventory MODEL_SOURCE_F16 differs from manifest")
    for manifest_key, inventory_key in (
        ("model_q8_bytes", "model_q8"),
        ("model_source_f16_bytes", "model_source_f16"),
    ):
        try:
            manifest_bytes = int(manifest[manifest_key])
        except (KeyError, ValueError) as exc:
            fail(f"manifest has invalid {manifest_key}")
            raise AssertionError from exc
        if manifest_bytes != inventory[inventory_key]["file_bytes"]:
            fail(f"manifest {manifest_key} differs from structural inventory")

    provenance = validate_provenance(root, manifest)
    sidecar = read_key_values(root_relative(root, manifest["source_f16_provenance"], "source provenance"))
    if sidecar.get("template_model") != manifest.get("model_q8"):
        fail("source provenance template differs from manifest MODEL_Q8")
    if sidecar.get("source_f16_model") != manifest.get("model_source_f16"):
        fail("source provenance output differs from manifest MODEL_SOURCE_F16")
    if int(sidecar.get("template_model_bytes", "-1")) != inventory["model_q8"]["file_bytes"]:
        fail("source provenance Q8 size differs from inventory")
    if int(sidecar.get("source_f16_model_bytes", "-1")) != inventory["model_source_f16"]["file_bytes"]:
        fail("source provenance F16 size differs from inventory")
    reproducibility = validate_reproducibility(root)
    dense_inventories = {
        "native-q8": read_dense_inventory(
            root_relative(root, manifest["q8_tensor_inventory"], "Q8 tensor inventory"),
            "native-q8",
        ),
        "source-f16": read_dense_inventory(
            root_relative(
                root,
                manifest["source_f16_tensor_inventory"],
                "source-F16 tensor inventory",
            ),
            "source-f16",
        ),
    }

    coverage: dict[str, dict[str, object]] = {}
    scores: dict[str, dict[str, dict[str, str]]] = {}
    quality: dict[str, dict[str, object]] = {}
    quality_memory: dict[str, dict[str, object]] = {}
    for arm in ARMS:
        validate_quality_log(root / f"quality/{arm}.log")
        coverage[arm] = validate_audit(
            root / f"quality/{arm}.dense-audit.csv", arm, dense_inventories[arm]
        )
        scores[arm] = read_scores(root / f"quality/{arm}.tsv")
        quality[arm] = quality_summary(scores[arm])
        quality_memory[arm] = read_memory(root / f"memory/quality-{arm}.csv")

    native_ids = set(scores["native-q8"])
    if set(scores["source-f16"]) != native_ids:
        fail("quality IDs differ between the two arms")
    if any(
        scores["source-f16"][case]["target_tokens"]
        != scores["native-q8"][case]["target_tokens"]
        for case in native_ids
    ):
        fail("quality target-token counts differ between the two arms")
    source_quality = quality["source-f16"]
    native_quality = quality["native-q8"]
    source_quality["delta_avg_nll_vs_native"] = float(source_quality["avg_nll"]) - float(
        native_quality["avg_nll"]
    )
    source_quality["first_match_delta_vs_native"] = int(source_quality["first_matches"]) - int(
        native_quality["first_matches"]
    )
    source_quality["greedy_lcp_delta_vs_native"] = float(
        source_quality["avg_greedy_lcp"]
    ) - float(native_quality["avg_greedy_lcp"])
    case_deltas = {
        case_id: float(scores["source-f16"][case_id]["avg_nll"])
        - float(scores["native-q8"][case_id]["avg_nll"])
        for case_id in native_ids
    }
    wins = sum(delta < 0 for delta in case_deltas.values())
    losses = sum(delta > 0 for delta in case_deltas.values())
    ci_lower, ci_upper, upper95 = bootstrap_nll_delta(
        sorted(native_ids), scores["native-q8"], scores["source-f16"]
    )
    source_quality["case_nll_wins_losses_ties"] = {
        "wins": wins,
        "losses": losses,
        "ties": len(case_deltas) - wins - losses,
    }
    source_quality["max_case_avg_nll_delta"] = max(case_deltas.values())
    source_quality["median_case_avg_nll_delta"] = statistics.median(case_deltas.values())
    source_quality["paired_bootstrap_95pct_ci"] = [ci_lower, ci_upper]
    source_quality["paired_bootstrap_one_sided_upper95"] = upper95

    runs_path = root_relative(root, manifest["runs_table"], "runs table")
    run_rows = read_rows(runs_path, "\t")
    require_columns(runs_path, run_rows, {"repeat", "slot", "arm", "csv", "log", "memory"}, "\t")
    expected_runs = repeats * len(ARMS)
    if len(run_rows) != expected_runs:
        fail(f"runs.tsv has {len(run_rows)} rows; expected {expected_runs}")
    by_context: dict[str, dict[int, list[dict[str, float]]]] = {
        arm: defaultdict(list) for arm in ARMS
    }
    seen: set[tuple[int, int]] = set()
    run_memory: dict[str, dict[str, object]] = {}
    for row in run_rows:
        repeat = as_int(row, "repeat", "runs.tsv", 1)
        slot = as_int(row, "slot", "runs.tsv", 1)
        if repeat > repeats or slot > 2 or (repeat, slot) in seen:
            fail(f"runs.tsv has invalid/duplicate repeat-slot {repeat}/{slot}")
        seen.add((repeat, slot))
        expected_arm = ARMS[((slot - 1) + ((repeat - 1) % 2)) % 2]
        arm = row.get("arm")
        if arm != expected_arm:
            fail(
                f"runs.tsv repeat={repeat} slot={slot} has arm={arm!r}; "
                f"expected exact AB/BA arm {expected_arm!r}"
            )
        csv_path = root_relative(root, row["csv"], "benchmark CSV")
        log_path = root_relative(root, row["log"], "benchmark log")
        memory_path = root_relative(root, row["memory"], "memory CSV")
        for path in (csv_path, log_path, memory_path):
            if not path.is_file() or path.stat().st_size == 0:
                fail(f"run evidence is missing: {path}")
        run_memory[f"{arm}-r{repeat}"] = read_memory(memory_path)
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if "CUDA EP forced pipeline split 22/21" not in log_text:
            fail(f"{log_path} lacks fixed split marker")
        require_direct_peer_markers(log_path, log_text)
        if "dense-exec-audit" in log_text.lower():
            fail(f"{log_path} indicates timed dense audit instrumentation")
        for ctx, values in read_bench(csv_path, gen_tokens).items():
            by_context[arm][ctx].append(values)
    if seen != {(repeat, slot) for repeat in range(1, repeats + 1) for slot in (1, 2)}:
        fail("performance repeat/slot matrix is incomplete")

    performance: dict[str, dict[str, object]] = {}
    for arm in ARMS:
        contexts: dict[str, object] = {}
        for ctx in (2048, 8192, 32768):
            rows = by_context[arm][ctx]
            if len(rows) != repeats:
                fail(f"{arm}/{ctx} has {len(rows)} repeats; expected {repeats}")
            medians = {
                key: statistics.median(row[key] for row in rows) for key in rows[0]
            }
            contexts[str(ctx)] = {"median": medians, "samples": rows}
        performance[arm] = {"contexts": contexts}
    decode_ratios: list[float] = []
    prefill_ratios: list[float] = []
    for ctx in (2048, 8192, 32768):
        native = performance["native-q8"]["contexts"][str(ctx)]["median"]  # type: ignore[index]
        source = performance["source-f16"]["contexts"][str(ctx)]["median"]  # type: ignore[index]
        source["gen_tps_ratio_vs_native"] = source["gen_tps"] / native["gen_tps"]
        source["prefill_tps_ratio_vs_native"] = source["prefill_tps"] / native["prefill_tps"]
        decode_ratios.append(source["gen_tps_ratio_vs_native"])
        prefill_ratios.append(source["prefill_tps_ratio_vs_native"])
    performance["source-f16"]["decode_geomean_ratio_vs_native"] = (
        math.prod(decode_ratios) ** (1.0 / len(decode_ratios))
    )
    performance["source-f16"]["prefill_geomean_ratio_vs_native"] = (
        math.prod(prefill_ratios) ** (1.0 / len(prefill_ratios))
    )

    payload: dict[str, object] = {
        "primary_experiment_integrity": True,
        "integrity_scope": {
            "structural_directory_and_extent_validation": True,
            "source_provenance_validated": True,
            "model_payload_hashing": False,
            "non_target_payload_identity": "producer-copy-guarantee-not-independently-rehashed",
        },
        "comparison": "native Q8/SM75 versus source-derived F16/cuBLAS",
        "q8_derived_diagnostic": "not run and not required",
        "inventory": inventory,
        "source_provenance": provenance,
        "reproducibility": reproducibility,
        "runtime_coverage": coverage,
        "quality": quality,
        "performance": performance,
        "memory_provenance": {"quality": quality_memory, "timed_runs": run_memory},
    }
    (root / "dense-f16-decode-ab-results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    ci = source_quality["paired_bootstrap_95pct_ci"]
    record = source_quality["case_nll_wins_losses_ties"]
    lines = [
        "# Dense decode A/B: native Q8 versus source-derived F16",
        "",
        "Primary experiment integrity: **PASS**",
        "",
        "The GGUF directories, raw KV metadata identity, payload extents, exact 344-tensor "
        "inventories, producer provenance, runtime paths, and AB/BA run order passed. "
        "Large model payloads were not hashed. Non-target payload identity relies on the "
        "recorded producer copy-through guarantee and is not independently rehashed. "
        "The fixed HF snapshot path, shard names/resolved basenames/sizes, config, and index "
        "were recorded, but shard contents were not authenticated.",
        "",
        "Q8-derived FP16 is an optional attribution experiment and was not required or run.",
        "",
        "## Production one-token decode coverage",
        "",
        "| Arm | Stored type | Exact backend | Decode tensors | Fallbacks |",
        "|---|---|---|---:|---:|",
        "| native-q8 | Q8_0 | sm75_native_q8 + sm75_native_q8_fused_shared_mid | "
        f"{coverage['native-q8']['distinct_decode_tensors']} | 0 |",
        f"| source-f16 | F16 | {SOURCE_F16_BACKEND} | "
        f"{coverage['source-f16']['distinct_decode_tensors']} | 0 |",
        "",
        "Native-Q8 fused shared-middle decode proof: "
        f"**{coverage['native-q8']['fused_shared_mid_decode_tensors']}/86 tensors**.",
        "",
        "## Official 100-case quality",
        "",
        "| Arm | Avg NLL | Delta vs native | First matches | Avg greedy LCP | API target MAE |",
        "|---|---:|---:|---:|---:|---:|",
        f"| native-q8 | {fnum(native_quality['avg_nll'], 9)} | +0.000000000 | "
        f"{native_quality['first_matches']}/{CASES} | {fnum(native_quality['avg_greedy_lcp'])} | "
        f"{fnum(native_quality['api_target_mae'], 6)} |",
        f"| source-f16 | {fnum(source_quality['avg_nll'], 9)} | "
        f"{float(source_quality['delta_avg_nll_vs_native']):+.9f} | "
        f"{source_quality['first_matches']}/{CASES} | {fnum(source_quality['avg_greedy_lcp'])} | "
        f"{fnum(source_quality['api_target_mae'], 6)} |",
        "",
        f"Source-F16 paired NLL-delta 95% CI: **[{ci[0]:+.9f}, {ci[1]:+.9f}]**; "
        f"case wins/losses/ties: **{record['wins']}/{record['losses']}/{record['ties']}**.",
        "",
        "## Prefill and decode performance (median tokens/s)",
        "",
        "| Context | Native prefill | Source-F16 prefill | Prefill ratio | "
        "Native decode | Source-F16 decode | Decode ratio |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ctx in (2048, 8192, 32768):
        native = performance["native-q8"]["contexts"][str(ctx)]["median"]  # type: ignore[index]
        source = performance["source-f16"]["contexts"][str(ctx)]["median"]  # type: ignore[index]
        lines.append(
            f"| {ctx} | {native['prefill_tps']:.3f} | {source['prefill_tps']:.3f} | "
            f"{source['prefill_tps_ratio_vs_native']:.3f}x | {native['gen_tps']:.3f} | "
            f"{source['gen_tps']:.3f} | {source['gen_tps_ratio_vs_native']:.3f}x |"
        )
    lines += [
        "",
        f"Source-F16 prefill geomean vs native: **{float(performance['source-f16']['prefill_geomean_ratio_vs_native']):.3f}x**.",
        f"Source-F16 decode geomean vs native: **{float(performance['source-f16']['decode_geomean_ratio_vs_native']):.3f}x**.",
        "",
    ]
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path)
    parser.add_argument("--validate-audit", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--inventory", type=Path)
    args = parser.parse_args()
    if args.validate_audit is not None:
        if args.arm is None or args.inventory is None or args.root is not None:
            fail("--validate-audit requires --arm, --inventory, and no ROOT")
        inventory = read_dense_inventory(args.inventory, args.arm)
        print(
            json.dumps(
                {
                    "arm": args.arm,
                    "coverage": validate_audit(
                        args.validate_audit, args.arm, inventory
                    ),
                },
                indent=2,
            )
        )
        return 0
    if args.root is None or args.arm is not None or args.inventory is not None:
        fail("usage: summarize-dense-f16-decode-ab.py ROOT")
    summarize(args.root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
