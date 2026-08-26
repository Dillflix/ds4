#!/usr/bin/env python3
"""Summarize one mixed-Q4/IQ2, balanced-T256 production trace."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def die(message: str) -> None:
    raise SystemExit(f"error: {message}")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        die(f"missing or empty input: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_range(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for part in text.split("/"):
        if "=" in part:
            key, value = part.split("=", 1)
            result[key] = value
    return result


def classify(row: dict[str, str]) -> str:
    name = row["name"]
    lower = name.lower()
    partner = bool(row["partner_range"])
    if partner:
        if row["kind"] == "memcpy":
            return "partner_t256_memcpy"
        if "f32_to_f16_kernel" in name:
            return "partner_t256_activation_convert"
        return "partner_t256_cublas"
    if row["kind"] == "memcpy":
        return "other_memcpy"
    if "moe_gate_up_mid_sm75_native_q4_tile8_kernel" in name:
        return "native_q4_gate_up"
    if (
        "moe_gate_up_mid_iq2_" in name
        or "moe_gate_up_mid_expert_tile4_row32_kernel" in name
    ):
        return "iq2_gate_up"
    if "moe_down_sm75_native_q4_tile_kernel" in name:
        return "native_q4_down"
    if "matmul_q8_0_mma_sm75_exact_kernel" in name:
        return "dense_q8_native"
    if "attention" in lower:
        return "attention"
    if "moe_" in lower:
        return "other_moe"
    return "other"


def main() -> int:
    if len(sys.argv) != 2:
        die("usage: summarize-sm75-native-q4-t256-profile.py OUTPUT_DIR")
    output = Path(sys.argv[1]).resolve()
    operations = read_csv(output / "operation-attribution.csv")
    benchmark = read_csv(output / "nsys" / "combined-benchmark.csv")
    bindings = read_csv(output / "nsys" / "bindings.csv")

    partner_bindings = [row for row in bindings if row["partner_offload"] == "1"]
    expected_binding_labels = {
        f"tensor:blk.{layer}.attn_output_b.weight" for layer in range(1, 43, 2)
    }
    actual_binding_labels = {row["label"] for row in partner_bindings}
    if len(partner_bindings) != 21 or actual_binding_labels != expected_binding_labels:
        missing_labels = sorted(expected_binding_labels - actual_binding_labels)
        extra_labels = sorted(actual_binding_labels - expected_binding_labels)
        die(
            "expected exactly one partner T256 binding for every odd layer; "
            f"found {len(partner_bindings)} bindings "
            f"(missing={missing_labels[:1]}, extra={extra_labels[:1]})"
        )
    for row in partner_bindings:
        if (
            row["in_dim"] != "8192"
            or row["out_dim"] != "4096"
            or row["partner_arithmetic"] != "f16"
            or row["consumer_device"] == row["resident_device"]
        ):
            die(f"invalid partner T256 binding: {row['label']}")

    groups: dict[str, dict[str, int]] = defaultdict(
        lambda: {"duration_ns": 0, "operations": 0, "bytes": 0}
    )
    partner_ranges: dict[str, dict[str, object]] = {}
    partner_range_labels: set[str] = set()
    partner_shape_errors: list[str] = []
    for row in operations:
        group = classify(row)
        values = groups[group]
        values["duration_ns"] += int(row["duration_ns"])
        values["operations"] += 1
        values["bytes"] += int(row["bytes"])
        range_name = row["partner_range"]
        if not range_name:
            continue
        fields = parse_range(range_name)
        expected = {
            "label": "attn_output_b",
            "tokens": "512",
            "in": "8192",
            "out": "4096",
            "result": "f32",
            "arithmetic": "f16",
        }
        mismatched = [
            f"{key}={fields.get(key, 'missing')}"
            for key, value in expected.items()
            if fields.get(key) != value
        ]
        if mismatched:
            partner_shape_errors.append(
                f"{range_name}: " + ", ".join(mismatched)
            )
        label = fields.get("label", "")
        if label:
            partner_range_labels.add(label)
        item = partner_ranges.setdefault(
            range_name,
            {
                "range": range_name,
                "label": label,
                "home_device": fields.get("home_device", ""),
                "partner_device": fields.get("partner_device", ""),
                "kernel_ns": 0,
                "kernel_count": 0,
                "memcpy_ns": 0,
                "memcpy_count": 0,
                "memcpy_bytes": 0,
            },
        )
        if row["kind"] == "kernel":
            item["kernel_ns"] = int(item["kernel_ns"]) + int(row["duration_ns"])
            item["kernel_count"] = int(item["kernel_count"]) + 1
        elif row["kind"] == "memcpy":
            item["memcpy_ns"] = int(item["memcpy_ns"]) + int(row["duration_ns"])
            item["memcpy_count"] = int(item["memcpy_count"]) + 1
            item["memcpy_bytes"] = int(item["memcpy_bytes"]) + int(row["bytes"])

    if partner_shape_errors:
        die("unexpected partner range shape: " + partner_shape_errors[0])
    required = {
        "native_q4_gate_up",
        "iq2_gate_up",
        "native_q4_down",
        "partner_t256_activation_convert",
        "partner_t256_cublas",
        "partner_t256_memcpy",
    }
    missing = sorted(required.difference(groups))
    if missing:
        die("trace omitted required combined-path groups: " + ", ".join(missing))
    if partner_range_labels != {"attn_output_b"}:
        die(
            "unexpected partner NVTX class labels: "
            + ", ".join(sorted(partner_range_labels))
        )

    prefill_tokens = int(benchmark[0]["prefill_tokens"])
    if prefill_tokens % 512:
        die(f"prefill token count is not divisible by 512: {prefill_tokens}")
    expected_projections = 21 * (prefill_tokens // 512)
    expected_counts = {
        "partner_t256_activation_convert": expected_projections,
        "partner_t256_cublas": expected_projections,
        "partner_t256_memcpy": 2 * expected_projections,
    }
    for group, expected_count in expected_counts.items():
        actual_count = groups[group]["operations"]
        if actual_count != expected_count:
            die(
                f"expected {expected_count} {group} operations, "
                f"found {actual_count}"
            )

    kernel_total_ns = sum(
        int(row["duration_ns"])
        for row in operations
        if row["kind"] == "kernel"
    )
    group_rows: list[dict[str, object]] = []
    for name, values in groups.items():
        group_rows.append(
            {
                "group": name,
                "duration_ns": values["duration_ns"],
                "kernel_time_pct": (
                    f"{100.0 * values['duration_ns'] / kernel_total_ns:.6f}"
                    if kernel_total_ns and not name.endswith("memcpy")
                    else "0.000000"
                ),
                "operations": values["operations"],
                "bytes": values["bytes"],
            }
        )
    group_rows.sort(key=lambda item: int(item["duration_ns"]), reverse=True)
    write_csv(
        output / "combined-kernel-groups.csv",
        ["group", "duration_ns", "kernel_time_pct", "operations", "bytes"],
        group_rows,
    )

    partner_rows = sorted(
        partner_ranges.values(), key=lambda item: str(item["label"])
    )
    write_csv(
        output / "partner-t256-ranges.csv",
        [
            "range", "label", "home_device", "partner_device",
            "kernel_ns", "kernel_count", "memcpy_ns", "memcpy_count",
            "memcpy_bytes",
        ],
        partner_rows,
    )

    tps = float(benchmark[0]["prefill_tps"])
    evidence = {
        "accepted": True,
        "prefill_tokens": prefill_tokens,
        "prefill_tps": tps,
        "annotated_kernel_time_ns": kernel_total_ns,
        "partner_t256_binding_count": len(partner_bindings),
        "partner_t256_projection_count": expected_projections,
        "partner_t256_nvtx_range_count": len(partner_ranges),
        "groups": {
            row["group"]: {
                "duration_ns": int(row["duration_ns"]),
                "kernel_time_pct": float(row["kernel_time_pct"]),
                "operations": int(row["operations"]),
                "bytes": int(row["bytes"]),
            }
            for row in group_rows
        },
    }
    (output / "combined-profile.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )

    by_name = {row["group"]: row for row in group_rows}
    with (output / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# SM75 mixed-Q4/IQ2 + balanced-T256 profile\n\n")
        handle.write(
            f"Bounded {prefill_tokens}-token prefill at a production 256K "
            f"allocation: **{tps:.2f} tokens/s**.\n\n"
        )
        handle.write("| Attributed group | GPU time | Share of annotated kernel time | Operations |\n")
        handle.write("|---|---:|---:|---:|\n")
        for name in (
            "native_q4_gate_up", "iq2_gate_up", "native_q4_down",
            "partner_t256_activation_convert", "partner_t256_cublas",
            "partner_t256_memcpy", "dense_q8_native", "attention",
            "other_moe", "other", "other_memcpy",
        ):
            row = by_name.get(name)
            if not row:
                continue
            handle.write(
                f"| {name} | {int(row['duration_ns']) / 1e6:.3f} ms | "
                f"{float(row['kernel_time_pct']):.2f}% | "
                f"{row['operations']} |\n"
            )
        copy_group = groups["partner_t256_memcpy"]
        handle.write(
            "\nPartner T256 coverage: "
            f"**{len(partner_bindings)} unique layer bindings** and "
            f"**{expected_projections} captured projections**; "
            f"{copy_group['bytes'] / 1073741824.0:.3f} GiB of captured "
            "peer-copy traffic.\n"
        )
        handle.write(
            "\nNsight Compute reports in `ncu/` provide occupancy, compute, "
            "memory, and warp-stall evidence for the requested bounded "
            "kernel set (full routed/dense coverage or targeted 32K "
            "attention).\n"
        )
    print((output / "summary.md").read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
