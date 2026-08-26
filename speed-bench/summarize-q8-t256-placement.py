#!/usr/bin/env python3
"""Validate and summarize deliberate SM75 T256 execution placement policies."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
import re


VARIANTS = ("native", "all-local", "balanced", "overflow", "all-partner")
EXPECTED_BINDINGS = {
    "native": (0, 0),
    "all-local": (86, 0),
    "balanced": (65, 21),
    "overflow": (79, 7),
    "all-partner": (43, 43),
}
EXPECTED_RESULTS = {
    "native": {"native_q8"},
    "all-local": {"f16_hit"},
    "balanced": {"f16_hit", "f16_partner_hit"},
    "overflow": {"f16_hit", "f16_partner_hit"},
    "all-partner": {"f16_partner_hit"},
}


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if reader.fieldnames is None:
                fail(f"{path} has no header")
            return list(reader)
    except OSError as exc:
        fail(f"cannot read {path}: {exc}")


def manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result.setdefault(key, value)
    return result


def is_t256(row: dict[str, str]) -> bool:
    return (
        row.get("in_dim") == "8192"
        and row.get("out_dim") == "4096"
        and "attn_output_b" in f"{row.get('module', '')} {row.get('label', '')}"
    )


def layer_of(row: dict[str, str]) -> int:
    value = row.get("layer", "")
    if value.isdigit():
        return int(value)
    match = re.search(r"blk\.(\d+)\.", row.get("label", ""))
    if not match:
        fail(f"cannot recover layer from {row.get('label', '<missing>')}")
    return int(match.group(1))


def devices_for_layer(layer: int) -> tuple[int, int]:
    return (0, 1) if layer <= 21 else (3, 2)


def partner_layer(variant: str, layer: int) -> bool:
    if variant == "balanced":
        return (layer & 1) != 0
    if variant == "overflow":
        return 15 <= layer <= 21
    return variant == "all-partner"


def validate_run(run: dict[str, str]) -> tuple[dict[int, float], dict[str, object]]:
    variant = run["variant"]
    binding_rows = rows(Path(run["bindings"]))
    t256_bindings = [row for row in binding_rows if is_t256(row)]
    local = [row for row in t256_bindings if row["partner_offload"] == "0"]
    partner = [row for row in t256_bindings if row["partner_offload"] == "1"]
    expected_local, expected_partner = EXPECTED_BINDINGS[variant]
    if (len(local), len(partner)) != (expected_local, expected_partner):
        fail(
            f"{variant} repeat {run['repeat']} has T256 bindings "
            f"{len(local)}+{len(partner)}, expected {expected_local}+{expected_partner}"
        )
    if any(row["partner_arithmetic"] != "f16" for row in t256_bindings):
        fail(f"{variant} repeat {run['repeat']} contains non-F16 T256 bindings")
    observed_bindings = Counter(
        (
            layer_of(row), int(row["consumer_device"]),
            int(row["resident_device"]), int(row["partner_offload"]),
        )
        for row in t256_bindings
    )
    expected_bindings: Counter[tuple[int, int, int, int]] = Counter()
    if variant != "native":
        for layer in range(43):
            home, peer = devices_for_layer(layer)
            # Fixed pair-resident consumer used by the sharded attention path.
            expected_bindings[(layer, peer, peer, 0)] += 1
            if partner_layer(variant, layer):
                expected_bindings[(layer, home, peer, 1)] += 1
            else:
                expected_bindings[(layer, home, home, 0)] += 1
    if observed_bindings != expected_bindings:
        fail(
            f"{variant} repeat {run['repeat']} has the expected aggregate "
            "binding count but the wrong per-layer consumer/resident mapping"
        )

    audit_rows = [row for row in rows(Path(run["audit"])) if is_t256(row)]
    if not audit_rows:
        fail(f"{variant} repeat {run['repeat']} has no T256 runtime evidence")
    result_counts = Counter(row["result"] for row in audit_rows)
    if set(result_counts) != EXPECTED_RESULTS[variant]:
        fail(
            f"{variant} repeat {run['repeat']} has wrong T256 paths: "
            f"{dict(result_counts)}"
        )
    layer_counts = Counter(layer_of(row) for row in audit_rows)
    if set(layer_counts) != set(range(43)) or len(set(layer_counts.values())) != 1:
        fail(f"{variant} repeat {run['repeat']} has unequal layer coverage")
    partner_layers = sorted({
        layer_of(row) for row in audit_rows
        if row["result"] == "f16_partner_hit"
    })
    expected_partner_layers = {
        "native": [],
        "all-local": [],
        "balanced": list(range(1, 43, 2)),
        "overflow": list(range(15, 22)),
        "all-partner": list(range(43)),
    }[variant]
    if partner_layers != expected_partner_layers:
        fail(
            f"{variant} repeat {run['repeat']} partner layers are "
            f"{partner_layers}, expected {expected_partner_layers}"
        )
    for row in audit_rows:
        layer = layer_of(row)
        home, peer = devices_for_layer(layer)
        uses_partner = partner_layer(variant, layer)
        if variant == "native":
            expected_result, expected_reason, expected_device = (
                "native_q8", "disabled_t256_by_env", home
            )
        elif uses_partner:
            expected_result, expected_reason, expected_device = (
                "f16_partner_hit", "nvlink_offload", peer
            )
        else:
            expected_result, expected_reason, expected_device = (
                "f16_hit", "resident", home
            )
        observed = (
            row["result"], row.get("reason", ""),
            int(row.get("physical_device", "-1")),
        )
        if observed != (expected_result, expected_reason, expected_device):
            fail(
                f"{variant} repeat {run['repeat']} layer {layer} executed "
                f"{observed}, expected "
                f"{(expected_result, expected_reason, expected_device)}"
            )

    log = Path(run["log"]).read_text(encoding="utf-8", errors="replace")
    for marker in (
        "CUDA EP forced pipeline split 22/21",
        f"t256-placement={'overflow' if variant == 'native' else variant}",
    ):
        if marker not in log:
            fail(f"{variant} repeat {run['repeat']} lacks marker: {marker}")
    if variant != "native" and "T256-output_b=86/86" not in log:
        fail(f"{variant} repeat {run['repeat']} did not admit 86/86 T256 bindings")

    perf = rows(Path(run["csv"]))
    values: dict[int, float] = {}
    for row in perf:
        context = int(row["ctx_tokens"])
        tps = float(row["prefill_tps"])
        if context in values or not math.isfinite(tps) or tps <= 0:
            fail(f"{variant} repeat {run['repeat']} contains invalid throughput")
        values[context] = tps
    if not values:
        fail(f"{variant} repeat {run['repeat']} has no performance rows")

    # Record all binding shapes so policy-induced displacement is explicit.
    binding_shapes = Counter(
        (row.get("in_dim", ""), row.get("out_dim", ""), row["partner_offload"])
        for row in binding_rows
    )
    evidence = {
        "repeat": int(run["repeat"]),
        "variant": variant,
        "t256_local_bindings": len(local),
        "t256_partner_bindings": len(partner),
        "t256_runtime_results": dict(sorted(result_counts.items())),
        "t256_calls_per_layer": next(iter(layer_counts.values())),
        "partner_layers": partner_layers,
        "total_bindings": len(binding_rows),
        "binding_shapes": {
            f"{key[0]}x{key[1]}:{'partner' if key[2] == '1' else 'local'}": value
            for key, value in sorted(binding_shapes.items())
        },
    }
    return values, evidence


def main() -> int:
    if len(sys.argv) != 2:
        fail("usage: summarize-q8-t256-placement.py OUTPUT_DIR")
    root = Path(sys.argv[1]).resolve()
    meta = manifest(root / "manifest.txt")
    if (
        meta.get("gpu_devices") != "0,3,1,2"
        or meta.get("gpu_vram") != "auto"
        or meta.get("stage_split") != "22/21"
        or meta.get("variants") != ",".join(VARIANTS)
    ):
        fail("manifest does not describe the fixed T256 placement experiment")

    run_rows = rows(root / "runs.tsv", "\t")
    samples: dict[tuple[int, str], dict[int, float]] = {}
    slots: dict[int, dict[int, str]] = defaultdict(dict)
    evidence: list[dict[str, object]] = []
    for run in run_rows:
        repeat = int(run["repeat"])
        slot = int(run["slot"])
        variant = run["variant"]
        if variant not in VARIANTS:
            fail(f"unknown variant in run table: {variant}")
        key = (repeat, variant)
        if key in samples or slot in slots[repeat]:
            fail("run table contains a duplicate variant or slot")
        slots[repeat][slot] = variant
        samples[key], run_evidence = validate_run(run)
        evidence.append(run_evidence)

    repeats = sorted({repeat for repeat, _variant in samples})
    if not repeats or repeats != list(range(1, len(repeats) + 1)):
        fail("repeats must be consecutive starting at one")
    expected = {(repeat, variant) for repeat in repeats for variant in VARIANTS}
    if set(samples) != expected:
        fail("run table lacks one result for every variant/repeat")
    contexts = sorted(set.intersection(*(set(value) for value in samples.values())))
    if any(set(value) != set(contexts) for value in samples.values()):
        fail("performance contexts differ between runs")
    for repeat in repeats:
        order = [slots[repeat][slot] for slot in sorted(slots[repeat])]
        expected_order = [VARIANTS[(slot + repeat - 1) % len(VARIANTS)] for slot in range(5)]
        if order != expected_order:
            fail(f"repeat {repeat} order is {order}, expected {expected_order}")

    summary: dict[str, dict[int, dict[str, float]]] = {}
    for variant in VARIANTS:
        summary[variant] = {}
        for context in contexts:
            values = [samples[(repeat, variant)][context] for repeat in repeats]
            median = statistics.median(values)
            summary[variant][context] = {
                "median_tps": median,
                "min_tps": min(values),
                "max_tps": max(values),
                "vs_native": median / statistics.median(
                    samples[(repeat, "native")][context] for repeat in repeats
                ),
                "vs_overflow": median / statistics.median(
                    samples[(repeat, "overflow")][context] for repeat in repeats
                ),
            }

    high_confidence = len(repeats) >= 5 and {
        tuple(slots[repeat][slot] for slot in sorted(slots[repeat]))
        for repeat in repeats[:5]
    } == {
        tuple(VARIANTS[offset:] + VARIANTS[:offset])
        for offset in range(5)
    }
    payload = {
        "experiment_integrity": True,
        "high_confidence_counterbalanced": high_confidence,
        "repeats": len(repeats),
        "contexts": contexts,
        "performance": summary,
        "run_evidence": evidence,
    }
    (root / "t256-placement.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# T256 execution-placement comparison",
        "",
        "Experiment integrity: **PASS**",
        "",
        f"Counterbalanced confidence: **{'PASS' if high_confidence else 'SCREEN ONLY'}** "
        f"({len(repeats)} repeats).",
        "",
        "Binding inventory and active runtime paths were validated independently for every run.",
        "",
        "| Context | Native | All local | 22 local + 21 partner | Overflow | All partner | Best |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for context in contexts:
        medians = {variant: summary[variant][context]["median_tps"] for variant in VARIANTS}
        best = max(medians, key=medians.get)
        lines.append(
            f"| {context} | {medians['native']:.2f} | {medians['all-local']:.2f} | "
            f"{medians['balanced']:.2f} | {medians['overflow']:.2f} | "
            f"{medians['all-partner']:.2f} | {best} |"
        )
    lines.extend((
        "",
        "`all-local` is allowed to displace lower-ranked cache entries in order to make all "
        "86 T256 copies local. The exported binding-shape inventory records that resource "
        "tradeoff; this is an end-to-end cache-policy comparison, not an isolated kernel test.",
        "",
    ))
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
