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
    "all-local": (43, 0),
    "balanced": (22, 21),
    "all-partner": (0, 43),
}
EXPECTED_RESULTS = {
    "native": {"native_q8"},
    "all-local": {"f16_hit"},
    "balanced": {"f16_hit", "f16_partner_hit"},
    "all-partner": {"f16_partner_hit"},
}
OVERFLOW_ELIGIBLE_PARTNER_LAYERS = frozenset(range(15, 22))


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
    return variant == "all-partner"


def validate_run(run: dict[str, str]) -> tuple[dict[int, float], dict[str, object]]:
    variant = run["variant"]
    binding_rows = rows(Path(run["bindings"]))
    allocation_rows = rows(Path(run["allocations"]))
    t256_bindings = [row for row in binding_rows if is_t256(row)]
    local = [row for row in t256_bindings if row["partner_offload"] == "0"]
    partner = [row for row in t256_bindings if row["partner_offload"] == "1"]
    binding_partner_layers = sorted(layer_of(row) for row in partner)
    binding_partner_layer_set = set(binding_partner_layers)
    if variant == "overflow":
        if len(t256_bindings) != 43 or len(local) + len(partner) != 43:
            fail(
                f"overflow repeat {run['repeat']} has T256 bindings "
                f"{len(local)}+{len(partner)}, expected 43 total live bindings"
            )
        if len(partner) > len(OVERFLOW_ELIGIBLE_PARTNER_LAYERS):
            fail(
                f"overflow repeat {run['repeat']} has {len(partner)} partner "
                "bindings, expected at most 7"
            )
        ineligible = sorted(
            binding_partner_layer_set - OVERFLOW_ELIGIBLE_PARTNER_LAYERS
        )
        if ineligible:
            fail(
                f"overflow repeat {run['repeat']} has ineligible partner "
                f"layers {ineligible}; eligible layers are 15-21"
            )
        if len(binding_partner_layers) != len(binding_partner_layer_set):
            fail(f"overflow repeat {run['repeat']} duplicates a partner layer")
    else:
        expected_local, expected_partner = EXPECTED_BINDINGS[variant]
        if (len(local), len(partner)) != (expected_local, expected_partner):
            fail(
                f"{variant} repeat {run['repeat']} has T256 bindings "
                f"{len(local)}+{len(partner)}, expected "
                f"{expected_local}+{expected_partner}"
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
            uses_partner = (
                layer in binding_partner_layer_set
                if variant == "overflow"
                else partner_layer(variant, layer)
            )
            if uses_partner:
                expected_bindings[(layer, home, peer, 1)] += 1
            else:
                expected_bindings[(layer, home, home, 0)] += 1
    if observed_bindings != expected_bindings:
        fail(
            f"{variant} repeat {run['repeat']} has the expected aggregate "
            "binding count but the wrong per-layer consumer/resident mapping"
        )

    allocation_by_id: dict[int, dict[str, str]] = {}
    for row in allocation_rows:
        allocation_id = int(row["allocation_id"])
        if allocation_id <= 0 or allocation_id in allocation_by_id:
            fail(
                f"{variant} repeat {run['repeat']} has an invalid or duplicate "
                f"physical allocation id {allocation_id}"
            )
        allocation_by_id[allocation_id] = row
        if row.get("usage_tracking") != "1":
            fail(f"{variant} repeat {run['repeat']} lacks warm-up liveness tracking")
        resident_bytes = int(row["resident_bytes"])
        used_calls = int(row["used_calls"])
        expected_dead = resident_bytes if used_calls == 0 else 0
        if int(row["dead_bytes"]) != expected_dead:
            fail(
                f"{variant} repeat {run['repeat']} allocation {allocation_id} "
                "has inconsistent dead-byte accounting"
            )

    t256_allocation_ids: set[int] = set()
    for row in t256_bindings:
        allocation_id = int(row.get("allocation_id", "0"))
        allocation = allocation_by_id.get(allocation_id)
        if not allocation:
            fail(
                f"{variant} repeat {run['repeat']} has a T256 binding without "
                "a physical allocation"
            )
        t256_allocation_ids.add(allocation_id)
        if row.get("live") != "1" or int(row.get("used_calls", "0")) <= 0:
            fail(
                f"{variant} repeat {run['repeat']} has a dead T256 logical binding"
            )
        if (
            allocation["storage_kind"] != "f16"
            or allocation["in_dim"] != "8192"
            or allocation["out_dim"] != "4096"
            or int(allocation["logical_aliases"]) != 1
            or int(allocation["live_aliases"]) != 1
            or int(allocation["used_calls"]) <= 0
            or int(allocation["dead_bytes"]) != 0
        ):
            fail(
                f"{variant} repeat {run['repeat']} T256 allocation "
                f"{allocation_id} is not one live physical weight"
            )
    expected_physical_t256 = 0 if variant == "native" else 43
    physical_t256_rows = [
        row for row in allocation_rows
        if row["storage_kind"] == "f16"
        and row["in_dim"] == "8192"
        and row["out_dim"] == "4096"
    ]
    if (
        len(t256_allocation_ids) != expected_physical_t256
        or len(physical_t256_rows) != expected_physical_t256
    ):
        fail(
            f"{variant} repeat {run['repeat']} has "
            f"{len(physical_t256_rows)} physical T256 allocations and "
            f"{len(t256_allocation_ids)} referenced, expected "
            f"{expected_physical_t256}"
        )

    audit_rows = [row for row in rows(Path(run["audit"])) if is_t256(row)]
    if not audit_rows:
        fail(f"{variant} repeat {run['repeat']} has no T256 runtime evidence")
    result_counts = Counter(row["result"] for row in audit_rows)
    expected_results = (
        {"f16_hit"} | ({"f16_partner_hit"} if partner else set())
        if variant == "overflow"
        else EXPECTED_RESULTS[variant]
    )
    if set(result_counts) != expected_results:
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
    expected_partner_layers = (
        binding_partner_layers
        if variant == "overflow"
        else {
            "native": [],
            "all-local": [],
            "balanced": list(range(1, 43, 2)),
            "all-partner": list(range(43)),
        }[variant]
    )
    if partner_layers != expected_partner_layers:
        fail(
            f"{variant} repeat {run['repeat']} partner layers are "
            f"{partner_layers}, expected {expected_partner_layers}"
        )
    for row in audit_rows:
        layer = layer_of(row)
        home, peer = devices_for_layer(layer)
        uses_partner = (
            layer in binding_partner_layer_set
            if variant == "overflow"
            else partner_layer(variant, layer)
        )
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
    if variant != "native" and "T256-output_b=43/43" not in log:
        fail(f"{variant} repeat {run['repeat']} did not admit 43/43 T256 bindings")

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
        "physical_allocations": len(allocation_rows),
        "physical_resident_bytes": sum(
            int(row["resident_bytes"]) for row in allocation_rows
        ),
        "physical_dead_bytes": sum(
            int(row["dead_bytes"]) for row in allocation_rows
        ),
        "t256_physical_allocations": len(t256_allocation_ids),
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
        or meta.get("overflow_partner_eligibility") != "15-21"
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

    overflow_observed = [
        {
            "repeat": item["repeat"],
            "local_bindings": item["t256_local_bindings"],
            "partner_bindings": item["t256_partner_bindings"],
            "partner_layers": item["partner_layers"],
        }
        for item in evidence
        if item["variant"] == "overflow"
    ]
    overflow_signatures = {
        (
            item["local_bindings"], item["partner_bindings"],
            tuple(item["partner_layers"]),
        )
        for item in overflow_observed
    }
    overflow_placement_stable = len(overflow_signatures) == 1
    high_confidence = overflow_placement_stable and len(repeats) >= 5 and {
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
        "overflow_placement_stable": overflow_placement_stable,
        "overflow_observed": overflow_observed,
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
        "Natural-overflow placement stability: "
        f"**{'PASS' if overflow_placement_stable else 'FAIL'}**.",
        "",
        "Binding inventory and active runtime paths were validated independently for every run.",
        "",
        "Observed natural-overflow placement (local/partner bindings): "
        + "; ".join(
            f"r{item['repeat']}={item['local_bindings']}/{item['partner_bindings']} "
            f"layers={item['partner_layers']}"
            for item in overflow_observed
        )
        + ".",
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
        "43 active T256 weights local. The exported binding and allocation inventories "
        "record that resource "
        "tradeoff; this is an end-to-end cache-policy comparison, not an isolated kernel test.",
        "",
    ))
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
