#!/usr/bin/env python3
"""Validate all-native Q8 versus the complete production FP16-cache policy."""

from __future__ import annotations

import csv
import json
import math
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path


CASES = 100
LAYERS = 43
T256_BINDINGS = 43
T256_LOCAL_BINDINGS = 0
T256_PARTNER_BINDINGS = 43
T256_UNIQUE_ALLOCATIONS = 43
T256_RUNTIME_CALLS = CASES * LAYERS
T256_PARTNER_CALLS = CASES * T256_PARTNER_BINDINGS
EXPECTED_PARTNER_LAYERS = list(range(LAYERS))
BINDING_COLUMNS = {
    "consumer_device", "resident_device", "partner_offload", "in_dim",
    "out_dim", "partner_arithmetic", "weight_offset", "weight_bytes",
    "resident_weight_bytes", "label", "allocation_id", "used_calls", "live",
}
ALLOCATION_COLUMNS = {
    "allocation_id", "storage_kind", "half_rounded", "physical_device",
    "weight_offset", "weight_bytes", "in_dim", "out_dim", "resident_bytes",
    "logical_aliases", "live_aliases", "used_calls", "dead_bytes",
    "usage_tracking",
}
AUDIT_COLUMNS = {
    "module", "label", "layer", "physical_device", "in_dim", "out_dim",
    "result", "reason",
}
SCORE_COLUMNS = {
    "id", "target_tokens", "nll", "avg_nll", "first_match", "greedy_lcp",
    "api_target_tokens", "api_target_mae", "api_top1_count",
    "api_top1_match", "api_topn_ref", "api_topn_hit", "api_pair_total",
    "api_pair_agree",
}


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


def require_columns(path: Path, rows: list[dict[str, str]], required: set[str]) -> None:
    fields = set(rows[0]) if rows else set()
    if not rows:
        # Empty binding exports are valid, so recover their header separately.
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                fields = set(next(csv.reader(handle)))
        except (OSError, StopIteration) as exc:
            fail(f"cannot inspect header in {path}: {exc}")
    missing = required - fields
    if missing:
        fail(f"{path} lacks columns: {','.join(sorted(missing))}")


def read_manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        fail(f"cannot read {path}: {exc}")
    for line in lines:
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
    if row.get("layer", "").isdigit():
        return int(row["layer"])
    match = re.search(r"blk\.(\d+)\.", row.get("label", ""))
    if not match:
        fail(f"cannot determine layer for binding {row.get('label', '<missing>')}")
    return int(match.group(1))


def devices_for_layer(layer: int) -> tuple[int, int]:
    return (0, 1) if layer <= 21 else (3, 2)


def non_t256_class(row: dict[str, str]) -> str:
    label = row.get("label", "")
    for needle, name in (
        ("attn_kv", "attn_kv"),
        ("attn_output_a", "attn_output_a"),
        ("attn_q_a", "attn_q_a"),
        ("attn_q_b", "attn_q_b"),
        ("ffn_down_shexp", "shared_down"),
        ("ffn_gate_shexp", "shared_gate"),
        ("ffn_up_shexp", "shared_up"),
    ):
        if needle in label:
            return name
    return "other"


def nonnegative(row: dict[str, str], key: str, context: str) -> int:
    try:
        value = int(row[key])
    except (KeyError, ValueError) as exc:
        fail(f"{context} has invalid {key}: {row.get(key, '<missing>')}")
        raise AssertionError from exc
    if value < 0:
        fail(f"{context} has negative {key}: {value}")
    return value


def expected_storage(row: dict[str, str]) -> tuple[str, int]:
    arithmetic = row["partner_arithmetic"]
    if row["partner_offload"] == "0" or arithmetic == "f16":
        return "f16", 0
    if arithmetic == "native-q8":
        fail(f"expanded binding unexpectedly names native-Q8 storage: {row['label']}")
    return "f32", int(arithmetic in {"w16-x16-sgemm", "w16-x32-sgemm"})


def validate_allocation_liveness(
    path: Path, bindings: list[dict[str, str]], arm: str,
) -> tuple[dict[int, dict[str, str]], dict[str, object]]:
    allocations = read_rows(path)
    require_columns(path, allocations, ALLOCATION_COLUMNS)
    by_id: dict[int, dict[str, str]] = {}
    for row in allocations:
        allocation_id = nonnegative(row, "allocation_id", f"{arm} allocation")
        if allocation_id == 0 or allocation_id in by_id:
            fail(f"{arm} has invalid or duplicate allocation_id={allocation_id}")
        if row["storage_kind"] not in {"f16", "f32"}:
            fail(f"{arm} allocation {allocation_id} has unsupported storage_kind")
        if row["usage_tracking"] != "1":
            fail(f"{arm} allocation {allocation_id} lacks usage tracking")
        if nonnegative(row, "logical_aliases", arm) == 0:
            fail(f"{arm} allocation {allocation_id} has no logical alias")
        if nonnegative(row, "live_aliases", arm) == 0:
            fail(f"{arm} allocation {allocation_id} has no live alias")
        if nonnegative(row, "used_calls", arm) == 0:
            fail(f"{arm} allocation {allocation_id} was never used")
        if nonnegative(row, "resident_bytes", arm) == 0:
            fail(f"{arm} allocation {allocation_id} has no resident payload")
        if nonnegative(row, "dead_bytes", arm) != 0:
            fail(f"{arm} allocation {allocation_id} contains dead expanded-weight bytes")
        by_id[allocation_id] = row

    aliases: Counter[int] = Counter()
    live_aliases: Counter[int] = Counter()
    used_calls: Counter[int] = Counter()
    for binding in bindings:
        context = f"{arm} binding {binding.get('label', '<missing>')}"
        allocation_id = nonnegative(binding, "allocation_id", context)
        if allocation_id == 0 or allocation_id not in by_id:
            fail(f"{context} has no matching expanded-weight allocation")
        if binding["live"] != "1" or nonnegative(binding, "used_calls", context) == 0:
            fail(f"{context} was exported but never used")
        allocation = by_id[allocation_id]
        storage, half_rounded = expected_storage(binding)
        expected = {
            "storage_kind": storage,
            "half_rounded": str(half_rounded),
            "physical_device": binding["resident_device"],
            "weight_offset": binding["weight_offset"],
            "weight_bytes": binding["weight_bytes"],
            "in_dim": binding["in_dim"],
            "out_dim": binding["out_dim"],
            "resident_bytes": binding["resident_weight_bytes"],
        }
        wrong = {
            key: (allocation.get(key), value)
            for key, value in expected.items()
            if allocation.get(key) != value
        }
        if wrong:
            fail(f"{context} allocation {allocation_id} does not match: {wrong}")
        aliases[allocation_id] += 1
        live_aliases[allocation_id] += 1
        used_calls[allocation_id] += int(binding["used_calls"])

    if set(by_id) != set(aliases):
        fail(f"{arm} contains expanded-weight allocations with no exported binding")
    for allocation_id, allocation in by_id.items():
        observed = (
            aliases[allocation_id], live_aliases[allocation_id],
            used_calls[allocation_id],
        )
        recorded = (
            int(allocation["logical_aliases"]), int(allocation["live_aliases"]),
            int(allocation["used_calls"]),
        )
        if observed != recorded:
            fail(
                f"{arm} allocation {allocation_id} alias/use totals differ: "
                f"bindings={observed} allocation={recorded}"
            )
    return by_id, {
        "allocations": len(allocations),
        "f16_allocations": sum(row["storage_kind"] == "f16" for row in allocations),
        "f32_allocations": sum(row["storage_kind"] == "f32" for row in allocations),
        "resident_bytes": sum(int(row["resident_bytes"]) for row in allocations),
        "dead_bytes": sum(int(row["dead_bytes"]) for row in allocations),
    }


def dynamic_non_t256_inventory(
    rows: list[dict[str, str]], allocations: dict[int, dict[str, str]],
) -> tuple[dict[str, int], dict[str, int]]:
    classes = Counter(non_t256_class(row) for row in rows)
    descriptors: Counter[str] = Counter()
    for row in rows:
        allocation = allocations[int(row["allocation_id"])]
        descriptor = ";".join((
            f"class={non_t256_class(row)}",
            f"label={row['label']}",
            f"shape={row['in_dim']}x{row['out_dim']}",
            f"consumer={row['consumer_device']}",
            f"resident={row['resident_device']}",
            f"arithmetic={row['partner_arithmetic']}",
            f"storage={allocation['storage_kind']}",
            f"half_rounded={allocation['half_rounded']}",
        ))
        descriptors[descriptor] += 1
    return dict(sorted(classes.items())), dict(sorted(descriptors.items()))


def validate_native(root: Path) -> dict[str, object]:
    binding_path = root / "quality/native-q8.bindings.csv"
    bindings = read_rows(binding_path)
    require_columns(binding_path, bindings, BINDING_COLUMNS)
    if bindings:
        fail(f"native-q8 exported {len(bindings)} FP16 bindings; expected zero")
    allocations, allocation_summary = validate_allocation_liveness(
        root / "quality/native-q8.allocations.csv", bindings, "native-q8"
    )
    if allocations:
        fail(
            f"native-q8 exported {len(allocations)} expanded-weight allocations; "
            "expected zero"
        )

    audit_path = root / "quality/native-q8.q8-audit.csv"
    audit = read_rows(audit_path)
    require_columns(audit_path, audit, AUDIT_COLUMNS)
    t256 = [row for row in audit if is_t256(row)]
    results = Counter(row["result"] for row in t256)
    reasons = Counter(row["reason"] for row in t256)
    layers = Counter(layer_of(row) for row in t256)
    if len(t256) != T256_RUNTIME_CALLS:
        fail(f"native-q8 executed {len(t256)} T256 calls; expected {T256_RUNTIME_CALLS}")
    if results != Counter({"native_q8": T256_RUNTIME_CALLS}):
        fail(f"native-q8 T256 execution is contaminated: {dict(results)}")
    if reasons != Counter({"disabled_by_env": T256_RUNTIME_CALLS}):
        fail(f"native-q8 T256 disable reason is not exact: {dict(reasons)}")
    if layers != Counter({layer: CASES for layer in range(LAYERS)}):
        fail("native-q8 did not execute every T256 layer exactly once per case")
    return {
        "t256_bindings": 0,
        "t256_runtime_calls": len(t256),
        "runtime_results": dict(sorted(results.items())),
        "runtime_reasons": dict(sorted(reasons.items())),
        "expanded_weights": allocation_summary,
    }


def validate_production_fp16_cache(root: Path) -> dict[str, object]:
    binding_path = root / "quality/production-fp16-cache.bindings.csv"
    bindings = read_rows(binding_path)
    require_columns(binding_path, bindings, BINDING_COLUMNS)
    t256 = [row for row in bindings if is_t256(row)]
    non_t256 = [row for row in bindings if not is_t256(row)]
    local = [row for row in t256 if row["partner_offload"] == "0"]
    partner = [row for row in t256 if row["partner_offload"] == "1"]
    if len(t256) != T256_BINDINGS:
        fail(f"full FP16 exported {len(t256)}/{T256_BINDINGS} T256 bindings")
    if len(local) != T256_LOCAL_BINDINGS or len(partner) != T256_PARTNER_BINDINGS:
        fail(
            "full FP16 T256 placement is not 0 local + 43 partner: "
            f"local={len(local)} partner={len(partner)}"
        )
    if any(row["partner_arithmetic"] != "f16" for row in t256):
        fail("full FP16 T256 bindings contain non-F16 arithmetic")
    if any(row["consumer_device"] == row["resident_device"] for row in partner):
        fail("full FP16 partner binding is resident on its consumer")
    partner_layers = sorted(layer_of(row) for row in partner)
    if partner_layers != EXPECTED_PARTNER_LAYERS:
        fail(f"full FP16 partner layers are {partner_layers}, expected 0-42")
    allocation_rows, allocation_summary = validate_allocation_liveness(
        root / "quality/production-fp16-cache.allocations.csv",
        bindings,
        "production FP16 cache",
    )
    unique_allocations = {int(row["allocation_id"]) for row in t256}
    if len(unique_allocations) != T256_UNIQUE_ALLOCATIONS:
        fail(
            "full FP16 T256 bindings do not share exactly 43 physical weights: "
            f"unique={len(unique_allocations)}"
        )
    if any(row["partner_offload"] != "0" for row in non_t256):
        fail("full FP16 contains a non-T256 partner binding")
    non_t256_classes, non_t256_descriptors = dynamic_non_t256_inventory(
        non_t256, allocation_rows
    )
    expected_bindings: Counter[tuple[int, int, int, int]] = Counter()
    for layer in range(LAYERS):
        home_device, partner_device = devices_for_layer(layer)
        expected_bindings[(layer, home_device, partner_device, 1)] += 1
    observed_bindings = Counter(
        (
            layer_of(row), int(row["consumer_device"]),
            int(row["resident_device"]), int(row["partner_offload"]),
        )
        for row in t256
    )
    if observed_bindings != expected_bindings:
        fail("full FP16 T256 per-layer consumer/resident mapping is incorrect")

    audit_path = root / "quality/production-fp16-cache.q8-audit.csv"
    audit = read_rows(audit_path)
    require_columns(audit_path, audit, AUDIT_COLUMNS)
    calls = [row for row in audit if is_t256(row)]
    results = Counter(row["result"] for row in calls)
    expected_results = Counter({
        "f16_partner_hit": T256_PARTNER_CALLS,
    })
    if len(calls) != T256_RUNTIME_CALLS:
        fail(f"full FP16 executed {len(calls)} T256 calls; expected {T256_RUNTIME_CALLS}")
    if results != expected_results:
        fail(f"full FP16 T256 execution is not complete: {dict(results)}")
    partner_hits = [row for row in calls if row["result"] == "f16_partner_hit"]
    if any(row["reason"] != "nvlink_offload" for row in partner_hits):
        fail("full FP16 partner audit contains a non-NVLink reason")
    hit_layers = Counter(layer_of(row) for row in partner_hits)
    if hit_layers != Counter({layer: CASES for layer in EXPECTED_PARTNER_LAYERS}):
        fail(f"full FP16 partner execution does not cover layers 0-42: {dict(hit_layers)}")
    for row in calls:
        layer = layer_of(row)
        _home_device, partner_device = devices_for_layer(layer)
        expected = ("f16_partner_hit", "nvlink_offload", partner_device)
        observed = (
            row["result"], row["reason"], int(row["physical_device"])
        )
        if observed != expected:
            fail(
                f"full FP16 layer {layer} executed {observed}, expected {expected}"
            )
    return {
        "t256_bindings": len(t256),
        "local_bindings": len(local),
        "partner_bindings": len(partner),
        "unique_t256_allocations": len(unique_allocations),
        "non_t256_bindings": len(non_t256),
        "non_t256_class_inventory": non_t256_classes,
        "non_t256_descriptor_inventory": non_t256_descriptors,
        "partner_layers": partner_layers,
        "t256_runtime_calls": len(calls),
        "runtime_results": dict(sorted(results.items())),
        "expanded_weights": allocation_summary,
    }


def read_scores(path: Path) -> dict[str, dict[str, str]]:
    rows = read_rows(path, "\t")
    require_columns(path, rows, SCORE_COLUMNS)
    result = {row["id"]: row for row in rows}
    if len(rows) != CASES or len(result) != CASES:
        fail(f"{path} must contain exactly {CASES} unique cases")
    for case_id, row in result.items():
        if int(row["target_tokens"]) <= 0:
            fail(f"{path} has no target tokens for {case_id}")
        for key in ("nll", "avg_nll", "api_target_mae"):
            if not math.isfinite(float(row[key])):
                fail(f"{path} contains non-finite {key} for {case_id}")
    return result


def isum(rows: dict[str, dict[str, str]], key: str) -> int:
    return sum(int(float(row.get(key, "0") or 0)) for row in rows.values())


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def weighted(rows: dict[str, dict[str, str]], key: str, count: str) -> float | None:
    denominator = isum(rows, count)
    if not denominator:
        return None
    return sum(
        float(row[key]) * int(float(row[count])) for row in rows.values()
    ) / denominator


def quality_summary(rows: dict[str, dict[str, str]]) -> dict[str, object]:
    tokens = sum(int(row["target_tokens"]) for row in rows.values())
    first_matches = isum(rows, "first_match")
    return {
        "cases": len(rows),
        "target_tokens": tokens,
        "avg_nll": sum(float(row["nll"]) for row in rows.values()) / tokens,
        "first_matches": first_matches,
        "first_match_rate": first_matches / len(rows),
        "avg_greedy_lcp": statistics.fmean(
            float(row["greedy_lcp"]) for row in rows.values()
        ),
        "api_target_mae": weighted(rows, "api_target_mae", "api_target_tokens"),
        "api_top1_rate": ratio(isum(rows, "api_top1_match"), isum(rows, "api_top1_count")),
        "api_topn_recall": ratio(isum(rows, "api_topn_hit"), isum(rows, "api_topn_ref")),
        "api_pair_rate": ratio(isum(rows, "api_pair_agree"), isum(rows, "api_pair_total")),
    }


def bootstrap_delta(
    ids: list[str], native: dict[str, dict[str, str]], candidate: dict[str, dict[str, str]],
    draws: int = 10000,
) -> tuple[float, float, float]:
    rng = random.Random(0x75_86)
    samples: list[float] = []
    for _ in range(draws):
        nll_delta = 0.0
        tokens = 0
        for _ in ids:
            case_id = ids[rng.randrange(len(ids))]
            nll_delta += float(candidate[case_id]["nll"]) - float(native[case_id]["nll"])
            tokens += int(native[case_id]["target_tokens"])
        samples.append(nll_delta / tokens)
    samples.sort()
    lower = samples[math.floor(0.025 * draws)]
    upper = samples[math.ceil(0.975 * draws) - 1]
    upper_one_sided = samples[math.ceil(0.95 * draws) - 1]
    return lower, upper, upper_one_sided


def fmt(value: object, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def validate_log(path: Path, required: list[str], forbidden: list[str]) -> None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        fail(f"cannot read {path}: {exc}")
    for marker in required:
        if marker not in text:
            fail(f"{path} lacks runtime marker: {marker}")
    for marker in forbidden:
        if marker in text:
            fail(f"{path} contains forbidden runtime marker: {marker}")


def main() -> int:
    coverage_only = len(sys.argv) == 4 and sys.argv[1] == "--coverage-only"
    if len(sys.argv) != 2 and not coverage_only:
        fail(
            "usage: summarize-q8-fp16-full-quality.py OUTPUT_DIR\n"
            "       summarize-q8-fp16-full-quality.py --coverage-only OUTPUT_DIR ARM"
        )
    root = Path(sys.argv[2] if coverage_only else sys.argv[1]).resolve()
    meta = read_manifest(root / "manifest.txt")
    expected_manifest = {
        "gpu_devices": "0,3,1,2",
        "gpu_vram": "auto",
        "stage_split": "22/21",
        "quality_ctx": "32769",
        "t256_layers": "0-42",
        "comparison": "all-native-q8-vs-complete-production-fp16-cache",
        "native_expected_expanded_bindings": "0",
        "fp16_expected_t256_bindings": "43/43",
        "fp16_expected_placement": "43-partner",
        "fp16_expected_unique_t256_allocations": "43",
        "fp16_non_t256_inventory": "dynamic-production-policy",
        "expanded_weight_liveness": "all-bindings-and-allocations-live",
        "model_hashing": "disabled",
    }
    wrong = {
        key: (meta.get(key), value)
        for key, value in expected_manifest.items()
        if meta.get(key) != value
    }
    if wrong:
        fail(f"manifest does not describe the fixed two-arm comparison: {wrong}")

    if coverage_only:
        arm = sys.argv[3]
        if arm == "native-q8":
            validate_log(
                root / "quality/native-q8.log",
                [
                    "score_official: runtime_path=production",
                    "CUDA EP forced pipeline split 22/21",
                    "t256-placement=overflow",
                ],
                ["CUDA q8 partner execution enabled:"],
            )
            coverage = validate_native(root)
        elif arm == "production-fp16-cache":
            validate_log(
                root / "quality/production-fp16-cache.log",
                [
                    "score_official: runtime_path=production",
                    "CUDA EP forced pipeline split 22/21",
                    "T256-output_b=43/43",
                    "partner=43 partner-arithmetic=f16",
                    "partner-classes=t256",
                    "partner-layers=0-42",
                    "home-order=frozen",
                    "t256-placement=all-partner",
                    "CUDA q8 partner execution enabled:",
                ],
                ["arithmetic=native-q8"],
            )
            coverage = validate_production_fp16_cache(root)
        else:
            fail(f"unknown coverage arm: {arm}")
        print(json.dumps({"arm": arm, "coverage": coverage}, indent=2))
        return 0

    validate_log(
        root / "quality/native-q8.log",
        [
            "score_official: runtime_path=production",
            "CUDA EP forced pipeline split 22/21",
            "t256-placement=overflow",
        ],
        ["CUDA q8 partner execution enabled:"],
    )
    validate_log(
        root / "quality/production-fp16-cache.log",
        [
            "score_official: runtime_path=production",
            "CUDA EP forced pipeline split 22/21",
            "T256-output_b=43/43",
            "partner=43 partner-arithmetic=f16",
            "partner-classes=t256",
            "partner-layers=0-42",
            "home-order=frozen",
            "t256-placement=all-partner",
            "CUDA q8 partner execution enabled:",
        ],
        ["arithmetic=native-q8"],
    )

    native_coverage = validate_native(root)
    candidate_coverage = validate_production_fp16_cache(root)

    planner = (root / "planner-unit.log").read_text(encoding="utf-8", errors="replace")
    gpu_test = (root / "gpu-exactness.log").read_text(encoding="utf-8", errors="replace")
    if "checks passed (0 failed)" not in planner:
        fail("planner regression evidence is missing")
    if "q8 partner projection exactness OK (3 classes)" not in gpu_test:
        fail("local/partner FP16 projection exactness evidence is missing")

    native = read_scores(root / "quality/native-q8.tsv")
    candidate = read_scores(root / "quality/production-fp16-cache.tsv")
    if set(native) != set(candidate):
        fail("quality case IDs differ between arms")
    ids = sorted(native)
    for case_id in ids:
        if native[case_id]["target_tokens"] != candidate[case_id]["target_tokens"]:
            fail(f"target-token count differs for {case_id}")

    native_summary = quality_summary(native)
    candidate_summary = quality_summary(candidate)
    nll_delta = float(candidate_summary["avg_nll"]) - float(native_summary["avg_nll"])
    nll_relative = nll_delta / float(native_summary["avg_nll"])
    ci_lower, ci_upper, upper95 = bootstrap_delta(ids, native, candidate)
    case_deltas = {
        case_id: float(candidate[case_id]["avg_nll"]) - float(native[case_id]["avg_nll"])
        for case_id in ids
    }
    wins = sum(delta < 0 for delta in case_deltas.values())
    losses = sum(delta > 0 for delta in case_deltas.values())
    ties = len(case_deltas) - wins - losses
    first_loss = int(native_summary["first_matches"]) - int(candidate_summary["first_matches"])
    lcp_loss = float(native_summary["avg_greedy_lcp"]) - float(candidate_summary["avg_greedy_lcp"])
    quality_pass = (
        upper95 <= 0.002
        and max(case_deltas.values()) <= 0.05
        and first_loss <= 1
        and lcp_loss <= 0.1
    )
    worst = sorted(case_deltas.items(), key=lambda item: item[1], reverse=True)[:10]

    payload = {
        "experiment_integrity": True,
        "comparison": "all-native Q8 vs complete production FP16-cache policy",
        "coverage": {
            "native_q8": native_coverage,
            "production_fp16_cache": candidate_coverage,
        },
        "quality": {
            "native_q8": native_summary,
            "production_fp16_cache": candidate_summary,
            "delta_nll_per_token": nll_delta,
            "relative_nll_delta": nll_relative,
            "paired_bootstrap_95pct_ci": [ci_lower, ci_upper],
            "paired_bootstrap_one_sided_upper95": upper95,
            "case_wins_losses_ties": {"wins": wins, "losses": losses, "ties": ties},
            "max_case_avg_nll_delta": max(case_deltas.values()),
            "median_case_avg_nll_delta": statistics.median(case_deltas.values()),
            "first_match_loss": first_loss,
            "average_greedy_lcp_loss": lcp_loss,
            "predeclared_noninferiority_pass": quality_pass,
            "gate": {
                "bootstrap_upper95_nll_delta_max": 0.002,
                "max_case_avg_nll_delta_max": 0.05,
                "first_match_loss_max": 1,
                "average_greedy_lcp_loss_max": 0.1,
            },
            "worst_cases_by_avg_nll_delta": [
                {"id": case_id, "delta": delta} for case_id, delta in worst
            ],
        },
    }
    (root / "quality-comparison.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# All-native Q8 vs complete production FP16-cache quality",
        "",
        "Experiment integrity: **PASS**",
        "",
        "This comparison contains only the two decision-relevant endpoints. The native arm "
        "has no expanded-weight cache. The candidate uses the complete production FP16-cache "
        "policy: every active T256 projection executes from one partner-resident F16 weight, "
        "while all dynamically admitted non-T256 FP16 projections remain local.",
        "",
        "## Proven execution coverage",
        "",
        "| Arm | T256 bindings | Local / partner bindings | Physical T256 weights | Runtime path |",
        "|---|---:|---:|---:|---|",
        f"| Native Q8 | 0 / 43 | 0 / 0 | 0 | {T256_RUNTIME_CALLS} native Q8 |",
        f"| Production FP16 cache | 43 / 43 | 0 / 43 | {T256_UNIQUE_ALLOCATIONS} | "
        f"{T256_PARTNER_CALLS} partner FP16 |",
        f"\nDynamic local non-T256 inventory: **{candidate_coverage['non_t256_bindings']} "
        "bindings**; every exported binding/allocation was used and dead expanded-weight "
        "bytes were zero. Exact class and descriptor inventories are recorded in "
        "`quality-comparison.json`.",
        "",
        "## Official-continuation quality",
        "",
        "Lower NLL and API target MAE are better; higher match, recall, and ordering rates are better.",
        "",
        "| Arm | Avg NLL | First match | Avg greedy LCP | API target MAE | API top-1 | API top-N | API pair order |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Native Q8 | {float(native_summary['avg_nll']):.9f} | "
        f"{float(native_summary['first_match_rate']):.1%} | "
        f"{float(native_summary['avg_greedy_lcp']):.3f} | "
        f"{fmt(native_summary['api_target_mae'], 9)} | "
        f"{fmt(native_summary['api_top1_rate'], 6)} | "
        f"{fmt(native_summary['api_topn_recall'], 6)} | "
        f"{fmt(native_summary['api_pair_rate'], 6)} |",
        f"| Production FP16 cache | {float(candidate_summary['avg_nll']):.9f} | "
        f"{float(candidate_summary['first_match_rate']):.1%} | "
        f"{float(candidate_summary['avg_greedy_lcp']):.3f} | "
        f"{fmt(candidate_summary['api_target_mae'], 9)} | "
        f"{fmt(candidate_summary['api_top1_rate'], 6)} | "
        f"{fmt(candidate_summary['api_topn_recall'], 6)} | "
        f"{fmt(candidate_summary['api_pair_rate'], 6)} |",
        "",
        "## Paired result",
        "",
        f"- Delta NLL/token (FP16 - native): **{nll_delta:+.9f}** ({nll_relative:+.3%}).",
        f"- Paired bootstrap 95% CI: **[{ci_lower:+.9f}, {ci_upper:+.9f}]**; "
        f"one-sided upper 95%: **{upper95:+.9f}**.",
        f"- Per-case NLL wins/losses/ties: **{wins}/{losses}/{ties}**.",
        f"- First-match loss: **{first_loss}**; average greedy-LCP loss: **{lcp_loss:+.3f}**.",
        f"- Predeclared non-inferiority gate: **{'PASS' if quality_pass else 'FAIL'}**.",
        "",
        "The gate compares each arm with the official continuations; it does not require the "
        "FP16 arm to reproduce native-Q8 logits or greedy tokens byte-for-byte.",
        "",
    ]
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
