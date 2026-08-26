#!/usr/bin/env python3
"""Validate and summarize native-Q8 versus complete T256 FP16 admission."""

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
T256_BINDINGS = 86
T256_HOME_BINDINGS = 79
T256_PARTNER_BINDINGS = 7
T256_RUNTIME_CALLS = CASES * LAYERS
T256_HOME_CALLS = CASES * (LAYERS - T256_PARTNER_BINDINGS)
T256_PARTNER_CALLS = CASES * T256_PARTNER_BINDINGS
EXPECTED_PARTNER_LAYERS = list(range(15, 22))
BINDING_COLUMNS = {
    "consumer_device", "resident_device", "partner_offload", "in_dim",
    "out_dim", "partner_arithmetic", "label",
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


def validate_native(root: Path) -> dict[str, object]:
    binding_path = root / "quality/native-q8.bindings.csv"
    bindings = read_rows(binding_path)
    require_columns(binding_path, bindings, BINDING_COLUMNS)
    if bindings:
        fail(f"native-q8 exported {len(bindings)} FP16 bindings; expected zero")

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
    }


def validate_full_fp16(root: Path) -> dict[str, object]:
    binding_path = root / "quality/fp16-t256-full.bindings.csv"
    bindings = read_rows(binding_path)
    require_columns(binding_path, bindings, BINDING_COLUMNS)
    t256 = [row for row in bindings if is_t256(row)]
    home = [row for row in t256 if row["partner_offload"] == "0"]
    partner = [row for row in t256 if row["partner_offload"] == "1"]
    if len(t256) != T256_BINDINGS:
        fail(f"full FP16 exported {len(t256)}/86 T256 bindings")
    if len(home) != T256_HOME_BINDINGS or len(partner) != T256_PARTNER_BINDINGS:
        fail(
            "full FP16 T256 placement is not 79 local + 7 partner: "
            f"local={len(home)} partner={len(partner)}"
        )
    if any(row["partner_arithmetic"] != "f16" for row in t256):
        fail("full FP16 T256 bindings contain non-F16 arithmetic")
    if any(row["consumer_device"] == row["resident_device"] for row in partner):
        fail("full FP16 partner binding is resident on its consumer")
    partner_layers = sorted(layer_of(row) for row in partner)
    if partner_layers != EXPECTED_PARTNER_LAYERS:
        fail(f"full FP16 partner layers are {partner_layers}, expected 15-21")
    expected_bindings: Counter[tuple[int, int, int, int]] = Counter()
    for layer in range(LAYERS):
        home_device, partner_device = devices_for_layer(layer)
        expected_bindings[(layer, partner_device, partner_device, 0)] += 1
        if layer in EXPECTED_PARTNER_LAYERS:
            expected_bindings[(layer, home_device, partner_device, 1)] += 1
        else:
            expected_bindings[(layer, home_device, home_device, 0)] += 1
    observed_bindings = Counter(
        (
            layer_of(row), int(row["consumer_device"]),
            int(row["resident_device"]), int(row["partner_offload"]),
        )
        for row in t256
    )
    if observed_bindings != expected_bindings:
        fail("full FP16 T256 per-layer consumer/resident mapping is incorrect")

    audit_path = root / "quality/fp16-t256-full.q8-audit.csv"
    audit = read_rows(audit_path)
    require_columns(audit_path, audit, AUDIT_COLUMNS)
    calls = [row for row in audit if is_t256(row)]
    results = Counter(row["result"] for row in calls)
    expected_results = Counter({
        "f16_hit": T256_HOME_CALLS,
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
        fail(f"full FP16 partner execution does not cover layers 15-21: {dict(hit_layers)}")
    for row in calls:
        layer = layer_of(row)
        home_device, partner_device = devices_for_layer(layer)
        if layer in EXPECTED_PARTNER_LAYERS:
            expected = ("f16_partner_hit", "nvlink_offload", partner_device)
        else:
            expected = ("f16_hit", "resident", home_device)
        observed = (
            row["result"], row["reason"], int(row["physical_device"])
        )
        if observed != expected:
            fail(
                f"full FP16 layer {layer} executed {observed}, expected {expected}"
            )
    return {
        "t256_bindings": len(t256),
        "local_bindings": len(home),
        "partner_bindings": len(partner),
        "partner_layers": partner_layers,
        "t256_runtime_calls": len(calls),
        "runtime_results": dict(sorted(results.items())),
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
        "t256_layers": "15-21",
        "comparison": "native-q8-vs-fp16-t256-86-of-86",
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
        elif arm == "fp16-t256-full":
            validate_log(
                root / "quality/fp16-t256-full.log",
                [
                    "score_official: runtime_path=production",
                    "CUDA EP forced pipeline split 22/21",
                    "T256-output_b=86/86",
                    "partner=7 partner-arithmetic=f16",
                    "partner-classes=t256",
                    "partner-layers=15-21",
                    "home-order=frozen",
                    "t256-placement=overflow",
                    "CUDA q8 partner execution enabled:",
                ],
                ["arithmetic=native-q8"],
            )
            coverage = validate_full_fp16(root)
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
        root / "quality/fp16-t256-full.log",
        [
            "score_official: runtime_path=production",
            "CUDA EP forced pipeline split 22/21",
            "T256-output_b=86/86",
            "partner=7 partner-arithmetic=f16",
            "partner-classes=t256",
            "partner-layers=15-21",
            "home-order=frozen",
            "t256-placement=overflow",
            "CUDA q8 partner execution enabled:",
        ],
        ["arithmetic=native-q8"],
    )

    native_coverage = validate_native(root)
    candidate_coverage = validate_full_fp16(root)

    planner = (root / "planner-unit.log").read_text(encoding="utf-8", errors="replace")
    gpu_test = (root / "gpu-exactness.log").read_text(encoding="utf-8", errors="replace")
    if "checks passed (0 failed)" not in planner:
        fail("planner regression evidence is missing")
    if "q8 partner projection exactness OK (3 classes)" not in gpu_test:
        fail("local/partner FP16 projection exactness evidence is missing")

    native = read_scores(root / "quality/native-q8.tsv")
    candidate = read_scores(root / "quality/fp16-t256-full.tsv")
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
        "comparison": "pure native Q8 vs complete 86/86 T256 FP16 admission",
        "coverage": {
            "native_q8": native_coverage,
            "fp16_t256_full": candidate_coverage,
        },
        "quality": {
            "native_q8": native_summary,
            "fp16_t256_full": candidate_summary,
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
        "# Native Q8 vs complete T256 FP16 quality",
        "",
        "Experiment integrity: **PASS**",
        "",
        "This comparison contains only the two decision-relevant endpoints. The native arm "
        "has no FP16 expansion bindings; the candidate admits every T256 projection, "
        "including the seven partner-resident projections.",
        "",
        "## Proven execution coverage",
        "",
        "| Arm | T256 bindings | Local / partner bindings | T256 runtime calls | Runtime path |",
        "|---|---:|---:|---:|---|",
        f"| Native Q8 | 0 / 86 | 0 / 0 | {T256_RUNTIME_CALLS} | {T256_RUNTIME_CALLS} native Q8 |",
        f"| Full FP16 T256 | 86 / 86 | 79 / 7 | {T256_RUNTIME_CALLS} | "
        f"{T256_HOME_CALLS} local FP16 + {T256_PARTNER_CALLS} partner FP16 |",
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
        f"| Full FP16 T256 | {float(candidate_summary['avg_nll']):.9f} | "
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
