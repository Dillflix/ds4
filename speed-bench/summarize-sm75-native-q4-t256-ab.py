#!/usr/bin/env python3
"""Summarize standard/native-Q4 with the same all-partner T256 policy."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


VARIANTS = (
    "standard-all-partner",
    "native-all-partner",
)
REQUIRED_RUN_COLUMNS = {
    "repeat",
    "slot",
    "variant",
    "model",
    "policy",
    "csv",
    "log",
    "audit",
    "bindings",
    "cache_before",
    "cache_after",
    "logits",
}
REQUIRED_BENCH_COLUMNS = {"ctx_tokens", "prefill_tokens", "prefill_tps"}
REQUIRED_AUDIT_COLUMNS = {
    "label", "layer", "physical_device", "in_dim", "out_dim", "result", "reason",
}
REQUIRED_BINDING_COLUMNS = {
    "consumer_device", "resident_device", "partner_offload", "weight_offset",
    "in_dim", "out_dim", "label",
}
LONG_CONTEXTS = (16384, 32768)
EFFECT = "native_all_partner_over_standard_all_partner"


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def resolve_path(value: str, base: Path, label: str) -> Path:
    if not value:
        fail(f"empty {label} path in runs.tsv")
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def read_table(
    path: Path,
    delimiter: str = ",",
    required: set[str] | None = None,
) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            fields = set(reader.fieldnames or ())
            missing = (required or set()) - fields
            if missing:
                fail(f"{path} lacks columns: {','.join(sorted(missing))}")
            return list(reader)
    except OSError as exc:
        fail(f"cannot read {path}: {exc}")


def parse_positive_int(value: str, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        fail(f"invalid integer for {label}: {value!r}")
    if parsed <= 0:
        fail(f"{label} must be positive, found {parsed}")
    return parsed


def parse_positive_float(value: str, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        fail(f"invalid floating-point value for {label}: {value!r}")
    if not math.isfinite(parsed) or parsed <= 0.0:
        fail(f"{label} must be positive and finite, found {value!r}")
    return parsed


def files_equal(left: Path, right: Path) -> bool:
    try:
        if left.stat().st_size != right.stat().st_size:
            return False
        with left.open("rb") as lhs, right.open("rb") as rhs:
            while True:
                left_chunk = lhs.read(1024 * 1024)
                right_chunk = rhs.read(1024 * 1024)
                if left_chunk != right_chunk:
                    return False
                if not left_chunk:
                    return True
    except OSError as exc:
        fail(f"cannot compare raw logits {left} and {right}: {exc}")


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def median_absolute_deviation(values: list[float]) -> float:
    median = statistics.median(values)
    return statistics.median(abs(value - median) for value in values)


def is_t256(row: dict[str, str]) -> bool:
    return (
        row.get("in_dim") == "8192"
        and row.get("out_dim") == "4096"
        and "attn_output_b" in row.get("label", "")
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


def format_number(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.12g}"
    return value


def main() -> int:
    if len(sys.argv) != 3:
        fail(
            "usage: summarize-sm75-native-q4-t256-ab.py "
            "RUNS.tsv OUTPUT_DIR"
        )

    runs_path = Path(sys.argv[1]).resolve()
    out_dir = Path(sys.argv[2]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = read_table(runs_path, "\t", REQUIRED_RUN_COLUMNS)
    if not runs:
        fail("runs.tsv is empty")

    run_index: dict[tuple[int, str], dict[str, object]] = {}
    slot_index: dict[int, dict[int, str]] = defaultdict(dict)
    model_values: dict[str, set[str]] = {"standard": set(), "native": set()}
    samples: dict[tuple[int, str], dict[int, tuple[int, float]]] = {}
    expected_policy = {variant: "all-partner" for variant in VARIANTS}
    base_dir = runs_path.parent

    for source_row in runs:
        repeat = parse_positive_int(source_row["repeat"], "repeat")
        slot = parse_positive_int(source_row["slot"], "slot")
        variant = source_row["variant"]
        if variant not in VARIANTS:
            fail(f"unsupported variant in runs.tsv: {variant!r}")
        if source_row["policy"] != expected_policy[variant]:
            fail(
                f"variant {variant} requires policy={expected_policy[variant]}, "
                f"found {source_row['policy']!r}"
            )
        key = (repeat, variant)
        if key in run_index:
            fail(f"duplicate run for repeat={repeat} variant={variant}")
        if slot in slot_index[repeat]:
            fail(f"duplicate slot for repeat={repeat} slot={slot}")

        family = variant.split("-", 1)[0]
        model_values[family].add(source_row["model"])
        paths = {
            name: resolve_path(source_row[name], base_dir, name)
            for name in (
                "csv", "log", "audit", "bindings", "cache_before",
                "cache_after", "logits",
            )
        }
        if not paths["logits"].is_dir():
            fail(f"logits path is not a directory: {paths['logits']}")

        bench_rows = read_table(paths["csv"], ",", REQUIRED_BENCH_COLUMNS)
        if not bench_rows:
            fail(f"benchmark CSV is empty: {paths['csv']}")
        values: dict[int, tuple[int, float]] = {}
        for bench_row in bench_rows:
            context = parse_positive_int(
                bench_row["ctx_tokens"],
                f"repeat {repeat} {variant} ctx_tokens",
            )
            if context in values:
                fail(
                    f"repeat {repeat} variant {variant} contains duplicate "
                    f"context {context}"
                )
            work = parse_positive_int(
                bench_row["prefill_tokens"],
                f"repeat {repeat} {variant} context {context} prefill_tokens",
            )
            tps = parse_positive_float(
                bench_row["prefill_tps"],
                f"repeat {repeat} {variant} context {context} prefill_tps",
            )
            values[context] = (work, tps)

        row: dict[str, object] = dict(source_row)
        row.update(paths)
        row["repeat"] = repeat
        row["slot"] = slot
        run_index[key] = row
        slot_index[repeat][slot] = variant
        samples[key] = values

    repeats = sorted({repeat for repeat, _ in run_index})
    if not repeats:
        fail("the integration check contains no repeats")
    if repeats != list(range(1, len(repeats) + 1)):
        fail(f"repeats must be consecutive from 1, found {repeats}")
    expected_keys = {
        (repeat, variant) for repeat in repeats for variant in VARIANTS
    }
    if set(run_index) != expected_keys:
        fail("runs.tsv must contain every variant exactly once per repeat")
    if any(len(values) != 1 for values in model_values.values()):
        fail("each standard/native model family must use one stable model path")
    standard_model = next(iter(model_values["standard"]))
    native_model = next(iter(model_values["native"]))
    if standard_model == native_model:
        fail("standard and native variants must name distinct model paths")

    expected_slots = set(range(1, len(VARIANTS) + 1))
    for repeat in repeats:
        if set(slot_index[repeat]) != expected_slots:
            fail(f"repeat {repeat} must contain slots 1..2 exactly once")
        if set(slot_index[repeat].values()) != set(VARIANTS):
            fail(f"repeat {repeat} must contain every variant exactly once")
        order = tuple(slot_index[repeat][slot] for slot in sorted(expected_slots))
        if order != VARIANTS:
            fail(f"repeat {repeat} order is {order}, expected {VARIANTS}")

    baseline = samples[(repeats[0], "standard-all-partner")]
    contexts = sorted(baseline)
    if not set(LONG_CONTEXTS).issubset(contexts):
        fail(
            f"benchmark must include 16K and 32K frontiers; found {contexts}"
        )
    expected_work = {context: baseline[context][0] for context in contexts}
    for (repeat, variant), values in samples.items():
        if set(values) != set(contexts):
            fail(
                f"repeat {repeat} variant {variant} has a mismatched frontier "
                f"set: {sorted(values)} versus {contexts}"
            )
        for context in contexts:
            work = values[context][0]
            if work != expected_work[context]:
                fail(
                    f"repeat {repeat} variant {variant} context {context} has "
                    f"mismatched prefill work: {work} versus "
                    f"{expected_work[context]}"
                )

    expected_logit_names = {
        f"frontier_{context:06d}.logits.f32" for context in contexts
    }
    raw_logits: dict[tuple[int, str, int], Path] = {}
    full_vocab_bytes: int | None = None
    for key, run in run_index.items():
        directory = run["logits"]
        assert isinstance(directory, Path)
        actual_names = {path.name for path in directory.glob("*.logits.f32")}
        if actual_names != expected_logit_names:
            repeat, variant = key
            fail(
                f"repeat {repeat} variant {variant} raw-logit frontier set "
                f"differs: {sorted(actual_names)} versus "
                f"{sorted(expected_logit_names)}"
            )
        for context in contexts:
            path = directory / f"frontier_{context:06d}.logits.f32"
            try:
                byte_count = path.stat().st_size
            except OSError as exc:
                fail(f"cannot stat raw logits {path}: {exc}")
            if byte_count <= 0 or byte_count % 4 != 0:
                fail(
                    f"raw full-vocabulary logits must contain nonempty float32 "
                    f"bytes: {path} has {byte_count} bytes"
                )
            if full_vocab_bytes is None:
                full_vocab_bytes = byte_count
            elif byte_count != full_vocab_bytes:
                fail(
                    f"raw full-vocabulary logit size differs: {path} has "
                    f"{byte_count} bytes, expected {full_vocab_bytes}"
                )
            raw_logits[(key[0], key[1], context)] = path

    exactness_rows: list[dict[str, object]] = []
    cross_layout_exact = True
    for repeat in repeats:
        for context in contexts:
            reference = raw_logits[(repeat, "standard-all-partner", context)]
            candidate = raw_logits[(repeat, "native-all-partner", context)]
            exact = files_equal(reference, candidate)
            cross_layout_exact = cross_layout_exact and exact
            exactness_rows.append({
                "kind": "cross-layout",
                "comparison": "standard-vs-native-all-partner",
                "variant": "all-partner",
                "repeat": repeat,
                "reference_repeat": repeat,
                "ctx_tokens": context,
                "reference": reference,
                "candidate": candidate,
                "reference_bytes": reference.stat().st_size,
                "candidate_bytes": candidate.stat().st_size,
                "exact": int(exact),
            })

    repeat_deterministic = True
    reference_repeat = repeats[0]
    for variant in VARIANTS:
        for repeat in repeats[1:]:
            for context in contexts:
                reference = raw_logits[(reference_repeat, variant, context)]
                candidate = raw_logits[(repeat, variant, context)]
                exact = files_equal(reference, candidate)
                repeat_deterministic = repeat_deterministic and exact
                exactness_rows.append({
                    "kind": "repeat-determinism",
                    "comparison": f"repeat-{reference_repeat}-vs-{repeat}",
                    "variant": variant,
                    "repeat": repeat,
                    "reference_repeat": reference_repeat,
                    "ctx_tokens": context,
                    "reference": reference,
                    "candidate": candidate,
                    "reference_bytes": reference.stat().st_size,
                    "candidate_bytes": candidate.stat().st_size,
                    "exact": int(exact),
                })

    policy_evidence_ok = True
    cache_stable = True
    run_evidence: list[dict[str, object]] = []
    native_marker = "SM75 native routed-Q4 layout enabled"
    planner_marker = "CUDA q8 fp16 benefit plan"
    partner_summary_marker = "CUDA q8 fp16 partner summary:"
    for repeat in repeats:
        for variant in VARIANTS:
            run = run_index[(repeat, variant)]
            log_path = run["log"]
            audit_path = run["audit"]
            bindings_path = run["bindings"]
            cache_before = run["cache_before"]
            cache_after = run["cache_after"]
            assert all(
                isinstance(path, Path)
                for path in (
                    log_path, audit_path, bindings_path,
                    cache_before, cache_after,
                )
            )
            try:
                log_text = log_path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                fail(f"cannot read {log_path}: {exc}")
            audit_rows = read_table(
                audit_path, ",", REQUIRED_AUDIT_COLUMNS
            )
            binding_rows = read_table(
                bindings_path, ",", REQUIRED_BINDING_COLUMNS
            )
            for cache_path in (cache_before, cache_after):
                try:
                    if cache_path.stat().st_size == 0:
                        fail(f"cache-state evidence is empty: {cache_path}")
                except OSError as exc:
                    fail(f"cannot stat cache-state evidence {cache_path}: {exc}")
            t256_audit = [row for row in audit_rows if is_t256(row)]
            t256_bindings = [row for row in binding_rows if is_t256(row)]
            fixed_bindings = [
                row for row in t256_bindings if row["partner_offload"] == "0"
            ]
            partner_bindings = [
                row for row in t256_bindings if row["partner_offload"] == "1"
            ]
            non_t256_bindings = [row for row in binding_rows if not is_t256(row)]
            family = variant.split("-", 1)[0]
            planner_ok = planner_marker in log_text
            layout_ok = (
                (family == "native" and native_marker in log_text)
                or (family == "standard" and native_marker not in log_text)
            )
            expected_bindings: Counter[tuple[int, int, int, int]] = Counter()
            for layer in range(43):
                home, peer = devices_for_layer(layer)
                expected_bindings[(layer, peer, peer, 0)] += 1
                expected_bindings[(layer, home, peer, 1)] += 1
            observed_bindings = Counter(
                (
                    layer_of(row), int(row["consumer_device"]),
                    int(row["resident_device"]), int(row["partner_offload"]),
                )
                for row in t256_bindings
            )
            unique_t256_allocations = {
                (row["resident_device"], row["weight_offset"])
                for row in t256_bindings
            }
            audit_layers = Counter(layer_of(row) for row in t256_audit)
            even_layer_coverage = (
                set(audit_layers) == set(range(43))
                and len(set(audit_layers.values())) == 1
            )
            audit_mapping_ok = all(
                row["result"] == "f16_partner_hit"
                and row["reason"] == "nvlink_offload"
                and int(row["physical_device"])
                == devices_for_layer(layer_of(row))[1]
                for row in t256_audit
            )
            policy_ok = (
                "partner-classes=t256" in log_text
                and "partner-layers=0-42" in log_text
                and "t256-placement=all-partner" in log_text
                and "T256-output_b=86/86" in log_text
                and "partner=43 partner-arithmetic=f16" in log_text
                and partner_summary_marker in log_text
                and len(fixed_bindings) == 43
                and len(partner_bindings) == 43
                and len(unique_t256_allocations) == 43
                and observed_bindings == expected_bindings
                and len(non_t256_bindings) >= 263
                and all(row["partner_offload"] == "0" for row in non_t256_bindings)
                and bool(t256_audit)
                and even_layer_coverage
                and audit_mapping_ok
            )
            stable = files_equal(cache_before, cache_after)
            cache_stable = cache_stable and stable
            run_ok = planner_ok and layout_ok and policy_ok
            policy_evidence_ok = policy_evidence_ok and run_ok
            run_evidence.append({
                "repeat": repeat,
                "variant": variant,
                "planner": planner_ok,
                "layout_dispatch": layout_ok,
                "policy": policy_ok,
                "partner_audit_calls": len(t256_audit),
                "fixed_t256_bindings": len(fixed_bindings),
                "partner_bindings": len(partner_bindings),
                "unique_t256_allocations": len(unique_t256_allocations),
                "non_t256_bindings": len(non_t256_bindings),
                "cache_stable": stable,
            })

    paired_rows: list[dict[str, object]] = []
    effect_values: dict[int, list[float]] = {
        context: [] for context in contexts
    }
    for repeat in repeats:
        for context in contexts:
            standard = samples[(repeat, "standard-all-partner")][context][1]
            native = samples[(repeat, "native-all-partner")][context][1]
            ratio = native / standard
            effect_values[context].append(ratio)
            paired_rows.append({
                "repeat": repeat,
                "ctx_tokens": context,
                "prefill_tokens": expected_work[context],
                "standard_all_partner_tps": standard,
                "native_all_partner_tps": native,
                EFFECT: ratio,
            })

    context_rows: list[dict[str, object]] = []
    context_payload: list[dict[str, object]] = []
    native_no_regression = True
    for context in contexts:
        values = effect_values[context]
        median = statistics.median(values)
        gate_pass = median >= 1.0
        native_no_regression = native_no_regression and gate_pass
        context_rows.append({
            "ctx_tokens": context,
            "effect": EFFECT,
            "samples": len(values),
            "median": median,
            "min": min(values),
            "max": max(values),
            "mad": median_absolute_deviation(values),
            "required_gate": ">=1.00",
            "pass": int(gate_pass),
        })
        context_payload.append({
            "ctx_tokens": context,
            "prefill_tokens": expected_work[context],
            "standard_median_tps": statistics.median(
                samples[(repeat, "standard-all-partner")][context][1]
                for repeat in repeats
            ),
            "native_median_tps": statistics.median(
                samples[(repeat, "native-all-partner")][context][1]
                for repeat in repeats
            ),
            "native_over_standard": median,
            "native_no_regression_pass": gate_pass,
        })

    gates = {
        "cross_layout_raw_logits_exact": cross_layout_exact,
        "repeat_deterministic_if_repeated": repeat_deterministic,
        "focused_standard_then_native_order": True,
        "policy_and_dispatch_evidence": policy_evidence_ok,
        "cache_state_stable": cache_stable,
        "native_all_partner_no_regression": native_no_regression,
    }
    accepted = all(gates.values())

    paired_fields = [
        "repeat", "ctx_tokens", "prefill_tokens",
        "standard_all_partner_tps", "native_all_partner_tps", EFFECT,
    ]
    write_csv(
        out_dir / "paired-effects.csv",
        paired_fields,
        [
            {key: format_number(value) for key, value in row.items()}
            for row in paired_rows
        ],
    )
    write_csv(
        out_dir / "context-summary.csv",
        [
            "ctx_tokens", "effect", "samples", "median", "min", "max",
            "mad", "required_gate", "pass",
        ],
        [
            {key: format_number(value) for key, value in row.items()}
            for row in context_rows
        ],
    )
    write_csv(
        out_dir / "exactness.csv",
        [
            "kind", "comparison", "variant", "repeat", "reference_repeat",
            "ctx_tokens", "reference", "candidate", "reference_bytes",
            "candidate_bytes", "exact",
        ],
        exactness_rows,
    )

    payload = {
        "accepted": accepted,
        "scope": {
            "variants": list(VARIANTS),
            "repeats": len(repeats),
            "contexts": contexts,
            "standard_model": standard_model,
            "native_model": native_model,
            "official_quality_suite": False,
        },
        "gates": gates,
        "contexts": context_payload,
        "evidence": {
            "raw_full_vocab_bytes": full_vocab_bytes,
            "cross_layout_comparisons":
                len(repeats) * len(contexts),
            "repeat_determinism_comparisons":
                len(VARIANTS) * (len(repeats) - 1) * len(contexts),
            "runs": run_evidence,
        },
        "interpretation": (
            "Both arms use the identical all-partner T256 policy, so the paired "
            "ratio isolates the tagged native-Q4 layout within the final combined "
            "configuration. Acceptance covers integration and throughput only; the separate "
            "official T256 quality-isolation gate still applies."
        ),
    }
    (out_dir / "acceptance.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# SM75 native-Q4 + all-partner T256 integration",
        "",
        f"Overall: **{'PASS' if accepted else 'FAIL'}**",
        "",
        (
            f"Scope: {len(repeats)} repeats, {len(contexts)} context "
            "frontiers, two focused variants."
        ),
        "",
        "This result covers dispatch integration and throughput only. The "
        "separate official T256 quality-isolation gate still applies.",
        "",
        "Both arms force the measured all-partner T256 winner. The ratio therefore "
        "measures only the effect of switching the routed Q4 tensors to the tagged "
        "SM75-native layout.",
        "",
        "| Context | Standard all-partner | Native all-partner | Native / standard |",
        "|---:|---:|---:|---:|",
    ]
    for item in context_payload:
        lines.append(
            f"| {int(item['ctx_tokens']) // 1024}K | "
            f"{float(item['standard_median_tps']):.2f} | "
            f"{float(item['native_median_tps']):.2f} | "
            f"{float(item['native_over_standard']):.4f}x |"
        )
    lines.extend((
        "",
        "## Acceptance evidence",
        "",
        f"- Standard/native raw full-vocabulary logits: "
        f"{'PASS' if cross_layout_exact else 'FAIL'}",
        f"- Per-variant repeat determinism: "
        f"{'PASS' if repeat_deterministic else 'FAIL'}"
        f" ({'not applicable with one repeat' if len(repeats) == 1 else 'evaluated'}).",
        "- Focused standard-then-native order: PASS",
        f"- Native layout and all-partner T256 policy evidence: "
        f"{'PASS' if policy_evidence_ok else 'FAIL'}",
        f"- Cache state frozen during timing: "
        f"{'PASS' if cache_stable else 'FAIL'}",
        f"- Native all-partner median never regresses versus standard all-partner: "
        f"{'PASS' if native_no_regression else 'FAIL'}",
        "",
    ))
    summary = "\n".join(lines)
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
