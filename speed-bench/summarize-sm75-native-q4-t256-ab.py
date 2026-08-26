#!/usr/bin/env python3
"""Summarize the four-way standard/native-Q4 x local/T256 prefill A/B."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


VARIANTS = (
    "standard-local",
    "standard-auto",
    "native-local",
    "native-auto",
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
REQUIRED_AUDIT_COLUMNS = {"in_dim", "out_dim", "result", "reason"}
REQUIRED_BINDING_COLUMNS = {"partner_offload", "in_dim", "out_dim"}
LONG_CONTEXTS = (16384, 32768)
EFFECTS = (
    "standard_auto_over_standard_local",
    "native_local_over_standard_local",
    "native_auto_over_native_local",
    "native_auto_over_standard_auto",
    "native_auto_over_standard_local",
    "interaction",
)


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
    expected_policy = {variant: variant.rsplit("-", 1)[1] for variant in VARIANTS}
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
    if len(repeats) < 4 or len(repeats) % 4 != 0:
        fail(
            "the A/B requires at least four repeats and a repeat count "
            "that is a multiple of four"
        )
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
            fail(f"repeat {repeat} must contain slots 1..4 exactly once")
        if set(slot_index[repeat].values()) != set(VARIANTS):
            fail(f"repeat {repeat} must contain every variant exactly once")
    for block_start in range(0, len(repeats), 4):
        block = repeats[block_start:block_start + 4]
        for slot in sorted(expected_slots):
            counts = Counter(slot_index[repeat][slot] for repeat in block)
            if counts != Counter({variant: 1 for variant in VARIANTS}):
                fail(
                    f"run order is not counterbalanced: repeats "
                    f"{block[0]}-{block[-1]} slot {slot} must contain every "
                    "variant exactly once"
                )

    baseline = samples[(repeats[0], "standard-local")]
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
        for policy in ("local", "auto"):
            standard_variant = f"standard-{policy}"
            native_variant = f"native-{policy}"
            for context in contexts:
                reference = raw_logits[(repeat, standard_variant, context)]
                candidate = raw_logits[(repeat, native_variant, context)]
                exact = files_equal(reference, candidate)
                cross_layout_exact = cross_layout_exact and exact
                exactness_rows.append({
                    "kind": "cross-layout",
                    "comparison": f"standard-vs-native-{policy}",
                    "variant": policy,
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
            partner_audit = [
                row for row in audit_rows
                if row["result"] == "f16_partner_hit"
                and row["reason"] == "nvlink_offload"
            ]
            partner_bindings = [
                row for row in binding_rows if row["partner_offload"] == "1"
            ]
            policy = expected_policy[variant]
            family = variant.split("-", 1)[0]
            planner_ok = planner_marker in log_text
            layout_ok = (
                (family == "native" and native_marker in log_text)
                or (family == "standard" and native_marker not in log_text)
            )
            if policy == "local":
                policy_ok = (
                    "partner-classes=none" in log_text
                    and partner_summary_marker not in log_text
                    and not partner_audit
                    and not partner_bindings
                )
            else:
                policy_ok = (
                    "partner-classes=t256" in log_text
                    and partner_summary_marker in log_text
                    and bool(partner_audit)
                    and bool(partner_bindings)
                    and all(
                        row["in_dim"] == "8192"
                        and row["out_dim"] == "4096"
                        for row in partner_audit
                    )
                    and all(
                        row["in_dim"] == "8192"
                        and row["out_dim"] == "4096"
                        for row in partner_bindings
                    )
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
                "partner_audit_calls": len(partner_audit),
                "partner_bindings": len(partner_bindings),
                "cache_stable": stable,
            })

    paired_rows: list[dict[str, object]] = []
    effect_values: dict[int, dict[str, list[float]]] = {
        context: {effect: [] for effect in EFFECTS} for context in contexts
    }
    for repeat in repeats:
        for context in contexts:
            standard_local = samples[(repeat, "standard-local")][context][1]
            standard_auto = samples[(repeat, "standard-auto")][context][1]
            native_local = samples[(repeat, "native-local")][context][1]
            native_auto = samples[(repeat, "native-auto")][context][1]
            effects = {
                "standard_auto_over_standard_local":
                    standard_auto / standard_local,
                "native_local_over_standard_local":
                    native_local / standard_local,
                "native_auto_over_native_local": native_auto / native_local,
                "native_auto_over_standard_auto": native_auto / standard_auto,
                "native_auto_over_standard_local": native_auto / standard_local,
                "interaction":
                    (native_auto / native_local)
                    / (standard_auto / standard_local),
            }
            for effect, value in effects.items():
                effect_values[context][effect].append(value)
            paired_rows.append({
                "repeat": repeat,
                "ctx_tokens": context,
                "prefill_tokens": expected_work[context],
                "standard_local_tps": standard_local,
                "standard_auto_tps": standard_auto,
                "native_local_tps": native_local,
                "native_auto_tps": native_auto,
                **effects,
            })

    context_rows: list[dict[str, object]] = []
    context_payload: list[dict[str, object]] = []
    t256_long_context_ok = True
    combined_no_regression = True
    for context in contexts:
        medians: dict[str, float] = {}
        for effect in EFFECTS:
            values = effect_values[context][effect]
            median = statistics.median(values)
            medians[effect] = median
            required_gate = ""
            gate_pass: bool | None = None
            if effect == "native_auto_over_native_local" and context in LONG_CONTEXTS:
                required_gate = ">=1.05"
                gate_pass = median >= 1.05
                t256_long_context_ok = t256_long_context_ok and gate_pass
            elif effect == "native_auto_over_standard_local":
                required_gate = ">=1.00"
                gate_pass = median >= 1.0
                combined_no_regression = combined_no_regression and gate_pass
            context_rows.append({
                "ctx_tokens": context,
                "effect": effect,
                "samples": len(values),
                "median": median,
                "min": min(values),
                "max": max(values),
                "mad": median_absolute_deviation(values),
                "required_gate": required_gate,
                "pass": "" if gate_pass is None else int(gate_pass),
            })
        context_payload.append({
            "ctx_tokens": context,
            "prefill_tokens": expected_work[context],
            "median_effects": medians,
            "native_auto_over_native_local_pass":
                None if context not in LONG_CONTEXTS
                else medians["native_auto_over_native_local"] >= 1.05,
            "native_auto_over_standard_local_pass":
                medians["native_auto_over_standard_local"] >= 1.0,
        })

    gates = {
        "cross_layout_raw_logits_exact": cross_layout_exact,
        "repeat_deterministic": repeat_deterministic,
        "counterbalanced_order": True,
        "policy_and_dispatch_evidence": policy_evidence_ok,
        "cache_state_stable": cache_stable,
        "native_auto_over_native_local_16k_32k": t256_long_context_ok,
        "native_auto_over_standard_local_no_regression":
            combined_no_regression,
    }
    accepted = all(gates.values())

    paired_fields = [
        "repeat", "ctx_tokens", "prefill_tokens",
        "standard_local_tps", "standard_auto_tps",
        "native_local_tps", "native_auto_tps", *EFFECTS,
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
                2 * len(repeats) * len(contexts),
            "repeat_determinism_comparisons":
                len(VARIANTS) * (len(repeats) - 1) * len(contexts),
            "runs": run_evidence,
        },
        "interpretation": (
            "Effects are paired ratios. The interaction is measured directly; "
            "component improvements are not assumed to multiply independently. "
            "Acceptance covers integration and throughput only; the separate "
            "official T256 quality-isolation gate still applies."
        ),
    }
    (out_dir / "acceptance.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# SM75 native-Q4 + T256 paired A/B",
        "",
        f"Overall: **{'PASS' if accepted else 'FAIL'}**",
        "",
        (
            f"Scope: {len(repeats)} repeats, {len(contexts)} context "
            "frontiers, four counterbalanced variants."
        ),
        "",
        "This result covers dispatch integration and throughput only. The "
        "separate official T256 quality-isolation gate still applies.",
        "",
        "The effects below are measured paired ratios. Component gains are not "
        "assumed to be additive or multiplicative; `interaction` reports the "
        "observed change in T256's ratio after switching to native Q4.",
        "",
        "| Context | Std auto/local | Native local/std local | "
        "Native auto/local | Native auto/std auto | Combined | Interaction |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in context_payload:
        medians = item["median_effects"]
        assert isinstance(medians, dict)
        lines.append(
            f"| {int(item['ctx_tokens']) // 1024}K | "
            f"{medians['standard_auto_over_standard_local']:.4f}x | "
            f"{medians['native_local_over_standard_local']:.4f}x | "
            f"{medians['native_auto_over_native_local']:.4f}x | "
            f"{medians['native_auto_over_standard_auto']:.4f}x | "
            f"{medians['native_auto_over_standard_local']:.4f}x | "
            f"{medians['interaction']:.4f}x |"
        )
    lines.extend((
        "",
        "## Acceptance evidence",
        "",
        f"- Standard/native raw full-vocabulary logits: "
        f"{'PASS' if cross_layout_exact else 'FAIL'}",
        f"- Per-variant repeat determinism: "
        f"{'PASS' if repeat_deterministic else 'FAIL'}",
        "- Four-way order balance: PASS",
        f"- Native layout and local/T256 policy evidence: "
        f"{'PASS' if policy_evidence_ok else 'FAIL'}",
        f"- Cache state frozen during timing: "
        f"{'PASS' if cache_stable else 'FAIL'}",
        f"- Native T256/local median >= 1.05 at 16K and 32K: "
        f"{'PASS' if t256_long_context_ok else 'FAIL'}",
        f"- Native-auto/standard-local median never regresses: "
        f"{'PASS' if combined_no_regression else 'FAIL'}",
        "",
    ))
    summary = "\n".join(lines)
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
