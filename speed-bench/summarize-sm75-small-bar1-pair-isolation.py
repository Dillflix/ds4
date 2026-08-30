#!/usr/bin/env python3

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path


VARIANT_ORDER = (
    "attention-off", "partner-bounce", "bounce-indexer-off", "partner-serialized",
    "indexer-off", "production"
)


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def read_kv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def checkpoint_max(text: str, family: str, event: str, pair: int) -> tuple[int, int]:
    pattern = re.compile(
        rf"{re.escape(family)} event={event} .*?home_tier={pair} .*?"
        rf"calls=(\d+) bytes=(\d+)"
    )
    calls = 0
    traffic = 0
    for match in pattern.finditer(text):
        calls = max(calls, int(match.group(1)))
        traffic = max(traffic, int(match.group(2)))
    return calls, traffic


def indexer_totals(text: str, event: str, pair: int) -> tuple[int, int]:
    count = 0
    traffic = 0
    for line in text.splitlines():
        if (f"decode indexer row audit event={event}" not in line or
                f"home_tier={pair} " not in line):
            continue
        count += 1
        match = re.search(r"transfer_bytes=(\d+)", line)
        if match:
            traffic += int(match.group(1))
    return count, traffic


def outcome(statuses: list[str]) -> str:
    if not statuses:
        return "not-run"
    if all(status == "passed" for status in statuses):
        return "passed"
    if any(status in {"failed", "interrupted-prior-run"} for status in statuses):
        return "failed"
    return "incomplete"


def inference(outcomes: dict[str, str], rows: list[dict[str, str]]) -> str:
    if any("power-limit-drift" in row.get("watch_status", "") for row in rows):
        return (
            "At least one arm changed power limit after its 250 W preflight. That arm "
            "is invalid for causal comparison; identify the external power-limit writer "
            "and rerun it."
        )
    attention = outcomes.get("attention-off", "not-run")
    production = outcomes.get("production", "not-run")
    bounce = outcomes.get("partner-bounce", "not-run")
    bounce_indexer = outcomes.get("bounce-indexer-off", "not-run")
    serialized = outcomes.get("partner-serialized", "not-run")
    indexer = outcomes.get("indexer-off", "not-run")
    if attention == "passed":
        return (
            "Pair 0 survived with only its prefill attention row split "
            "disabled while full dense admission, direct partner projections, and "
            "both prefill/decode indexer splits remained. Against the reproduced "
            "attention-rows-on failures, "
            "this makes pair-0 prefill row splitting or its interaction a necessary "
            "condition in this pass."
        )
    if attention == "failed":
        return (
            "Pair 0 still failed with its prefill attention row split disabled. "
            "That split is therefore not necessary for the failure; the remaining "
            "common path is full partner projection work/traffic, aggregate load, "
            "power delivery, or the shared PCIe/root-complex path."
        )
    if production in {"not-run", "incomplete"}:
        return (
            "The production control has no durable outcome yet. Resume the same "
            "directory; earlier failures remain evidence, but causal comparison "
            "requires the final control."
        )
    if production == "passed":
        return (
            "The full-production failure was not reproduced, so this pass cannot "
            "identify a necessary trigger. Preserve the raw traffic and phase evidence."
        )
    if indexer == "passed" and bounce_indexer == "passed":
        return (
            "Removing pair-0 decode-indexer row splitting prevented the reproduced "
            "failure under both direct-peer and host-bounce Q8 transport. This makes "
            "the indexer path or its interaction a necessary condition in this pass."
        )
    if bounce == "passed" and serialized == "passed":
        return (
            "The direct overlapped production path failed while host-staged transport "
            "and serialized direct-peer execution survived. Partner weights and cuBLAS "
            "work were retained, narrowing the trigger to direct P2P/BAR1 traffic, "
            "overlap, or the higher instantaneous load they create."
        )
    if bounce == "passed" and serialized != "passed":
        return (
            "Host-staged pair-0 Q8 transport survived while direct production failed. "
            "This implicates the direct P2P/BAR1 path or its timing envelope, but the "
            "slower host-staged schedule is not a power-matched proof by itself."
        )
    if serialized == "passed" and bounce != "passed":
        return (
            "Serializing otherwise unchanged pair-0 peer projections survived while "
            "production failed. This implicates overlap, concurrency, or an instantaneous "
            "power/traffic transient; direct peer transport alone was not sufficient."
        )
    if indexer == "passed":
        return (
            "Disabling only pair-0 decode-indexer row splitting survived while the "
            "production path failed. The indexer path or its interaction is necessary "
            "under direct transport, pending the host-bounce factorial arm."
        )
    if all(outcomes.get(variant) == "failed" for variant in VARIANT_ORDER):
        return (
            "Every workload-preserving arm failed. The evidence does not isolate direct "
            "P2P, indexer rows, or overlap; partner computation, aggregate pair load, "
            "power delivery, or the shared PCIe/root-complex path remains."
        )
    return (
        "The completed arms do not yet isolate a necessary trigger. Resume the same "
        "directory until all requested variants have durable results."
    )


def main() -> None:
    if len(sys.argv) != 2:
        fail("usage: summarize-sm75-small-bar1-pair-isolation.py OUTPUT_DIR")
    root = Path(sys.argv[1]).resolve()
    production = root / "production"
    if not production.is_dir():
        fail(f"production directory not found: {production}")

    rows: list[dict[str, str]] = []
    statuses: defaultdict[str, list[str]] = defaultdict(list)
    for result_path in sorted(production.glob("*.result")):
        values = read_kv(result_path)
        variant = values.get("variant", "unknown")
        status = values.get("status", "unknown")
        stem = result_path.name.removesuffix(".result")
        log_path = production / f"{stem}.log"
        log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
        bench_path = production / f"{stem}.csv"
        bench_row: dict[str, str] = {}
        if bench_path.exists():
            with bench_path.open(newline="") as handle:
                bench_rows = list(csv.DictReader(handle))
            if bench_rows:
                bench_row = bench_rows[-1]
        watch_path = root / "telemetry" / f"{stem}-watch-event.txt"
        watch_values = read_kv(watch_path) if watch_path.exists() else {}
        q8_begin_calls, q8_begin_bytes = checkpoint_max(
            log_text, "q8 partner transfer audit", "begin", 0
        )
        q8_complete_calls, q8_complete_bytes = checkpoint_max(
            log_text, "q8 partner transfer audit", "complete", 0
        )
        row_begin_calls, row_begin_bytes = indexer_totals(log_text, "begin", 0)
        row_complete_calls, row_complete_bytes = indexer_totals(log_text, "complete", 0)
        transport_modes = sorted(set(re.findall(
            r"q8 partner transfer audit event=begin home_tier=0 .*?transport=(\S+)",
            log_text,
        )))
        serialized_modes = sorted(set(re.findall(
            r"q8 partner transfer audit event=begin home_tier=0 .*?serialized=(\S+)",
            log_text,
        )))
        row = {
            "variant": variant,
            "status": status,
            "exit_status": values.get("exit_status", ""),
            "last_phase": values.get("last_phase", ""),
            "last_event": values.get("last_event", ""),
            "last_current": values.get("last_current", ""),
            "last_total": values.get("last_total", ""),
            "prefill_tps": bench_row.get("prefill_tps", ""),
            "decode_tps": bench_row.get("gen_tps", ""),
            "pair0_q8_begin_checkpoint_calls": str(q8_begin_calls),
            "pair0_q8_begin_checkpoint_bytes": str(q8_begin_bytes),
            "pair0_q8_complete_checkpoint_calls": str(q8_complete_calls),
            "pair0_q8_complete_checkpoint_bytes": str(q8_complete_bytes),
            "pair0_indexer_begin_calls": str(row_begin_calls),
            "pair0_indexer_begin_bytes": str(row_begin_bytes),
            "pair0_indexer_complete_calls": str(row_complete_calls),
            "pair0_indexer_complete_bytes": str(row_complete_bytes),
            "pair0_q8_transport": ",".join(transport_modes),
            "pair0_q8_serialized": ",".join(serialized_modes),
            "watch_status": watch_values.get("status", ""),
            "result": str(result_path),
            "log": str(log_path),
        }
        rows.append(row)
        statuses[variant].append(status)

    order = {variant: index for index, variant in enumerate(VARIANT_ORDER)}
    rows.sort(key=lambda row: (order.get(row["variant"], 99), row["result"]))
    fieldnames = list(rows[0]) if rows else [
        "variant", "status", "exit_status", "last_phase", "last_event"
    ]
    with (root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    outcomes = {variant: outcome(statuses.get(variant, []))
                for variant in VARIANT_ORDER}
    conclusion = inference(outcomes, rows)
    lines = [
        "# SM75 small-BAR1 pair isolation",
        "",
        "Pair 0 is logical tier 0<->2, physical GPU 0<->1 in the required "
        "`GPU_DEVICES=0,3,1,2` layout. Every arm retains the 344/344 admission "
        "plan, partner-resident weights, and partner projection arithmetic. The "
        "diagnostics vary only pair-0 Q8 transport/synchronization, pair-0 "
        "prefill attention rows, and pair-0 prefill/decode-indexer rows.",
        "",
        "| Variant | Outcome | Prefill tok/s | Decode tok/s | Last phase | Last event | "
        "Q8 transport | Serialized | Pair-0 Q8 begun bytes* | "
        "Pair-0 indexer begun bytes | Watch event |",
        "| --- | --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['status']} | {row.get('prefill_tps', '')} | "
            f"{row.get('decode_tps', '')} | {row['last_phase']} | {row['last_event']} | "
            f"{row.get('pair0_q8_transport', '')} | "
            f"{row.get('pair0_q8_serialized', '')} | "
            f"{row.get('pair0_q8_begin_checkpoint_bytes', '0')} | "
            f"{row.get('pair0_indexer_begin_bytes', '0')} | "
            f"{row.get('watch_status', '')} |"
        )
    lines.extend([
        "",
        "\\* Q8 byte counters are durable 64-call checkpoints, not a final total if "
        "the device was lost between checkpoints. Indexer bytes are logged per dispatch.",
        "",
        "## Current inference",
        "",
        conclusion,
        "",
        "No NVLink counter command is run by this harness. Pair traffic is measured "
        "only from the in-process Q8 and indexer transfer-audit records.",
    ])
    (root / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
