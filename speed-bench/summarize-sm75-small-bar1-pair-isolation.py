#!/usr/bin/env python3

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path


VARIANT_ORDER = (
    "partner-off", "indexer-off", "both-off", "admission-off", "production"
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
    production = outcomes.get("production", "not-run")
    partner = outcomes.get("partner-off", "not-run")
    indexer = outcomes.get("indexer-off", "not-run")
    both = outcomes.get("both-off", "not-run")
    admission = outcomes.get("admission-off", "not-run")
    if production != "failed":
        return (
            "The full-production failure was not reproduced, so this pass cannot "
            "identify a necessary trigger. Preserve the raw traffic and phase evidence."
        )
    if partner == "passed" and indexer != "passed":
        return (
            "Pair-0 partner execution is necessary for the reproduced failure. "
            "Because pair-0 partner weights and scratch remained admitted in that "
            "passing arm, peer-cache admission alone is not sufficient."
        )
    if partner == "failed" and admission == "passed" and indexer != "passed":
        return (
            "Pair-0 execution-off still failed with peer weights/scratch admitted, "
            "while pair-0 admission-off passed. The differentiating trigger is peer "
            "cache/scratch admission (or its materialization), not decode-indexer rows."
        )
    if indexer == "passed" and partner != "passed":
        return (
            "Pair-0 decode-indexer row transfers are necessary for the reproduced "
            "failure; partner execution without those row transfers survived."
        )
    if partner == "passed" and indexer == "passed":
        return (
            "Removing either pair-0 runtime transfer class prevents the reproduced "
            "failure. The evidence identifies a shared P2P/interaction requirement, "
            "not one uniquely guilty feature; admission alone survived."
        )
    if both == "passed":
        return (
            "Neither runtime path alone was sufficient to prevent failure, but removing "
            "both did. This implicates their aggregate/interaction load; admitted peer "
            "weights remained present and therefore are not sufficient by themselves."
        )
    failed_both = [row for row in rows
                   if row["variant"] == "both-off" and row["status"] != "passed"]
    if failed_both and all(row["last_phase"] in {
            "engine-create", "session-create", "untimed-warmup", "measured-prefill"
            } for row in failed_both):
        return (
            "The pair still failed before decode with both pair-0 runtime features "
            "disabled. Decode-indexer row transfers are excluded. Peer-cache admission "
            "or another unchanged prefill/P2P path remains; an admission-off confirmation "
            "would be required to distinguish those."
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
        nvlink_path = root / "nvlink" / f"{stem}.log"
        nvlink_text = (nvlink_path.read_text(errors="replace")
                       if nvlink_path.exists() else "")
        q8_begin_calls, q8_begin_bytes = checkpoint_max(
            log_text, "q8 partner transfer audit", "begin", 0
        )
        q8_complete_calls, q8_complete_bytes = checkpoint_max(
            log_text, "q8 partner transfer audit", "complete", 0
        )
        row_begin_calls, row_begin_bytes = indexer_totals(log_text, "begin", 0)
        row_complete_calls, row_complete_bytes = indexer_totals(log_text, "complete", 0)
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
            "nvlink_counter_snapshots": str(nvlink_text.count("snapshot_utc=")),
            "nvlink_counter_supported": (
                "no" if "counter_status=unsupported-or-unavailable" in nvlink_text
                else ("yes" if "snapshot_utc=" in nvlink_text else "unknown")
            ),
            "result": str(result_path),
            "log": str(log_path),
            "nvlink_log": str(nvlink_path),
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
        "`GPU_DEVICES=0,3,1,2` layout. `partner-off` suppresses execution only "
        "after retaining the 344/344 admission plan; `admission-off` is the "
        "separate materialization control.",
        "",
        "| Variant | Outcome | Prefill tok/s | Decode tok/s | Last phase | Last event | "
        "Pair-0 Q8 begun bytes* | Pair-0 indexer begun bytes | NVLink snapshots |",
        "| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['status']} | {row.get('prefill_tps', '')} | "
            f"{row.get('decode_tps', '')} | {row['last_phase']} | {row['last_event']} | "
            f"{row.get('pair0_q8_begin_checkpoint_bytes', '0')} | "
            f"{row.get('pair0_indexer_begin_bytes', '0')} | "
            f"{row.get('nvlink_counter_snapshots', '0')} |"
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
        "Raw `nvidia-smi nvlink -g 0/1` samples are retained under `nvlink/`; "
        "unsupported counters are explicitly labeled rather than treated as zero traffic.",
    ])
    (root / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
