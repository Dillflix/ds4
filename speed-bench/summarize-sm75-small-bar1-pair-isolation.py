#!/usr/bin/env python3

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path


VARIANT_ORDER = (
    "attention-off", "attention-host-bounce", "attention-q8-host-bounce",
    "attention-q8-pre-gather-fence", "attention-q8-activation-fence",
    "attention-q8-global-compute-fence", "attention-q8-direct-gather-fence",
    "attention-q8-rows-serialized", "attention-q8-row-compute-off",
    "attention-row-query-shadow", "attention-row-partner-shadow",
    "attention-row-gather-shadow",
    "attention-q8-async-completion",
    "attention-q8-phase-audit",
    "attention-q8-targeted-phase-audit",
    "attention-q8-l14-l15-phase-audit", "attention-q8-l12-phase-audit",
    "attention-query-dst", "attention-gather-dst",
    "attention-both-dst", "attention-phase-audit", "attention-end-fence",
    "attention-row-boundary-audit", "partner-bounce", "bounce-indexer-off",
    "partner-serialized", "indexer-off", "production"
)

Q8_WINDOW_L14_LABEL = "tensor:blk.14.attn_output_b.weight"
Q8_WINDOW_L14_OFFSET = "143571266304"
Q8_WINDOW_L15_LABEL = "tensor:blk.15.attn_output_b.weight"
Q8_WINDOW_L15_OFFSET = "143723876608"
Q8_L12_LABEL = "tensor:blk.12.attn_output_b.weight"
Q8_L12_OFFSET = "143236281600"
Q8_WINDOW_TARGETS = {
    Q8_WINDOW_L14_LABEL: Q8_WINDOW_L14_OFFSET,
    Q8_WINDOW_L15_LABEL: Q8_WINDOW_L15_OFFSET,
}
Q8_WINDOW_CHAIN = (
    ("begin", "activation-prepare"),
    ("activation-complete", "activation-copy"),
    ("pre-compute-sync-begin", "pre-compute-sync"),
    ("pre-compute-complete", "pre-compute-sync"),
    ("compute-submitted", "compute"),
    ("compute-complete", "compute-sync"),
    ("result-complete", "result-gather"),
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


def attention_copy_schedules(text: str, pair: int) -> tuple[str, str]:
    query: set[str] = set()
    gather: set[str] = set()
    pattern = re.compile(
        r"prefill attention row audit dispatch=split .*?"
        rf"home={pair} partner=\d+ .*?query_copy_stream=(\S+) "
        r"gather_copy_stream=(\S+)"
    )
    for match in pattern.finditer(text):
        query.add(match.group(1))
        gather.add(match.group(2))
    return ",".join(sorted(query)), ",".join(sorted(gather))


def attention_copy_transports(text: str, pair: int) -> tuple[str, str]:
    query: set[str] = set()
    gather: set[str] = set()
    pattern = re.compile(
        r"prefill attention row audit dispatch=split .*?"
        rf"home={pair} partner=\d+ .*?query_copy_transport=(\S+) "
        r"gather_copy_transport=(\S+)"
    )
    for match in pattern.finditer(text):
        query.add(match.group(1))
        gather.add(match.group(2))
    return ",".join(sorted(query)), ",".join(sorted(gather))


def attention_aux_transports(text: str, pair: int) -> tuple[str, str]:
    cache = set(re.findall(
        rf"prefill attention cache mirror transport=(\S+) home_tier={pair} ",
        text,
    ))
    topk: set[str] = set()
    pattern = re.compile(
        r"prefill attention row audit dispatch=split .*?"
        rf"home={pair} partner=\d+ .*?topk_copy_transport=(\S+)"
    )
    for match in pattern.finditer(text):
        topk.add(match.group(1))
    return ",".join(sorted(cache)), ",".join(sorted(topk))


def attention_host_bounce_cache_classes(text: str, pair: int) -> str:
    classes = set(re.findall(
        rf"prefill attention cache mirror transport=host-bounce "
        rf"home_tier={pair} partner_tier=\d+ class=(\S+) event=complete",
        text,
    ))
    return ",".join(sorted(classes))


def last_attention_host_bounce_checkpoint(text: str, pair: int) -> str:
    last = ""
    pattern = re.compile(
        r"prefill attention host-bounce checkpoint event=(\S+) "
        r"kind=(\S+) layer=(\d+) pos=(\d+) tokens=(\d+) "
        rf"home={pair} partner=(\d+)"
    )
    for match in pattern.finditer(text):
        event, kind, layer, pos, tokens, partner = match.groups()
        last = (
            f"{event}:{kind}:layer{layer}:pos{pos}:tokens{tokens}:"
            f"home{pair}:partner{partner}"
        )
    return last


def first_host_bounce_failure_context(text: str) -> str:
    prefixes = (
        "ds4: CUDA prefill attention host-bounce failure ",
        "ds4: CUDA q8 partner host-bounce failure ",
    )
    lines = text.splitlines()
    for index, line in enumerate(lines):
        for prefix in prefixes:
            if line.startswith(prefix):
                context = line.removeprefix(prefix).strip()
                phase = ""
                if index > 0:
                    match = re.search(
                        r"CUDA default-stream bounce (alloc|d2h|h2d) failed:",
                        lines[index - 1],
                    )
                    if match:
                        phase = f"cuda_phase={match.group(1)} "
                return phase + context
    return ""


def _kv_record(line: str, prefix: str) -> dict[str, str]:
    if prefix not in line:
        return {}
    payload = line.split(prefix, 1)[1]
    record: dict[str, str] = {}
    for token in payload.split():
        if "=" in token:
            key, value = token.split("=", 1)
            record[key] = value
    return record


def q8_partner_async_completion_records(
        text: str, kind: str) -> list[dict[str, str]]:
    prefixes = {
        "checkpoint": "q8 partner async completion checkpoint ",
        "failure": "q8 partner async completion failure ",
        "summary": "q8 partner async completion summary ",
    }
    prefix = prefixes[kind]
    return [
        _kv_record(line, prefix) for line in text.splitlines()
        if prefix in line
    ]


def q8_partner_async_completion_enabled(text: str, pair: int = 0) -> bool:
    return (
        "CUDA q8 partner async completion audit enabled: "
        f"logical_pairs={pair} marker=partner-default-stream-mapped-host "
        "event=dedicated-post-marker interpretation=positive-only"
    ) in text


Q8_UINT64_MASK = (1 << 64) - 1


def _q8_u64(record: dict[str, str], key: str) -> int | None:
    raw = record.get(key, "")
    if not re.fullmatch(r"[0-9]+", raw):
        return None
    value = int(raw)
    return value if value <= Q8_UINT64_MASK else None


def q8_partner_async_completion_marker_state(text: str) -> str:
    if not q8_partner_async_completion_enabled(text, 0):
        return "not-enabled-for-pair0"
    failures = q8_partner_async_completion_records(text, "failure")
    if failures:
        return "failure-record-present"
    summaries = q8_partner_async_completion_records(text, "summary")
    if len(summaries) != 1:
        return f"summary-count:{len(summaries)}"
    summary = summaries[0]
    if summary.get("home_tier") != "0" or summary.get("partner_tier") != "2":
        return "wrong-pair"
    if summary.get("partners_synchronized") != "yes":
        return "partner-not-synchronized"
    begun = _q8_u64(summary, "begun")
    submitted = _q8_u64(summary, "submitted")
    confirmed = _q8_u64(summary, "confirmed")
    sequence = _q8_u64(summary, "last_sequence")
    complement = _q8_u64(summary, "last_complement")
    if None in {begun, submitted, confirmed, sequence, complement}:
        return "malformed-summary"
    assert begun is not None and submitted is not None
    assert confirmed is not None and sequence is not None
    assert complement is not None
    if begun <= 0 or begun != submitted or submitted != confirmed:
        return "count-mismatch"
    if sequence != begun:
        return "last-sequence-mismatch"
    if complement != ((~sequence) & Q8_UINT64_MASK):
        return "last-complement-mismatch"
    return "complete"


def q8_partner_async_completion_checkpoint_valid(
        record: dict[str, str]) -> bool:
    if record.get("home_tier") != "0" or record.get("partner_tier") != "2":
        return False
    values = {
        key: _q8_u64(record, key)
        for key in ("begun", "submitted", "confirmed", "sequence", "complement")
    }
    if any(value is None for value in values.values()):
        return False
    sequence = values["sequence"]
    assert sequence is not None
    return (
        sequence > 0 and
        values["begun"] == sequence and
        values["submitted"] == sequence and
        values["confirmed"] == sequence and
        values["complement"] == ((~sequence) & Q8_UINT64_MASK) and
        record.get("evidence") == "post-compute-confirmed"
    )


def q8_partner_async_completion_failure_classification(
        record: dict[str, str]) -> str:
    required = {
        "stage", "current_sequence", "marker_sequence", "marker_complement",
        "marker_matches", "event_status", "interpretation", "begun",
        "submitted", "confirmed", "home_tier", "home_device",
        "partner_tier", "partner_device", "tokens", "in", "out",
        "binding_label", "passed_label", "weight_offset",
    }
    if any(not record.get(key) for key in required):
        return "invalid-failure-record"
    if (record["home_tier"] != "0" or record["home_device"] != "0" or
            record["partner_tier"] != "2" or
            record["partner_device"] != "1"):
        return "invalid-failure-record"
    if record["binding_label"] == "unavailable":
        return "invalid-failure-record"
    numeric_keys = (
        "current_sequence", "marker_sequence", "marker_complement", "begun",
        "submitted", "confirmed", "tokens", "in", "out", "weight_offset",
    )
    values = {key: _q8_u64(record, key) for key in numeric_keys}
    if any(value is None for value in values.values()):
        return "invalid-failure-record"
    current = values["current_sequence"]
    marker_sequence = values["marker_sequence"]
    marker_complement = values["marker_complement"]
    begun = values["begun"]
    submitted = values["submitted"]
    confirmed = values["confirmed"]
    assert current is not None and marker_sequence is not None
    assert marker_complement is not None and begun is not None
    assert submitted is not None and confirmed is not None
    if (current == 0 or begun != current or submitted > begun or
            confirmed > submitted):
        return "invalid-failure-record"
    marker_valid = (
        marker_sequence == current and
        marker_complement == ((~current) & Q8_UINT64_MASK)
    )
    marker_claim = record["marker_matches"]
    if marker_claim not in {"yes", "no"}:
        return "invalid-failure-record"
    if (marker_claim == "yes") != marker_valid:
        return "invalid-failure-record"
    event_complete = record["event_status"] == "complete"
    if event_complete and submitted != current:
        return "invalid-failure-record"
    if marker_valid:
        expected = (
            "post-compute-confirmed" if event_complete else
            "post-compute-marker-positive-event-unavailable"
        )
        classification = (
            "current-call-post-compute-confirmed" if event_complete else
            "current-call-marker-observed-event-unavailable"
        )
    elif event_complete:
        expected = "post-compute-event-confirmed-marker-invalid"
        classification = (
            "current-call-post-compute-event-confirmed-marker-invalid"
        )
    else:
        expected = "inconclusive-no-positive-marker"
        classification = "current-call-inconclusive"
    if record["interpretation"] != expected:
        return "invalid-failure-record"
    return classification


def format_q8_partner_async_completion_failure(
        record: dict[str, str], classification: str) -> str:
    if not record:
        return ""
    keys = (
        "stage", "current_sequence", "marker_sequence", "marker_complement",
        "marker_matches", "event_status", "interpretation", "begun",
        "submitted", "confirmed", "home_tier", "home_device",
        "partner_tier", "partner_device", "binding_label", "weight_offset",
    )
    fields = [f"record_classification={classification}"]
    fields.extend(f"{key}={record.get(key, '')}" for key in keys)
    return " ".join(fields)


def q8_partner_async_completion_summary(
        text: str) -> tuple[str, str, str, str, str, str, str]:
    checkpoints = q8_partner_async_completion_records(text, "checkpoint")
    failures = q8_partner_async_completion_records(text, "failure")
    summaries = q8_partner_async_completion_records(text, "summary")
    checkpoint = checkpoints[-1] if checkpoints else {}
    failure = failures[-1] if failures else {}
    summary = summaries[-1] if summaries else {}
    last_checkpoint = ""
    checkpoint_valid = q8_partner_async_completion_checkpoint_valid(checkpoint)
    if checkpoint_valid:
        last_checkpoint = (
            f"sequence={checkpoint.get('sequence', '')} "
            f"begun={checkpoint.get('begun', '')} "
            f"submitted={checkpoint.get('submitted', '')} "
            f"confirmed={checkpoint.get('confirmed', '')} "
            f"evidence={checkpoint.get('evidence', '')}"
        )
    failure_classification = (
        q8_partner_async_completion_failure_classification(failure)
        if failure else ""
    )
    if failure:
        classification = failure_classification
    elif q8_partner_async_completion_marker_state(text) == "complete":
        classification = "completed-all-confirmed"
    elif checkpoint_valid:
        classification = "prior-calls-confirmed-current-call-unresolved"
    elif checkpoint:
        classification = "invalid-checkpoint-record"
    elif q8_partner_async_completion_enabled(text, 0):
        classification = "armed-no-positive-evidence"
    else:
        classification = ""
    return (
        last_checkpoint,
        format_q8_partner_async_completion_failure(
            failure, failure_classification),
        classification,
        summary.get("begun", failure.get("begun", "")),
        summary.get("submitted", failure.get("submitted", "")),
        summary.get("confirmed", failure.get("confirmed", "")),
        summary.get("partners_synchronized", ""),
    )


def q8_partner_pre_gather_fence_records(
        text: str, kind: str) -> list[dict[str, str]]:
    prefixes = {
        "checkpoint": "q8 partner pre-gather fence checkpoint ",
        "failure": "q8 partner pre-gather fence failure ",
    }
    prefix = prefixes[kind]
    return [
        record for line in text.splitlines()
        if (record := _kv_record(line, prefix))
    ]


def q8_partner_pre_gather_fence_enabled(text: str, pair: int = 0) -> bool:
    return (
        "CUDA q8 partner pre-gather fence audit enabled: "
        f"logical_pairs={pair} boundary=post-marker-event-sync "
        "marker=exact-before-result-d2h"
    ) in text


def q8_partner_pre_gather_armed_records(text: str) -> list[dict[str, str]]:
    prefix = "q8 partner pre-gather armed "
    return [
        _kv_record(line, prefix) for line in text.splitlines()
        if prefix in line
    ]


def q8_partner_pre_gather_armed_record_valid(
        record: dict[str, str]) -> bool:
    required = {
        "current_sequence", "marker_sequence", "marker_complement",
        "home_tier", "home_device", "partner_tier", "partner_device",
    }
    if any(not record.get(key) for key in required):
        return False
    if (record["home_tier"] != "0" or record["home_device"] != "0" or
            record["partner_tier"] != "2" or
            record["partner_device"] != "1"):
        return False
    current = _q8_u64(record, "current_sequence")
    marker_sequence = _q8_u64(record, "marker_sequence")
    marker_complement = _q8_u64(record, "marker_complement")
    if current is None or marker_sequence is None or marker_complement is None:
        return False
    return (
        current > 0 and marker_sequence == current and
        marker_complement == ((~current) & Q8_UINT64_MASK)
    )


def q8_partner_pre_gather_armed_sequence_state(
        records: list[dict[str, str]]) -> str:
    if not records:
        return "armed-count:0"
    if any(not q8_partner_pre_gather_armed_record_valid(record)
           for record in records):
        return "invalid-armed-record"
    sequences = [
        _q8_u64(record, "current_sequence") for record in records
    ]
    if sequences != list(range(1, len(records) + 1)):
        return "armed-sequence-mismatch"
    return "complete"


def format_q8_partner_pre_gather_armed_record(
        record: dict[str, str]) -> str:
    if not record:
        return ""
    keys = (
        "current_sequence", "marker_sequence", "marker_complement",
        "home_tier", "home_device", "partner_tier", "partner_device",
    )
    return " ".join(f"{key}={record.get(key, '')}" for key in keys)


def q8_partner_pre_gather_returned_records(text: str) -> list[dict[str, str]]:
    prefix = "q8 partner pre-gather returned "
    return [
        _kv_record(line, prefix) for line in text.splitlines()
        if prefix in line
    ]


def q8_partner_pre_gather_returned_record_valid(
        record: dict[str, str]) -> bool:
    required = {
        "current_sequence", "result_gather_status", "home_tier",
        "home_device", "partner_tier", "partner_device",
    }
    if any(not record.get(key) for key in required):
        return False
    if (record["result_gather_status"] != "success" or
            record["home_tier"] != "0" or record["home_device"] != "0" or
            record["partner_tier"] != "2" or
            record["partner_device"] != "1"):
        return False
    current = _q8_u64(record, "current_sequence")
    return current is not None and current > 0


def format_q8_partner_pre_gather_returned_record(
        record: dict[str, str]) -> str:
    if not record:
        return ""
    keys = (
        "current_sequence", "result_gather_status", "home_tier",
        "home_device", "partner_tier", "partner_device",
    )
    return " ".join(f"{key}={record.get(key, '')}" for key in keys)


def q8_partner_pre_gather_call_sequence_state(text: str) -> str:
    armed = q8_partner_pre_gather_armed_records(text)
    returned = q8_partner_pre_gather_returned_records(text)
    if not armed and not returned:
        return "empty"
    armed_state = q8_partner_pre_gather_armed_sequence_state(armed)
    if armed_state != "complete":
        return armed_state
    if any(not q8_partner_pre_gather_returned_record_valid(record)
           for record in returned):
        return "invalid-returned-record"
    returned_sequences = [
        _q8_u64(record, "current_sequence") for record in returned
    ]
    if returned_sequences != list(range(1, len(returned) + 1)):
        return "returned-sequence-mismatch"
    if len(returned) > len(armed) or len(armed) - len(returned) > 1:
        return "armed-returned-count-mismatch"

    observed: list[tuple[str, int | None]] = []
    armed_prefix = "q8 partner pre-gather armed "
    returned_prefix = "q8 partner pre-gather returned "
    for line in text.splitlines():
        if armed_prefix in line:
            record = _kv_record(line, armed_prefix)
            observed.append(("armed", _q8_u64(record, "current_sequence")))
        elif returned_prefix in line:
            record = _kv_record(line, returned_prefix)
            observed.append(("returned", _q8_u64(record, "current_sequence")))
    expected: list[tuple[str, int]] = []
    for sequence in range(1, len(returned) + 1):
        expected.extend((("armed", sequence), ("returned", sequence)))
    if len(armed) > len(returned):
        expected.append(("armed", len(armed)))
    if observed != expected:
        return "armed-returned-order-mismatch"
    return "complete"


def q8_partner_compute_fence_marker_state(text: str) -> str:
    enable = (
        "ds4: CUDA q8 partner compute fence audit enabled: logical_pairs=0 "
        "boundary=post-submit-device-sync "
        "scope=every-selected-partner-call identity=dynamic"
    )
    if text.splitlines().count(enable) != 1:
        return "invalid-enable-record"

    prefix = "q8 partner compute fence "
    observed: list[tuple[str, dict[str, str]]] = []
    for line in text.splitlines():
        if line == enable:
            continue
        if prefix not in line:
            continue
        before, _, tail = line.partition(prefix)
        if not before.endswith("CUDA "):
            return "invalid-record-prefix"
        kind, separator, fields = tail.partition(" ")
        if not separator or kind not in {"armed", "returned", "failure"}:
            return "invalid-record-kind"
        observed.append((kind, _kv_record("record " + fields, "record ")))
    if not observed:
        return "empty"
    if any(kind == "failure" for kind, _ in observed):
        return "compute-sync-failure"

    required = {
        "sequence", "status", "home_tier", "home_device",
        "partner_tier", "partner_device", "tokens", "in", "out",
        "binding_label", "passed_label", "weight_offset",
    }
    expected_sequence = 1
    index = 0
    while index < len(observed):
        if index + 1 >= len(observed):
            return "unmatched-armed-record"
        armed_kind, armed = observed[index]
        returned_kind, returned = observed[index + 1]
        if armed_kind != "armed" or returned_kind != "returned":
            return "armed-returned-order-mismatch"
        if any(not armed.get(key) or not returned.get(key) for key in required):
            return "invalid-record"
        armed_without_status = dict(armed)
        returned_without_status = dict(returned)
        armed_status = armed_without_status.pop("status", None)
        returned_status = returned_without_status.pop("status", None)
        if (armed_without_status != returned_without_status or
                armed_status != "submitted" or
                returned_status != "complete"):
            return "armed-returned-identity-mismatch"
        if (_q8_u64(armed, "sequence") != expected_sequence or
                armed["home_tier"] != "0" or armed["home_device"] != "0" or
                armed["partner_tier"] != "2" or
                armed["partner_device"] != "1" or
                armed["binding_label"] == "unavailable" or
                armed["passed_label"] == "unavailable" or
                any((_q8_u64(armed, key) or 0) == 0
                    for key in ("tokens", "in", "out", "weight_offset"))):
            return "invalid-record"
        expected_sequence += 1
        index += 2
    return "complete"


def q8_partner_compute_fence_summary(
        text: str) -> tuple[str, str, str, str, str]:
    records: dict[str, list[str]] = {
        "armed": [], "returned": [], "failure": [],
    }
    for line in text.splitlines():
        for kind in records:
            marker = f"ds4: CUDA q8 partner compute fence {kind} "
            if marker in line:
                records[kind].append(line.strip())
                break
    armed_count = len(records["armed"])
    returned_count = len(records["returned"])
    last_armed = records["armed"][-1] if armed_count else ""
    last_returned = records["returned"][-1] if returned_count else ""
    last_failure = records["failure"][-1] if records["failure"] else ""
    if records["failure"]:
        classification = "compute-synchronize-failed"
    elif armed_count == 0 and returned_count == 0:
        classification = "not-run"
    elif armed_count == returned_count:
        classification = "all-observed-compute-fences-returned"
    elif armed_count == returned_count + 1:
        classification = "compute-fence-return-not-observed"
    else:
        classification = "invalid-compute-fence-record"
    return (
        last_armed, last_returned, last_failure,
        classification, f"{armed_count}/{returned_count}",
    )


def q8_partner_direct_gather_records(
        text: str) -> list[tuple[str, dict[str, str], str]]:
    prefix = "q8 partner direct gather "
    records: list[tuple[str, dict[str, str], str]] = []
    for line in text.splitlines():
        if "q8 partner direct gather audit enabled:" in line:
            continue
        if prefix not in line:
            continue
        before, _, tail = line.partition(prefix)
        if not before.endswith("CUDA "):
            records.append(("invalid", {}, line.strip()))
            continue
        kind, separator, fields = tail.partition(" ")
        if not separator or kind not in {"armed", "returned", "failure"}:
            records.append(("invalid", {}, line.strip()))
            continue
        records.append((
            kind, _kv_record("record " + fields, "record "), line.strip()
        ))
    return records


def q8_partner_direct_gather_record_classification(
        kind: str, record: dict[str, str]) -> str:
    required = {
        "sequence", "status", "result_d2h_attempted",
        "result_d2h_completed", "result_h2d_attempted",
        "result_h2d_completed", "home_tier", "home_device",
        "partner_tier", "partner_device", "tokens", "in", "out",
        "binding_label", "passed_label", "weight_offset",
    }
    if kind not in {"armed", "returned", "failure"} or any(
            not record.get(key) for key in required):
        return "invalid-direct-gather-record"
    if (record["home_tier"] != "0" or record["home_device"] != "0" or
            record["partner_tier"] != "2" or
            record["partner_device"] != "1" or
            record["binding_label"] == "unavailable" or
            record["passed_label"] == "unavailable" or
            any((_q8_u64(record, key) or 0) == 0 for key in (
                "sequence", "tokens", "in", "out", "weight_offset"
            ))):
        return "invalid-direct-gather-record"
    flags = tuple(record[key] for key in (
        "result_d2h_attempted", "result_d2h_completed",
        "result_h2d_attempted", "result_h2d_completed",
    ))
    if any(flag not in {"yes", "no"} for flag in flags):
        return "invalid-direct-gather-record"
    d2h_attempted, d2h_completed, h2d_attempted, h2d_completed = flags
    if ((d2h_attempted == "no" and d2h_completed == "yes") or
            (h2d_attempted == "no" and h2d_completed == "yes") or
            (h2d_attempted == "yes" and d2h_completed != "yes")):
        return "invalid-direct-gather-record"
    if kind == "armed":
        return (
            "post-compute-sync-before-result-d2h"
            if record["status"] == "post-compute-sync" and
            flags == ("no", "no", "no", "no") else
            "invalid-direct-gather-record"
        )
    if kind == "returned":
        return (
            "result-d2h-and-h2d-completed"
            if record["status"] == "copy-complete" and
            flags == ("yes", "yes", "yes", "yes") else
            "invalid-direct-gather-record"
        )
    if record["status"] != "copy-failed":
        return "invalid-direct-gather-record"
    if flags == ("no", "no", "no", "no"):
        return "failure-before-result-d2h-attempt"
    if flags == ("yes", "no", "no", "no"):
        return "result-d2h-attempt-failed"
    if flags == ("yes", "yes", "no", "no"):
        return "result-d2h-complete-before-result-h2d-attempt"
    if flags == ("yes", "yes", "yes", "no"):
        return "result-h2d-attempt-failed"
    if flags == ("yes", "yes", "yes", "yes"):
        return "failure-after-result-h2d-completed"
    return "invalid-direct-gather-record"


def q8_partner_direct_gather_marker_state(text: str) -> str:
    enable = (
        "ds4: CUDA q8 partner direct gather audit enabled: logical_pairs=0 "
        "boundary=compute-sync-to-synchronous-host-bounce "
        "mapped_host_marker=no event=no identity=dynamic"
    )
    if text.splitlines().count(enable) != 1:
        return "invalid-enable-record"
    records = q8_partner_direct_gather_records(text)
    if not records:
        return "empty"
    expected_sequence = 1
    index = 0
    while index < len(records):
        kind, armed, _ = records[index]
        if kind != "armed" or q8_partner_direct_gather_record_classification(
                kind, armed) == "invalid-direct-gather-record":
            return "invalid-armed-record"
        if _q8_u64(armed, "sequence") != expected_sequence:
            return "sequence-mismatch"
        if index + 1 >= len(records):
            return "unmatched-armed-record"
        completed_kind, completed, _ = records[index + 1]
        classification = q8_partner_direct_gather_record_classification(
            completed_kind, completed
        )
        if classification == "invalid-direct-gather-record":
            return "invalid-completion-record"
        identity_keys = (
            "sequence", "home_tier", "home_device", "partner_tier",
            "partner_device", "tokens", "in", "out", "binding_label",
            "passed_label", "weight_offset",
        )
        if any(armed[key] != completed[key] for key in identity_keys):
            return "armed-completion-identity-mismatch"
        if completed_kind == "failure":
            return f"copy-failure:{classification}"
        if completed_kind != "returned":
            return "armed-completion-order-mismatch"
        expected_sequence += 1
        index += 2
    return "complete"


def q8_partner_direct_gather_summary(
        text: str) -> tuple[str, str, str, str, str]:
    records = q8_partner_direct_gather_records(text)
    armed = [raw for kind, _, raw in records if kind == "armed"]
    returned = [raw for kind, _, raw in records if kind == "returned"]
    failures = [
        (record, raw) for kind, record, raw in records if kind == "failure"
    ]
    state = (
        q8_partner_direct_gather_marker_state(text)
        if records or "q8 partner direct gather audit enabled:" in text
        else "not-run"
    )
    classification = state
    if failures and state.startswith("copy-failure:"):
        classification = state.removeprefix("copy-failure:")
    return (
        armed[-1] if armed else "",
        returned[-1] if returned else "",
        failures[-1][1] if failures else "",
        classification,
        f"{len(armed)}/{len(returned)}/{len(failures)}",
    )


def q8_partner_pre_gather_checkpoint_sequence_state(
        text: str, armed_count: int) -> str:
    checkpoints = q8_partner_pre_gather_fence_records(text, "checkpoint")
    if any(
            q8_partner_pre_gather_fence_record_classification(
                record, "checkpoint"
            ) == "invalid-fence-record"
            for record in checkpoints):
        return "invalid-checkpoint-record"
    sequences = [
        _q8_u64(record, "current_sequence") for record in checkpoints
    ]
    expected: list[int] = []
    if armed_count > 0:
        expected.append(1)
        expected.extend(range(64, armed_count + 1, 64))
    next_sequence = armed_count + 1
    trailing_allowed = next_sequence == 1 or next_sequence % 64 == 0
    trailing = trailing_allowed and sequences == expected + [next_sequence]
    if sequences != expected and not trailing:
        return "checkpoint-cadence-mismatch"

    checkpoint_prefix = "q8 partner pre-gather fence checkpoint "
    armed_prefix = "q8 partner pre-gather armed "
    checkpoint_positions: dict[int, int] = {}
    armed_positions: dict[int, int] = {}
    returned_positions: dict[int, int] = {}
    returned_prefix = "q8 partner pre-gather returned "
    for index, line in enumerate(text.splitlines()):
        if checkpoint_prefix in line:
            record = _kv_record(line, checkpoint_prefix)
            sequence = _q8_u64(record, "current_sequence")
            if sequence is not None:
                checkpoint_positions[sequence] = index
        elif armed_prefix in line:
            record = _kv_record(line, armed_prefix)
            sequence = _q8_u64(record, "current_sequence")
            if sequence is not None:
                armed_positions[sequence] = index
        elif returned_prefix in line:
            record = _kv_record(line, returned_prefix)
            sequence = _q8_u64(record, "current_sequence")
            if sequence is not None:
                returned_positions[sequence] = index
    if any(
            sequence not in armed_positions or
            checkpoint_positions.get(sequence, -1) >= armed_positions[sequence] or
            (sequence > 1 and
             checkpoint_positions.get(sequence, -1) <=
             returned_positions.get(sequence - 1, -1))
            for sequence in expected):
        return "checkpoint-order-mismatch"
    if trailing:
        trailing_position = checkpoint_positions.get(next_sequence, -1)
        if (trailing_position < 0 or
                (armed_count > 0 and trailing_position <=
                 returned_positions.get(armed_count, -1)) or
                next_sequence in armed_positions):
            return "checkpoint-order-mismatch"
        return "complete-trailing-checkpoint"
    return "complete"


def q8_partner_pre_gather_failure_order_valid(text: str) -> bool:
    failure_prefix = "q8 partner pre-gather fence failure "
    evidence_prefixes = (
        "q8 partner pre-gather armed ",
        "q8 partner pre-gather returned ",
    )
    lines = text.splitlines()
    failure_positions = [
        index for index, line in enumerate(lines) if failure_prefix in line
    ]
    if len(failure_positions) != 1:
        return False
    evidence_positions = [
        index for index, line in enumerate(lines)
        if any(prefix in line for prefix in evidence_prefixes)
    ]
    return not evidence_positions or failure_positions[0] > max(evidence_positions)


def q8_partner_pre_gather_fence_record_classification(
        record: dict[str, str], kind: str) -> str:
    required = {
        "event", "stage", "current_sequence", "marker_sequence",
        "marker_complement", "marker_matches", "event_status",
        "result_d2h_attempted", "result_d2h_completed",
        "result_h2d_attempted", "result_h2d_completed", "interpretation",
        "attempted",
        "confirmed", "failed", "result_gather_failed_after_confirmed",
        "home_tier", "home_device", "partner_tier", "partner_device",
        "tokens", "in", "out", "binding_label", "passed_label",
        "weight_offset",
    }
    if any(not record.get(key) for key in required):
        return "invalid-fence-record"
    if (record["home_tier"] != "0" or record["home_device"] != "0" or
            record["partner_tier"] != "2" or
            record["partner_device"] != "1" or
            record["binding_label"] == "unavailable" or
            record["passed_label"] == "unavailable"):
        return "invalid-fence-record"
    numeric_keys = (
        "current_sequence", "marker_sequence", "marker_complement",
        "attempted", "confirmed", "failed",
        "result_gather_failed_after_confirmed", "tokens", "in", "out",
        "weight_offset",
    )
    values = {key: _q8_u64(record, key) for key in numeric_keys}
    if any(value is None for value in values.values()):
        return "invalid-fence-record"
    current = values["current_sequence"]
    marker_sequence = values["marker_sequence"]
    marker_complement = values["marker_complement"]
    attempted = values["attempted"]
    confirmed = values["confirmed"]
    failed = values["failed"]
    gather_failed = values["result_gather_failed_after_confirmed"]
    assert current is not None and marker_sequence is not None
    assert marker_complement is not None and attempted is not None
    assert confirmed is not None and failed is not None
    assert gather_failed is not None
    if any(values[key] == 0 for key in ("tokens", "in", "out")):
        return "invalid-fence-record"
    if (current == 0 or attempted == 0 or confirmed > attempted or
            failed > attempted or gather_failed > attempted):
        return "invalid-fence-record"
    marker_valid = (
        marker_sequence == current and
        marker_complement == ((~current) & Q8_UINT64_MASK)
    )
    if record["marker_matches"] not in {"yes", "no"}:
        return "invalid-fence-record"
    if (record["marker_matches"] == "yes") != marker_valid:
        return "invalid-fence-record"

    event = record["event"]
    stage = record["stage"]
    event_status = record["event_status"]
    d2h_attempted = record["result_d2h_attempted"]
    d2h_completed = record["result_d2h_completed"]
    h2d_attempted = record["result_h2d_attempted"]
    h2d_completed = record["result_h2d_completed"]
    interpretation = record["interpretation"]
    if any(value not in {"yes", "no"} for value in (
            d2h_attempted, d2h_completed, h2d_attempted, h2d_completed)):
        return "invalid-fence-record"
    if (d2h_attempted == "no" and d2h_completed == "yes") or \
            (h2d_attempted == "no" and h2d_completed == "yes") or \
            (h2d_attempted == "yes" and d2h_completed != "yes"):
        return "invalid-fence-record"
    if event != "state-invalid" and attempted != current:
        return "invalid-fence-record"
    if kind == "checkpoint":
        valid = (
            event == "complete" and stage == "pre-result-d2h" and
            event_status == "complete" and marker_valid and
            d2h_attempted == "no" and d2h_completed == "no" and
            h2d_attempted == "no" and h2d_completed == "no" and
            interpretation == "post-compute-confirmed-before-result-d2h" and
            confirmed == attempted and failed == 0 and gather_failed == 0
        )
        if not valid:
            return "invalid-fence-record"
        return "post-compute-confirmed-before-result-gather"
    if kind != "failure":
        return "invalid-fence-record"
    if event == "state-invalid":
        valid = (
            stage == "pre-result-d2h" and
            event_status == "not-synchronized" and
            d2h_attempted == "no" and d2h_completed == "no" and
            h2d_attempted == "no" and h2d_completed == "no" and
            interpretation == "failure-surfaced-before-result-d2h" and
            current != attempted and confirmed + 1 == attempted and
            failed == 1 and gather_failed == 0
        )
        return "pre-gather-state-failure" if valid else "invalid-fence-record"
    if event == "sync-failed":
        valid = (
            stage == "pre-result-d2h" and event_status != "complete" and
            d2h_attempted == "no" and d2h_completed == "no" and
            h2d_attempted == "no" and h2d_completed == "no" and
            interpretation == "failure-surfaced-before-result-d2h" and
            confirmed + 1 == attempted and failed == 1 and gather_failed == 0
        )
        if not valid:
            return "invalid-fence-record"
        activation_tag = current | (1 << 63)
        activation_complement = (~activation_tag) & Q8_UINT64_MASK
        if (marker_sequence == current and
                marker_complement == activation_complement and
                record["marker_matches"] == "no"):
            return "mapped-host-marker-partial-write"
        return "pre-gather-stream-failure"
    if event == "marker-invalid":
        valid = (
            stage == "pre-result-d2h" and event_status == "complete" and
            not marker_valid and d2h_attempted == "no" and
            d2h_completed == "no" and
            h2d_attempted == "no" and h2d_completed == "no" and
            interpretation == "post-compute-event-confirmed-marker-invalid" and
            confirmed + 1 == attempted and failed == 1 and gather_failed == 0
        )
        return "marker-channel-failure" if valid else "invalid-fence-record"
    if event == "result-gather-failed":
        copy_boundary_valid = (
            (d2h_attempted == "yes" and d2h_completed == "no" and
             h2d_attempted == "no" and h2d_completed == "no" and
             interpretation ==
             "failure-surfaced-after-confirmed-result-d2h-attempt") or
            (d2h_attempted == "yes" and d2h_completed == "yes" and
             h2d_attempted == "no" and h2d_completed == "no" and
             interpretation ==
             "failure-surfaced-after-confirmed-result-d2h-complete-before-result-h2d") or
            (d2h_attempted == "yes" and d2h_completed == "yes" and
             h2d_attempted == "yes" and h2d_completed == "no" and
             interpretation ==
             "failure-surfaced-after-confirmed-result-h2d-attempt") or
            (d2h_attempted == "yes" and d2h_completed == "yes" and
             h2d_attempted == "yes" and h2d_completed == "yes" and
             interpretation ==
             "failure-surfaced-after-confirmed-result-h2d-complete") or
            (d2h_attempted == "no" and d2h_completed == "no" and
             h2d_attempted == "no" and h2d_completed == "no" and
             interpretation ==
             "failure-surfaced-after-confirmed-before-result-d2h")
        )
        valid = (
            stage == "result-gather" and event_status == "complete" and
            marker_valid and copy_boundary_valid and
            confirmed == attempted and failed == 0 and gather_failed == 1
        )
        if not valid:
            return "invalid-fence-record"
        return (
            "post-compute-confirmed-result-gather-failed-before-d2h-attempt"
            if d2h_attempted == "no" else
            "post-compute-confirmed-result-d2h-failed"
            if d2h_completed == "no" else
            "post-compute-confirmed-result-d2h-complete-result-h2d-not-attempted"
            if h2d_attempted == "no" else
            "post-compute-confirmed-result-h2d-failed"
            if h2d_completed == "no" else
            "post-compute-confirmed-result-h2d-complete-later-gather-failure"
        )
    return "invalid-fence-record"


def format_q8_partner_pre_gather_fence_record(
        record: dict[str, str], classification: str) -> str:
    if not record:
        return ""
    keys = (
        "event", "stage", "current_sequence", "marker_sequence",
        "marker_complement", "marker_matches", "event_status",
        "result_d2h_attempted", "result_d2h_completed",
        "result_h2d_attempted", "result_h2d_completed", "interpretation",
        "attempted", "confirmed",
        "failed", "result_gather_failed_after_confirmed", "home_tier",
        "home_device", "partner_tier", "partner_device", "binding_label",
        "weight_offset",
    )
    fields = [f"record_classification={classification}"]
    fields.extend(f"{key}={record.get(key, '')}" for key in keys)
    return " ".join(fields)


def q8_partner_pre_gather_fence_marker_state(text: str) -> str:
    if not q8_partner_pre_gather_fence_enabled(text, 0):
        return "not-enabled-for-pair0"
    armed = q8_partner_pre_gather_armed_records(text)
    returned = q8_partner_pre_gather_returned_records(text)
    call_state = q8_partner_pre_gather_call_sequence_state(text)
    checkpoint_state = q8_partner_pre_gather_checkpoint_sequence_state(
        text, len(armed)
    )
    failures = q8_partner_pre_gather_fence_records(text, "failure")
    if failures:
        if len(failures) != 1:
            return f"failure-count:{len(failures)}"
        classification = q8_partner_pre_gather_fence_record_classification(
            failures[-1], "failure"
        )
        if call_state not in {"complete", "empty"}:
            return call_state
        if checkpoint_state != "complete":
            return checkpoint_state
        current = _q8_u64(failures[-1], "current_sequence")
        attempted = _q8_u64(failures[-1], "attempted")
        last_armed = len(armed)
        event = failures[-1].get("event", "")
        relationship_valid = (
            attempted == last_armed + 1 and current != attempted and
            len(returned) == last_armed
            if event == "state-invalid"
            else current == last_armed + 1 and len(returned) == last_armed
            if event in {"sync-failed", "marker-invalid"}
            else (current == last_armed and last_armed > 0 and
                  len(returned) == last_armed - 1)
            if event == "result-gather-failed" else False
        )
        if (not relationship_valid or
                not q8_partner_pre_gather_failure_order_valid(text)):
            return "failure-armed-sequence-mismatch"
        return (
            "failure-record-present" if classification != "invalid-fence-record"
            else "invalid-failure-record"
        )
    if call_state != "complete":
        return call_state
    if checkpoint_state != "complete":
        return checkpoint_state
    checkpoints = q8_partner_pre_gather_fence_records(text, "checkpoint")
    if not checkpoints:
        return "checkpoint-count:0"
    if any(
            q8_partner_pre_gather_fence_record_classification(
                record, "checkpoint"
            ) == "invalid-fence-record"
            for record in checkpoints):
        return "invalid-checkpoint-record"
    checkpoint_sequences = [
        _q8_u64(record, "current_sequence") for record in checkpoints
    ]
    if (any(sequence is None for sequence in checkpoint_sequences) or
            checkpoint_sequences != sorted(set(checkpoint_sequences))):
        return "checkpoint-order-mismatch"
    summaries = q8_partner_async_completion_records(text, "summary")
    if len(summaries) != 1:
        return f"summary-count:{len(summaries)}"
    summary = summaries[0]
    if summary.get("home_tier") != "0" or summary.get("partner_tier") != "2":
        return "wrong-pair"
    counts = {
        key: _q8_u64(summary, key)
        for key in (
            "pre_gather_attempted", "pre_gather_confirmed",
            "pre_gather_failed",
            "pre_gather_result_gather_failed_after_confirmed",
            "pre_gather_last_sequence",
        )
    }
    if any(value is None for value in counts.values()):
        return "malformed-summary"
    attempted = counts["pre_gather_attempted"]
    confirmed = counts["pre_gather_confirmed"]
    failed = counts["pre_gather_failed"]
    gather_failed = counts[
        "pre_gather_result_gather_failed_after_confirmed"
    ]
    last_sequence = counts["pre_gather_last_sequence"]
    assert attempted is not None and confirmed is not None
    assert failed is not None and gather_failed is not None
    assert last_sequence is not None
    if (attempted <= 0 or attempted != confirmed or failed != 0 or
            gather_failed != 0 or last_sequence != attempted):
        return "count-mismatch"
    if len(armed) != attempted or len(returned) != attempted:
        return "call-attempted-count-mismatch"
    async_begun = _q8_u64(summary, "begun")
    async_submitted = _q8_u64(summary, "submitted")
    async_confirmed = _q8_u64(summary, "confirmed")
    async_sequence = _q8_u64(summary, "last_sequence")
    if any(value != attempted for value in (
            async_begun, async_submitted, async_confirmed, async_sequence)):
        return "async-fence-count-mismatch"
    expected_checkpoints = [1]
    expected_checkpoints.extend(range(64, attempted + 1, 64))
    if checkpoint_sequences != expected_checkpoints:
        return "checkpoint-cadence-mismatch"
    if q8_partner_async_completion_marker_state(text) != "complete":
        return "async-summary-not-complete"
    return "complete"


def q8_partner_pre_gather_fence_summary(
        text: str, run_status: str = ""
) -> tuple[str, str, str, str, str, str, str, str, str, str, str]:
    if not q8_partner_pre_gather_fence_enabled(text, 0):
        return ("", "", "", "", "", "", "", "", "", "", "")
    checkpoints = q8_partner_pre_gather_fence_records(text, "checkpoint")
    failures = q8_partner_pre_gather_fence_records(text, "failure")
    summaries = q8_partner_async_completion_records(text, "summary")
    armed = q8_partner_pre_gather_armed_records(text)
    returned = q8_partner_pre_gather_returned_records(text)
    checkpoint = checkpoints[-1] if checkpoints else {}
    failure = failures[-1] if failures else {}
    summary = summaries[-1] if summaries else {}
    last_armed = armed[-1] if armed else {}
    last_returned = returned[-1] if returned else {}
    call_state = q8_partner_pre_gather_call_sequence_state(text)
    checkpoint_state = q8_partner_pre_gather_checkpoint_sequence_state(
        text, len(armed)
    )
    checkpoint_classification = (
        q8_partner_pre_gather_fence_record_classification(
            checkpoint, "checkpoint"
        ) if checkpoint else ""
    )
    failure_classification = (
        q8_partner_pre_gather_fence_record_classification(failure, "failure")
        if failure else ""
    )
    corroborated_device_loss = run_status in {
        "failed-device-loss", "interrupted-prior-run-device-loss",
        "interrupted-no-result-device-loss",
    }
    if failure:
        current = _q8_u64(failure, "current_sequence")
        attempted_count = _q8_u64(failure, "attempted")
        event = failure.get("event", "")
        failure_relationship_valid = (
            attempted_count == len(armed) + 1 and
            current != attempted_count and len(returned) == len(armed)
            if event == "state-invalid"
            else current == len(armed) + 1 and len(returned) == len(armed)
            if event in {"sync-failed", "marker-invalid"}
            else (current == len(armed) and bool(armed) and
                  len(returned) == len(armed) - 1)
            if event == "result-gather-failed" else False
        )
        if (len(failures) != 1 or
                call_state not in {"complete", "empty"} or
                checkpoint_state != "complete" or
                not failure_relationship_valid or
                not q8_partner_pre_gather_failure_order_valid(text)):
            classification = "invalid-fence-record"
        else:
            classification = failure_classification
    elif q8_partner_pre_gather_fence_marker_state(text) == "complete":
        classification = "completed-all-pre-gather-confirmed"
    elif (armed or returned) and call_state != "complete":
        classification = "invalid-fence-record"
    elif (corroborated_device_loss and not summary and
          checkpoint_state == "complete-trailing-checkpoint" and
          call_state in {"complete", "empty"}):
        classification = (
            "post-compute-confirmed-before-result-gather-armed-status-not-observed"
        )
    elif checkpoint_state != "complete":
        classification = "invalid-fence-record"
    elif (corroborated_device_loss and
          call_state == "complete" and len(armed) == len(returned) + 1 and
          not summary):
        classification = (
            "post-compute-confirmed-result-gather-return-not-observed"
        )
    elif (corroborated_device_loss and
          call_state == "complete" and bool(armed) and
          len(armed) == len(returned) and not summary):
        classification = "last-confirmed-gather-returned-subsequent-locus-unresolved"
    elif summary:
        classification = "invalid-fence-record"
    elif checkpoint_classification == "post-compute-confirmed-before-result-gather":
        classification = "prior-pre-gather-calls-confirmed-current-call-unresolved"
    elif checkpoint:
        classification = "invalid-fence-record"
    elif q8_partner_pre_gather_fence_enabled(text, 0):
        classification = "armed-no-fence-evidence"
    else:
        classification = ""
    source = summary if summary else (failure if failure else checkpoint)
    attempted = source.get(
        "pre_gather_attempted", source.get("attempted", "")
    )
    confirmed = source.get(
        "pre_gather_confirmed", source.get("confirmed", "")
    )
    failed = source.get("pre_gather_failed", source.get("failed", ""))
    gather_failed = source.get(
        "pre_gather_result_gather_failed_after_confirmed",
        source.get("result_gather_failed_after_confirmed", ""),
    )
    if (not summary and not failure and call_state == "complete" and
            last_armed):
        attempted = last_armed.get("current_sequence", "")
        confirmed = attempted
        failed = "0"
        gather_failed = "0"
    formatted_failure_classification = (
        "invalid-fence-record"
        if failure and classification == "invalid-fence-record"
        else failure_classification
    )
    return (
        format_q8_partner_pre_gather_fence_record(
            checkpoint, checkpoint_classification
        ),
        format_q8_partner_pre_gather_fence_record(
            failure, formatted_failure_classification
        ),
        format_q8_partner_pre_gather_armed_record(last_armed),
        format_q8_partner_pre_gather_returned_record(last_returned),
        classification,
        attempted,
        confirmed,
        failed,
        gather_failed,
        str(len(armed)),
        str(len(returned)),
    )


Q8_PRE_ACTIVATION_MARKER_TAG = 1 << 63
Q8_PRE_ACTIVATION_IDENTITY_KEYS = (
    "home_tier", "home_device", "partner_tier", "partner_device", "bytes",
    "tokens", "in", "out", "binding_label", "passed_label",
    "weight_offset",
)
Q8_PRE_ACTIVATION_ARMED_KEYS = {
    "current_sequence", "marker_sequence", "marker_complement",
    "activation_d2h_status", "destination_sync_status",
    *Q8_PRE_ACTIVATION_IDENTITY_KEYS,
}
Q8_PRE_ACTIVATION_RETURNED_KEYS = {
    "current_sequence", "activation_h2d_status",
    *Q8_PRE_ACTIVATION_IDENTITY_KEYS,
}
Q8_PRE_ACTIVATION_FENCE_KEYS = {
    "event", "stage", "current_sequence", "marker_sequence",
    "marker_complement", "marker_matches", "cuda_status",
    "destination_sync_status", "activation_d2h_attempted",
    "activation_d2h_completed", "activation_h2d_attempted",
    "activation_h2d_completed", "interpretation", "attempted", "confirmed",
    "returned", "failed", *Q8_PRE_ACTIVATION_IDENTITY_KEYS,
}
Q8_PRE_ACTIVATION_SUMMARY_KEYS = {
    "home_tier", "partner_tier", "attempted", "confirmed", "returned",
    "failed", "last_sequence", "shared_slot_sequence",
    "shared_slot_complement", "partners_synchronized",
}


def _q8_pre_activation_records(text: str, suffix: str) -> list[dict[str, str]]:
    prefix = f"ds4: CUDA q8 partner pre-activation {suffix}"
    records: list[dict[str, str]] = []
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        payload = line[len(prefix):]
        record: dict[str, str] = {}
        invalid = False
        for token in payload.split():
            if "=" not in token:
                invalid = True
                continue
            key, value = token.split("=", 1)
            if not key or not value or key in record:
                invalid = True
            record[key] = value
        if invalid:
            record["__invalid__"] = "yes"
        records.append(record)
    return records


def q8_partner_pre_activation_fence_enabled(
        text: str, pair: int = 0) -> bool:
    exact = (
        "ds4: CUDA q8 partner pre-activation fence audit enabled: "
        f"logical_pairs={pair} boundary=post-source-d2h-destination-sync "
        "marker=destination-default-stream-tagged-exact-before-activation-h2d"
    )
    prefix = "ds4: CUDA q8 partner pre-activation fence audit enabled:"
    observed = [line for line in text.splitlines() if line.startswith(prefix)]
    return observed == [exact]


def q8_partner_pre_activation_fence_records(
        text: str, kind: str) -> list[dict[str, str]]:
    suffixes = {
        "checkpoint": "fence checkpoint ",
        "failure": "fence failure ",
        "summary": "fence summary ",
    }
    return _q8_pre_activation_records(text, suffixes[kind])


def q8_partner_pre_activation_armed_records(
        text: str) -> list[dict[str, str]]:
    return _q8_pre_activation_records(text, "armed ")


def q8_partner_pre_activation_returned_records(
        text: str) -> list[dict[str, str]]:
    return _q8_pre_activation_records(text, "returned ")


def q8_partner_pre_activation_record_order_state(text: str) -> str:
    enable_prefix = (
        "ds4: CUDA q8 partner pre-activation fence audit enabled:"
    )
    record_prefixes = {
        "armed": "ds4: CUDA q8 partner pre-activation armed ",
        "returned": "ds4: CUDA q8 partner pre-activation returned ",
        "checkpoint": (
            "ds4: CUDA q8 partner pre-activation fence checkpoint "
        ),
        "failure": "ds4: CUDA q8 partner pre-activation fence failure ",
        "summary": "ds4: CUDA q8 partner pre-activation fence summary ",
    }
    activation_prefix = "ds4: CUDA q8 partner pre-activation "
    positions: dict[str, list[int]] = {
        kind: [] for kind in record_prefixes
    }
    enable_positions: list[int] = []
    for index, line in enumerate(text.splitlines()):
        if line.startswith(enable_prefix):
            enable_positions.append(index)
            continue
        matched = False
        for kind, prefix in record_prefixes.items():
            if line.startswith(prefix):
                positions[kind].append(index)
                matched = True
                break
        if line.startswith(activation_prefix) and not matched:
            return "unknown-activation-record"
    if len(enable_positions) != 1:
        return f"activation-enable-count:{len(enable_positions)}"
    evidence_positions = [
        position for kind_positions in positions.values()
        for position in kind_positions
    ]
    if evidence_positions and min(evidence_positions) < enable_positions[0]:
        return "activation-enable-order-mismatch"
    summary_positions = positions["summary"]
    if (summary_positions and
            summary_positions[-1] != max(evidence_positions)):
        return "activation-summary-order-mismatch"
    return "complete"


def _q8_pre_activation_tagged(sequence: int | None) -> int | None:
    if sequence is None or sequence <= 0 or sequence >= Q8_PRE_ACTIVATION_MARKER_TAG:
        return None
    return sequence | Q8_PRE_ACTIVATION_MARKER_TAG


def _q8_pre_activation_marker_exact(
        current: int | None, sequence: int | None,
        complement: int | None) -> bool:
    tagged = _q8_pre_activation_tagged(current)
    return (
        tagged is not None and sequence == tagged and
        complement == ((~tagged) & Q8_UINT64_MASK)
    )


def q8_partner_pre_activation_identity_valid(
        record: dict[str, str]) -> bool:
    if any(not record.get(key) for key in Q8_PRE_ACTIVATION_IDENTITY_KEYS):
        return False
    if (record["home_tier"] != "0" or record["home_device"] != "0" or
            record["partner_tier"] != "2" or
            record["partner_device"] != "1" or
            record["binding_label"] == "unavailable" or
            record["passed_label"] == "unavailable"):
        return False
    values = {
        key: _q8_u64(record, key)
        for key in ("bytes", "tokens", "in", "out", "weight_offset")
    }
    return (
        all(value is not None for value in values.values()) and
        all(values[key] and values[key] > 0
            for key in ("bytes", "tokens", "in", "out"))
    )


def q8_partner_pre_activation_identity(
        record: dict[str, str]) -> tuple[str, ...]:
    return tuple(record.get(key, "") for key in Q8_PRE_ACTIVATION_IDENTITY_KEYS)


def q8_partner_pre_activation_summary_state(
        summary: dict[str, str], armed: list[dict[str, str]],
        returned: list[dict[str, str]],
        failure: dict[str, str] | None = None,
        allow_unsynchronized: bool = False) -> str:
    if set(summary) != Q8_PRE_ACTIVATION_SUMMARY_KEYS:
        return "malformed-activation-summary"
    if (summary.get("home_tier") != "0" or
            summary.get("partner_tier") != "2" or
            summary.get("partners_synchronized") not in {"yes", "no"}):
        return "malformed-activation-summary"
    synchronized = summary["partners_synchronized"] == "yes"
    if not synchronized and not allow_unsynchronized:
        return "activation-summary-partner-not-synchronized"
    values = {
        key: _q8_u64(summary, key) for key in (
            "attempted", "confirmed", "returned", "failed",
            "last_sequence", "shared_slot_sequence",
            "shared_slot_complement",
        )
    }
    if any(value is None for value in values.values()):
        return "malformed-activation-summary"
    shared_sequence = values["shared_slot_sequence"]
    shared_complement = values["shared_slot_complement"]
    assert shared_sequence is not None and shared_complement is not None
    # A later post-compute marker may overwrite the activation tag.  A
    # synchronized teardown still has to observe one internally exact pair.
    if (synchronized and not failure and
            shared_complement != ((~shared_sequence) & Q8_UINT64_MASK)):
        return "activation-summary-shared-slot-mismatch"
    if failure:
        expected = {
            key: _q8_u64(failure, key)
            for key in ("attempted", "confirmed", "returned", "failed")
        }
        if (any(value is None for value in expected.values()) or
                any(values[key] != expected[key] for key in expected) or
                values["last_sequence"] != expected["confirmed"]):
            return "activation-summary-failure-mismatch"
        return "complete"
    attempted = values["attempted"]
    confirmed = values["confirmed"]
    returned_count = values["returned"]
    failed = values["failed"]
    assert attempted is not None and confirmed is not None
    assert returned_count is not None and failed is not None
    allowed_return_gap = 1 if not synchronized and allow_unsynchronized else 0
    if (attempted <= 0 or attempted != confirmed or failed != 0 or
            returned_count > confirmed or
            confirmed - returned_count > allowed_return_gap or
            values["last_sequence"] != attempted or
            len(armed) != attempted or len(returned) != returned_count):
        return "activation-summary-count-mismatch"
    return "complete"


def q8_partner_pre_activation_armed_record_valid(
        record: dict[str, str]) -> bool:
    if (set(record) != Q8_PRE_ACTIVATION_ARMED_KEYS or
            record["activation_d2h_status"] != "complete" or
            record["destination_sync_status"] != "complete" or
            not q8_partner_pre_activation_identity_valid(record)):
        return False
    current = _q8_u64(record, "current_sequence")
    marker_sequence = _q8_u64(record, "marker_sequence")
    marker_complement = _q8_u64(record, "marker_complement")
    return _q8_pre_activation_marker_exact(
        current, marker_sequence, marker_complement
    )


def q8_partner_pre_activation_returned_record_valid(
        record: dict[str, str]) -> bool:
    if (set(record) != Q8_PRE_ACTIVATION_RETURNED_KEYS or
            record.get("activation_h2d_status") != "success" or
            not q8_partner_pre_activation_identity_valid(record)):
        return False
    current = _q8_u64(record, "current_sequence")
    return _q8_pre_activation_tagged(current) is not None


def format_q8_partner_pre_activation_call_record(
        record: dict[str, str]) -> str:
    if not record:
        return ""
    keys = (
        "current_sequence", "marker_sequence", "marker_complement",
        "activation_d2h_status", "destination_sync_status",
        "activation_h2d_status", *Q8_PRE_ACTIVATION_IDENTITY_KEYS,
    )
    return " ".join(
        f"{key}={record.get(key, '')}" for key in keys if key in record
    )


def q8_partner_pre_activation_call_sequence_state(text: str) -> str:
    armed = q8_partner_pre_activation_armed_records(text)
    returned = q8_partner_pre_activation_returned_records(text)
    if not armed and not returned:
        return "empty"
    if any(not q8_partner_pre_activation_armed_record_valid(record)
           for record in armed):
        return "invalid-activation-armed-record"
    if any(not q8_partner_pre_activation_returned_record_valid(record)
           for record in returned):
        return "invalid-activation-returned-record"
    armed_sequences = [
        _q8_u64(record, "current_sequence") for record in armed
    ]
    returned_sequences = [
        _q8_u64(record, "current_sequence") for record in returned
    ]
    if armed_sequences != list(range(1, len(armed) + 1)):
        return "activation-armed-sequence-mismatch"
    if returned_sequences != list(range(1, len(returned) + 1)):
        return "activation-returned-sequence-mismatch"
    if len(returned) > len(armed) or len(armed) - len(returned) > 1:
        return "activation-armed-returned-count-mismatch"
    if any(
            q8_partner_pre_activation_identity(armed[index]) !=
            q8_partner_pre_activation_identity(returned[index])
            for index in range(len(returned))):
        return "activation-armed-returned-binding-mismatch"

    observed: list[tuple[str, int | None]] = []
    armed_prefix = "ds4: CUDA q8 partner pre-activation armed "
    returned_prefix = "ds4: CUDA q8 partner pre-activation returned "
    for line in text.splitlines():
        if line.startswith(armed_prefix):
            record = _q8_pre_activation_records(line, "armed ")[0]
            observed.append(("armed", _q8_u64(record, "current_sequence")))
        elif line.startswith(returned_prefix):
            record = _q8_pre_activation_records(line, "returned ")[0]
            observed.append(("returned", _q8_u64(record, "current_sequence")))
    expected: list[tuple[str, int]] = []
    for sequence in range(1, len(returned) + 1):
        expected.extend((("armed", sequence), ("returned", sequence)))
    if len(armed) > len(returned):
        expected.append(("armed", len(armed)))
    return (
        "complete" if observed == expected
        else "activation-armed-returned-order-mismatch"
    )


def q8_partner_pre_activation_fence_record_classification(
        record: dict[str, str], kind: str) -> str:
    if (set(record) != Q8_PRE_ACTIVATION_FENCE_KEYS or
            not q8_partner_pre_activation_identity_valid(record)):
        return "invalid-activation-fence-record"
    numeric = {
        key: _q8_u64(record, key) for key in (
            "current_sequence", "marker_sequence", "marker_complement",
            "attempted", "confirmed", "returned", "failed",
        )
    }
    if any(value is None for value in numeric.values()):
        return "invalid-activation-fence-record"
    current = numeric["current_sequence"]
    marker_sequence = numeric["marker_sequence"]
    marker_complement = numeric["marker_complement"]
    attempted = numeric["attempted"]
    confirmed = numeric["confirmed"]
    returned = numeric["returned"]
    failed = numeric["failed"]
    assert current is not None and attempted is not None
    assert confirmed is not None and returned is not None and failed is not None
    sequence_domain_exhausted = (
        kind == "failure" and record.get("event") == "state-invalid" and
        record.get("stage") == "pre-activation-marker" and
        record.get("cuda_status") == "sequence-domain-exhausted" and
        current == Q8_PRE_ACTIVATION_MARKER_TAG
    )
    if (attempted != current or confirmed > attempted or returned > confirmed or
            failed > attempted or
            (_q8_pre_activation_tagged(current) is None and
             not sequence_domain_exhausted)):
        return "invalid-activation-fence-record"
    marker_exact = _q8_pre_activation_marker_exact(
        current, marker_sequence, marker_complement
    )
    marker_claim = record["marker_matches"]
    if marker_claim not in {"yes", "no"} or \
            ((marker_claim == "yes") != marker_exact):
        return "invalid-activation-fence-record"
    booleans = (
        "activation_d2h_attempted", "activation_d2h_completed",
        "activation_h2d_attempted", "activation_h2d_completed",
    )
    if any(record[key] not in {"yes", "no"} for key in booleans):
        return "invalid-activation-fence-record"
    d2h_attempted = record["activation_d2h_attempted"] == "yes"
    d2h_completed = record["activation_d2h_completed"] == "yes"
    h2d_attempted = record["activation_h2d_attempted"] == "yes"
    h2d_completed = record["activation_h2d_completed"] == "yes"
    if ((d2h_completed and not d2h_attempted) or
            (h2d_completed and not h2d_attempted) or
            (h2d_attempted and not d2h_completed)):
        return "invalid-activation-fence-record"

    if kind == "checkpoint":
        valid = (
            record["event"] == "marker-confirmed" and
            record["stage"] == "pre-activation-h2d" and
            record["cuda_status"] == "complete" and
            record["destination_sync_status"] == "complete" and
            d2h_attempted and d2h_completed and not h2d_attempted and
            not h2d_completed and marker_exact and
            record["interpretation"] ==
            "activation-d2h-and-destination-stream-confirmed-before-h2d" and
            confirmed == current and returned + 1 == current and failed == 0
        )
        return (
            "activation-fence-confirmed-before-h2d"
            if valid else "invalid-activation-fence-record"
        )
    if kind != "failure" or failed != 1 or returned + 1 != current:
        return "invalid-activation-fence-record"
    event_specs = {
        "copy-contract-invalid": (
            "pre-activation-d2h", "not-attempted", "not-attempted",
            (False, False, False, False), "failure-before-activation-d2h",
            "activation-audit-state-failed", "zero",
        ),
        "bounce-alloc-failed": (
            "activation-staging", "error", "not-attempted",
            (False, False, False, False), "failure-before-activation-d2h",
            "activation-source-or-staging-setup-failed", "zero",
        ),
        "bounce-free-failed": (
            "activation-staging", "error", "not-attempted",
            (False, False, False, False),
            "failure-surfaced-by-bounce-free-before-activation-d2h",
            "activation-source-or-staging-setup-failed", "zero",
        ),
        "source-switch-failed": (
            "activation-d2h", "cudaSetDevice-failed", "not-attempted",
            (False, False, False, False), "failure-before-activation-d2h",
            "activation-source-or-staging-setup-failed", "zero",
        ),
        "activation-d2h-failed": (
            "activation-d2h", "error", "not-attempted",
            (True, False, False, False),
            "failure-surfaced-by-activation-d2h",
            "activation-source-d2h-failed", "zero",
        ),
        "destination-switch-failed": (
            "pre-activation-h2d", "cudaSetDevice-failed", "not-attempted",
            (True, True, False, False),
            "failure-after-activation-d2h-before-destination-sync",
            "activation-destination-switch-or-setup-failed", "zero",
        ),
        "destination-sync-failed": (
            "pre-activation-h2d", "error", "failed",
            (True, True, False, False),
            "failure-surfaced-by-pre-h2d-destination-device-sync",
            "activation-pre-h2d-device-sync-failed", "zero",
        ),
        "marker-launch-failed": (
            "pre-activation-marker", "error", "complete",
            (True, True, False, False),
            "failure-surfaced-by-pre-h2d-marker-launch",
            "activation-marker-channel-failure", "zero",
        ),
        "event-record-failed": (
            "pre-activation-marker", "error", "complete",
            (True, True, False, False),
            "failure-surfaced-by-pre-h2d-event-record",
            "activation-marker-channel-failure", "zero",
        ),
        "event-sync-failed": (
            "pre-activation-marker", "error", "complete",
            (True, True, False, False),
            "failure-surfaced-by-pre-h2d-event-sync",
            "activation-marker-channel-failure", "observed",
        ),
        "marker-validation-failed": (
            "pre-activation-marker", "complete", "complete",
            (True, True, False, False),
            "pre-h2d-event-confirmed-marker-invalid",
            "activation-marker-channel-failure", "invalid",
        ),
        "activation-h2d-failed": (
            "activation-h2d", "error", "complete",
            (True, True, True, False),
            "failure-surfaced-by-activation-h2d-after-confirmed-boundary",
            "activation-fence-confirmed-h2d-failed", "exact",
        ),
    }
    if record["event"] == "state-invalid":
        state_signature = (record["stage"], record["cuda_status"])
        if state_signature not in {
                ("pre-activation-d2h", "not-attempted"),
                ("pre-activation-marker", "sequence-domain-exhausted")}:
            return "invalid-activation-fence-record"
        spec = (
            state_signature[0], state_signature[1], "not-attempted",
            (False, False, False, False), "failure-before-activation-d2h",
            "activation-audit-state-failed", "zero",
        )
    else:
        spec = event_specs.get(record["event"])
    if not spec:
        return "invalid-activation-fence-record"
    (stage, cuda_status, destination_sync_status, flags, interpretation,
     classification, marker_mode) = spec
    actual_flags = (
        d2h_attempted, d2h_completed, h2d_attempted, h2d_completed,
    )
    cuda_valid = (
        record["cuda_status"] not in {"", "complete", "not-attempted"}
        if cuda_status == "error"
        else record["cuda_status"] == cuda_status
    )
    marker_state_valid = {
        "zero": marker_sequence == 0 and marker_complement == 0 and
                marker_claim == "no",
        "observed": (marker_claim == "yes") == marker_exact,
        "invalid": marker_claim == "no" and not marker_exact,
        "exact": marker_claim == "yes" and marker_exact,
    }[marker_mode]
    expected_confirmed = current if record["event"] == "activation-h2d-failed" \
        else current - 1
    valid = (
        record["stage"] == stage and cuda_valid and
        record["destination_sync_status"] == destination_sync_status and
        actual_flags == flags and record["interpretation"] == interpretation and
        confirmed == expected_confirmed and marker_state_valid
    )
    return classification if valid else "invalid-activation-fence-record"


def format_q8_partner_pre_activation_fence_record(
        record: dict[str, str], classification: str) -> str:
    if not record:
        return ""
    keys = (
        "event", "stage", "current_sequence", "marker_sequence",
        "marker_complement", "marker_matches", "cuda_status",
        "destination_sync_status", "activation_d2h_attempted",
        "activation_d2h_completed", "activation_h2d_attempted",
        "activation_h2d_completed", "interpretation", "attempted",
        "confirmed", "returned", "failed", *Q8_PRE_ACTIVATION_IDENTITY_KEYS,
    )
    return " ".join(
        [f"record_classification={classification}"] +
        [f"{key}={record.get(key, '')}" for key in keys]
    )


def q8_partner_pre_activation_checkpoint_sequence_state(
        text: str, allow_trailing_gap: bool = False) -> str:
    armed = q8_partner_pre_activation_armed_records(text)
    checkpoints = q8_partner_pre_activation_fence_records(text, "checkpoint")
    if any(
            q8_partner_pre_activation_fence_record_classification(
                record, "checkpoint"
            ) != "activation-fence-confirmed-before-h2d"
            for record in checkpoints):
        return "invalid-activation-checkpoint-record"
    sequences = [
        _q8_u64(record, "current_sequence") for record in checkpoints
    ]
    expected = [sequence for sequence in range(1, len(armed) + 1)
                if sequence == 1 or sequence % 64 == 0]
    if (allow_trailing_gap and expected and armed and
            expected[-1] == len(armed) and sequences == expected[:-1]):
        expected = expected[:-1]
    if sequences != expected:
        return "activation-checkpoint-cadence-mismatch"
    armed_by_sequence = {
        _q8_u64(record, "current_sequence"): record for record in armed
    }
    positions: dict[str, dict[int, int]] = {
        "armed": {}, "checkpoint": {}, "returned": {},
    }
    prefixes = {
        "armed": "ds4: CUDA q8 partner pre-activation armed ",
        "checkpoint": "ds4: CUDA q8 partner pre-activation fence checkpoint ",
        "returned": "ds4: CUDA q8 partner pre-activation returned ",
    }
    for index, line in enumerate(text.splitlines()):
        for kind, prefix in prefixes.items():
            if line.startswith(prefix):
                match = re.search(r"(?:^| )current_sequence=(\d+)(?: |$)", line)
                sequence = int(match.group(1)) if match else None
                if sequence is not None:
                    positions[kind][sequence] = index
                break
    for record in checkpoints:
        sequence = _q8_u64(record, "current_sequence")
        if sequence is None or sequence not in armed_by_sequence:
            return "activation-checkpoint-order-mismatch"
        if q8_partner_pre_activation_identity(record) != \
                q8_partner_pre_activation_identity(armed_by_sequence[sequence]):
            return "activation-checkpoint-binding-mismatch"
        armed_position = positions["armed"].get(sequence, -1)
        checkpoint_position = positions["checkpoint"].get(sequence, -1)
        returned_position = positions["returned"].get(sequence)
        if (armed_position < 0 or checkpoint_position <= armed_position or
                (returned_position is not None and
                 checkpoint_position >= returned_position)):
            return "activation-checkpoint-order-mismatch"
    return "complete"


def q8_partner_pre_activation_failure_history_valid(
        text: str, failure: dict[str, str]) -> bool:
    armed = q8_partner_pre_activation_armed_records(text)
    returned = q8_partner_pre_activation_returned_records(text)
    current = _q8_u64(failure, "current_sequence")
    sequence_domain_exhausted = (
        failure.get("event") == "state-invalid" and
        failure.get("stage") == "pre-activation-marker" and
        failure.get("cuda_status") == "sequence-domain-exhausted" and
        current == Q8_PRE_ACTIVATION_MARKER_TAG
    )
    if sequence_domain_exhausted:
        # A complete per-call history cannot be materialized at 2^63 calls.
        # The exact terminal counters were already checked by the record
        # classifier; accept a terminal exhaustion record without pretending
        # that a short breadcrumb list can represent that history.
        relationship = not armed and not returned
    elif failure.get("event") == "activation-h2d-failed":
        relationship = (
            current == len(armed) and len(armed) == len(returned) + 1 and
            bool(armed) and
            q8_partner_pre_activation_identity(failure) ==
            q8_partner_pre_activation_identity(armed[-1])
        )
    else:
        relationship = (
            current == len(armed) + 1 and len(armed) == len(returned)
        )
    if not relationship:
        return False
    lines = text.splitlines()
    failure_prefix = "ds4: CUDA q8 partner pre-activation fence failure "
    evidence_prefixes = (
        "ds4: CUDA q8 partner pre-activation armed ",
        "ds4: CUDA q8 partner pre-activation returned ",
        "ds4: CUDA q8 partner pre-activation fence checkpoint ",
    )
    failure_positions = [
        index for index, line in enumerate(lines) if failure_prefix in line
    ]
    evidence_positions = [
        index for index, line in enumerate(lines)
        if any(prefix in line for prefix in evidence_prefixes)
    ]
    return (
        len(failure_positions) == 1 and
        (not evidence_positions or failure_positions[0] > max(evidence_positions))
    )


def q8_partner_pre_activation_fence_marker_state(text: str) -> str:
    if not q8_partner_pre_activation_fence_enabled(text, 0):
        return "not-enabled-exactly-once-for-pair0"
    order_state = q8_partner_pre_activation_record_order_state(text)
    if order_state != "complete":
        return order_state
    call_state = q8_partner_pre_activation_call_sequence_state(text)
    checkpoint_state = q8_partner_pre_activation_checkpoint_sequence_state(text)
    armed = q8_partner_pre_activation_armed_records(text)
    returned = q8_partner_pre_activation_returned_records(text)
    failures = q8_partner_pre_activation_fence_records(text, "failure")
    summaries = q8_partner_pre_activation_fence_records(text, "summary")
    if failures:
        if len(failures) != 1:
            return f"activation-failure-count:{len(failures)}"
        classification = q8_partner_pre_activation_fence_record_classification(
            failures[0], "failure"
        )
        if classification == "invalid-activation-fence-record":
            return classification
        if call_state not in {"complete", "empty"}:
            return call_state
        if checkpoint_state != "complete":
            return checkpoint_state
        if not q8_partner_pre_activation_failure_history_valid(text, failures[0]):
            return "activation-failure-history-mismatch"
        if len(summaries) > 1:
            return f"activation-summary-count:{len(summaries)}"
        if summaries:
            summary_state = q8_partner_pre_activation_summary_state(
                summaries[0], armed, returned, failures[0],
                allow_unsynchronized=True,
            )
            if summary_state != "complete":
                return summary_state
        return "failure-record-present"
    if call_state != "complete":
        return call_state
    if checkpoint_state != "complete":
        return checkpoint_state
    if len(armed) != len(returned):
        return "activation-h2d-return-count-mismatch"
    if len(summaries) != 1:
        return f"activation-summary-count:{len(summaries)}"
    return q8_partner_pre_activation_summary_state(
        summaries[0], armed, returned
    )


def q8_partner_pre_activation_fence_summary(
        text: str, run_status: str = ""
) -> tuple[str, str, str, str, str, str, str, str, str, str, str]:
    if not q8_partner_pre_activation_fence_enabled(text, 0):
        empty = ("", "", "", "", "", "", "", "", "", "", "")
        return (
            ("", "", "", "", "invalid-activation-fence-record",
             "", "", "", "", "0", "0")
            if "q8 partner pre-activation" in text else empty
        )
    checkpoints = q8_partner_pre_activation_fence_records(text, "checkpoint")
    failures = q8_partner_pre_activation_fence_records(text, "failure")
    summaries = q8_partner_pre_activation_fence_records(text, "summary")
    armed = q8_partner_pre_activation_armed_records(text)
    returned = q8_partner_pre_activation_returned_records(text)
    checkpoint = checkpoints[-1] if checkpoints else {}
    failure = failures[-1] if failures else {}
    summary = summaries[-1] if summaries else {}
    last_armed = armed[-1] if armed else {}
    last_returned = returned[-1] if returned else {}
    order_state = q8_partner_pre_activation_record_order_state(text)
    call_state = q8_partner_pre_activation_call_sequence_state(text)
    device_loss = run_status in {
        "failed-device-loss", "interrupted-prior-run-device-loss",
        "interrupted-no-result-device-loss",
    }
    failure_classification = (
        q8_partner_pre_activation_fence_record_classification(
            failure, "failure"
        ) if failure else ""
    )
    summary_state = ""
    if len(summaries) == 1:
        summary_state = q8_partner_pre_activation_summary_state(
            summary, armed, returned, failure if failure else None,
            allow_unsynchronized=(device_loss or bool(failure)),
        )
    elif summaries:
        summary_state = f"activation-summary-count:{len(summaries)}"
    unsynchronized_cleanup_summary = (
        len(summaries) == 1 and summary_state == "complete" and
        summary.get("partners_synchronized") == "no"
    )
    checkpoint_state = q8_partner_pre_activation_checkpoint_sequence_state(
        text, allow_trailing_gap=(
            device_loss and not failures and
            (not summaries or unsynchronized_cleanup_summary) and
            len(armed) == len(returned) + 1
        )
    )
    if failure:
        if (len(failures) != 1 or
                failure_classification == "invalid-activation-fence-record" or
                order_state != "complete" or
                call_state not in {"complete", "empty"} or
                checkpoint_state != "complete" or
                (summary and summary_state != "complete") or
                not q8_partner_pre_activation_failure_history_valid(
                     text, failure)):
            classification = "invalid-activation-fence-record"
        else:
            classification = failure_classification
    elif (order_state != "complete" or
          call_state not in {"complete", "empty"} or
          checkpoint_state != "complete"):
        classification = "invalid-activation-fence-record"
    elif summary:
        if len(summaries) != 1 or summary_state != "complete":
            classification = "invalid-activation-fence-record"
        elif device_loss and len(armed) == len(returned) + 1:
            checkpoint_sequence = _q8_u64(checkpoint, "current_sequence")
            armed_sequence = _q8_u64(last_armed, "current_sequence")
            classification = (
                "activation-fence-confirmed-h2d-return-not-observed-"
                "trailing-checkpoint"
                if checkpoint_sequence == armed_sequence else
                "activation-fence-confirmed-h2d-return-not-observed"
            )
        elif device_loss:
            classification = "activation-h2d-returned-subsequent-locus-unresolved"
        else:
            classification = "completed-all-activation-h2d-returned"
    elif device_loss and not summary and len(armed) == len(returned) + 1:
        checkpoint_sequence = _q8_u64(checkpoint, "current_sequence")
        armed_sequence = _q8_u64(last_armed, "current_sequence")
        classification = (
            "activation-fence-confirmed-h2d-return-not-observed-"
            "trailing-checkpoint"
            if checkpoint_sequence == armed_sequence else
            "activation-fence-confirmed-h2d-return-not-observed"
        )
    elif device_loss and not summary and armed and len(armed) == len(returned):
        classification = "activation-h2d-returned-subsequent-locus-unresolved"
    elif checkpoint:
        classification = "prior-activation-fence-calls-confirmed-current-unresolved"
    else:
        classification = "activation-fence-enabled-no-call-evidence"
    if failure:
        source = failure
    elif summary:
        source = summary
    else:
        # Sparse checkpoints are not current counters.  Validated breadcrumbs
        # are exact through the last armed/returned call observed in the log.
        source = {
            "attempted": str(len(armed)),
            "confirmed": str(len(armed)),
            "returned": str(len(returned)),
            "failed": "0",
        }
    return (
        format_q8_partner_pre_activation_fence_record(
            checkpoint,
            q8_partner_pre_activation_fence_record_classification(
                checkpoint, "checkpoint"
            ) if checkpoint else "",
        ),
        format_q8_partner_pre_activation_fence_record(
            failure, failure_classification
        ),
        format_q8_partner_pre_activation_call_record(last_armed),
        format_q8_partner_pre_activation_call_record(last_returned),
        classification,
        source.get("attempted", str(len(armed))),
        source.get("confirmed", str(len(armed))),
        source.get("returned", str(len(returned))),
        source.get("failed", "0"),
        str(len(armed)),
        str(len(returned)),
    )


def q8_partner_phase_audit_markers(
        text: str, pair: int | None) -> list[dict[str, str]]:
    pattern = re.compile(
        r"q8 partner phase audit sequence=(\d+) event=(\S+) stage=(\S+) "
        r"binding_label=(\S+) passed_label=(\S+) weight_offset=(\d+) "
        r"home_tier=(-?\d+) home_device=(-?\d+) "
        r"partner_tier=(-?\d+) partner_device=(-?\d+) "
        r"tokens=(\d+) in=(\d+) out=(\d+) transfer_bytes=(\d+) "
        r"result_bytes=(\d+) cuda_error=(.*)$"
    )
    markers: list[dict[str, str]] = []
    lines = text.splitlines()
    run_phase = "startup"
    for index, line in enumerate(lines):
        if "starting untimed CUDA warm-up frontier" in line:
            run_phase = "warmup"
        elif "completed untimed CUDA warm-up frontier" in line:
            run_phase = "post-warmup"
        elif "prefill fault breadcrumb event=chunk-begin" in line:
            run_phase = "measured-prefill"
        match = pattern.search(line)
        if not match:
            continue
        fields = (
            "sequence", "event", "stage", "binding_label", "passed_label",
            "weight_offset", "home_tier", "home_device", "partner_tier",
            "partner_device", "tokens", "in", "out", "transfer_bytes",
            "result_bytes", "cuda_error",
        )
        marker = dict(zip(fields, match.groups()))
        if pair is not None and marker["home_tier"] != str(pair):
            continue
        marker["run_phase"] = run_phase
        marker["cuda_phase"] = ""
        if index > 0:
            phase_match = re.search(
                r"CUDA default-stream bounce (alloc|d2h|h2d) failed:",
                lines[index - 1],
            )
            if phase_match:
                marker["cuda_phase"] = phase_match.group(1)
        markers.append(marker)
    return markers


def q8_partner_phase_audit_occurrence_records(
        text: str) -> list[dict[str, str]]:
    pattern = re.compile(
        r"q8 partner phase audit (skipped|selected) occurrence=(\d+) "
        r"(?:sequence=(\d+) )?binding_label=(\S+) weight_offset=(\d+)"
    )
    records: list[dict[str, str]] = []
    run_phase = "startup"
    for line in text.splitlines():
        if "starting untimed CUDA warm-up frontier" in line:
            run_phase = "warmup"
        elif "completed untimed CUDA warm-up frontier" in line:
            run_phase = "post-warmup"
        elif "prefill fault breadcrumb event=chunk-begin" in line:
            run_phase = "measured-prefill"
        match = pattern.search(line)
        if not match:
            continue
        disposition, occurrence, sequence, binding_label, weight_offset = (
            match.groups()
        )
        records.append({
            "disposition": disposition,
            "occurrence": occurrence,
            "sequence": sequence or "",
            "binding_label": binding_label,
            "weight_offset": weight_offset,
            "run_phase": run_phase,
        })
    return records


def format_q8_partner_phase_audit_occurrence(
        record: dict[str, str]) -> str:
    sequence = (
        f" sequence={record['sequence']}" if record.get("sequence") else ""
    )
    return (
        f"disposition={record['disposition']} "
        f"occurrence={record['occurrence']}{sequence} "
        f"binding_label={record['binding_label']} "
        f"weight_offset={record['weight_offset']} "
        f"run_phase={record['run_phase']}"
    )


def q8_partner_phase_audit_occurrence_summary(
        text: str, disposition: str) -> str:
    records = [
        record for record in q8_partner_phase_audit_occurrence_records(text)
        if record["disposition"] == disposition
    ]
    return (
        format_q8_partner_phase_audit_occurrence(records[-1])
        if records else ""
    )


def q8_l12_occurrence_mapping_state(text: str) -> str:
    records = q8_partner_phase_audit_occurrence_records(text)
    if not records:
        return "not-observed"
    if len(records) != 2:
        return f"invalid-record-count:{len(records)}"
    skipped, selected = records
    if (
            skipped == {
                "disposition": "skipped",
                "occurrence": "1",
                "sequence": "",
                "binding_label": Q8_L12_LABEL,
                "weight_offset": Q8_L12_OFFSET,
                "run_phase": "warmup",
            } and
            selected == {
                "disposition": "selected",
                "occurrence": "2",
                "sequence": "1",
                "binding_label": Q8_L12_LABEL,
                "weight_offset": Q8_L12_OFFSET,
                "run_phase": "measured-prefill",
            }):
        return "verified"
    return "invalid-order-phase-or-identity"


def q8_partner_phase_audit_window_marker_state(
        text: str, expected_per_binding: int) -> str:
    if expected_per_binding <= 0:
        return "invalid-expected-count"
    marker_prefix = "ds4: CUDA q8 partner phase audit sequence="
    raw_count = sum(marker_prefix in line for line in text.splitlines())
    markers = q8_partner_phase_audit_markers(text, None)
    if len(markers) != raw_count:
        return "malformed-marker"

    chains: defaultdict[int, list[tuple[str, str]]] = defaultdict(list)
    sequence_binding: dict[int, str] = {}
    for marker in markers:
        if (marker["home_tier"] != "0" or marker["home_device"] != "0" or
                marker["partner_tier"] != "2" or
                marker["partner_device"] != "1"):
            return "wrong-pair-or-device"
        binding = marker["binding_label"]
        if (binding not in Q8_WINDOW_TARGETS or
                marker["weight_offset"] != Q8_WINDOW_TARGETS[binding]):
            return "wrong-target-tuple"
        if (marker["passed_label"] != "attn_output_b" or
                marker["tokens"] != "512" or marker["in"] != "8192" or
                marker["out"] != "4096" or
                marker["transfer_bytes"] != "8388608" or
                marker["result_bytes"] != "8388608" or
                marker["cuda_error"] != "none"):
            return "wrong-target-shape-or-result"
        sequence = int(marker["sequence"])
        prior_binding = sequence_binding.setdefault(sequence, binding)
        if prior_binding != binding:
            return "sequence-identity-changed"
        chains[sequence].append((marker["event"], marker["stage"]))

    expected_total = expected_per_binding * 2
    if set(chains) != set(range(1, expected_total + 1)):
        return "sequence-set-mismatch"
    counts = {label: 0 for label in Q8_WINDOW_TARGETS}
    layer14_seen = 0
    layer15_seen = 0
    for sequence in range(1, expected_total + 1):
        if tuple(chains[sequence]) != Q8_WINDOW_CHAIN:
            return f"sequence-{sequence}-chain-mismatch"
        binding = sequence_binding[sequence]
        counts[binding] += 1
        if binding == Q8_WINDOW_L14_LABEL:
            layer14_seen += 1
        else:
            layer15_seen += 1
            if layer15_seen > layer14_seen:
                return "layer15-preceded-layer14"
    if any(count != expected_per_binding for count in counts.values()):
        return "per-binding-count-mismatch"
    return "complete"


def q8_partner_phase_audit_window_summary(
        text: str, pair: int) -> tuple[str, str, str, str]:
    markers = q8_partner_phase_audit_markers(text, pair)
    exact_markers = [
        marker for marker in markers
        if (marker["binding_label"] in Q8_WINDOW_TARGETS and
            marker["weight_offset"] == Q8_WINDOW_TARGETS[
                marker["binding_label"]])
    ]
    measured_started = "prefill fault breadcrumb event=chunk-begin" in text
    measured_markers = [
        marker for marker in exact_markers
        if marker.get("run_phase") == "measured-prefill"
    ]
    analysis_markers = measured_markers if measured_started else exact_markers
    failure_events = {
        "activation-copy-failed", "pre-compute-sync-failed",
        "compute-submit-failed", "compute-sync-failed",
        "result-copy-failed", "home-restore-failed",
    }
    complete_markers = [
        marker for marker in analysis_markers
        if marker["event"] == "result-complete"
    ]
    all_complete_markers = [
        marker for marker in exact_markers
        if marker["event"] == "result-complete"
    ]
    complete_sequences = {
        label: {
            marker["sequence"] for marker in complete_markers
            if marker["binding_label"] == label
        }
        for label in Q8_WINDOW_TARGETS
    }
    all_complete_sequences = {
        label: {
            marker["sequence"] for marker in all_complete_markers
            if marker["binding_label"] == label
        }
        for label in Q8_WINDOW_TARGETS
    }
    last_complete = (
        format_q8_partner_phase_audit_marker(all_complete_markers[-1])
        if all_complete_markers else ""
    )
    first_failure_index = next((
        index for index, marker in enumerate(analysis_markers)
        if marker["event"] in failure_events
    ), None)
    first_failure = (
        analysis_markers[first_failure_index]
        if first_failure_index is not None else {}
    )
    classification = "inconclusive"
    if measured_started and not measured_markers:
        warmup_completed = {
            marker["binding_label"] for marker in all_complete_markers
            if marker.get("run_phase") == "warmup"
        }
        if warmup_completed == set(Q8_WINDOW_TARGETS):
            classification = "measured-target-window-not-reached"
    elif first_failure:
        binding = first_failure["binding_label"]
        event = first_failure["event"]
        if binding == Q8_WINDOW_L14_LABEL:
            classification = {
                "activation-copy-failed": "layer14-activation-copy",
                "pre-compute-sync-failed": "before-layer14-compute",
                "compute-submit-failed": "layer14-compute-submit",
                "compute-sync-failed":
                    "layer14-compute-or-concurrent-partner-work",
            }.get(event, "layer14-recovery-failure")
            if event == "result-copy-failed":
                _, _, base_classification = q8_partner_phase_audit_summary(
                    text, pair
                )
                classification = (
                    "layer14-" + base_classification
                    if base_classification != "inconclusive"
                    else "layer14-result-copy-after-compute"
                )
        else:
            prior_markers = analysis_markers[:first_failure_index]
            l15_started = {
                marker["sequence"] for marker in prior_markers
                if (marker["binding_label"] == Q8_WINDOW_L15_LABEL and
                    marker["event"] == "begin")
            }
            l14_completed = {
                marker["sequence"] for marker in prior_markers
                if (marker["binding_label"] == Q8_WINDOW_L14_LABEL and
                    marker["event"] == "result-complete")
            }
            prior_l14_complete = (
                first_failure["sequence"] in l15_started and
                len(l14_completed) >= len(l15_started)
            )
            if not prior_l14_complete:
                classification = (
                    "layer15-failure-without-prior-layer14-completion"
                )
            elif event in {"activation-copy-failed", "pre-compute-sync-failed"}:
                classification = "between-layer14-result-and-layer15-compute"
            elif event == "compute-submit-failed":
                classification = "layer15-compute-submit-after-layer14"
            elif event == "compute-sync-failed":
                classification = "layer15-compute-or-concurrent-partner-work"
            elif event == "result-copy-failed":
                _, _, base_classification = q8_partner_phase_audit_summary(
                    text, pair
                )
                classification = (
                    "layer15-" + base_classification
                    if base_classification != "inconclusive"
                    else "layer15-result-copy-after-compute"
                )
            else:
                classification = "layer15-recovery-failure-after-layer14"
    elif analysis_markers and analysis_markers[-1]["event"] != "result-complete":
        if analysis_markers[-1]["binding_label"] == Q8_WINDOW_L14_LABEL:
            classification = (
                "layer14-partial-after-prior-completion"
                if complete_markers else "layer14-partial-no-completion"
            )
        elif (len(complete_sequences[Q8_WINDOW_L14_LABEL]) >
              len(complete_sequences[Q8_WINDOW_L15_LABEL])):
            classification = "layer14-complete-layer15-not-complete"
        else:
            classification = "layer15-partial-without-confirmed-layer14"
    elif (len(complete_sequences[Q8_WINDOW_L14_LABEL]) >
          len(complete_sequences[Q8_WINDOW_L15_LABEL])):
        classification = "layer14-complete-layer15-not-complete"
    elif complete_sequences[Q8_WINDOW_L14_LABEL] and complete_sequences[
            Q8_WINDOW_L15_LABEL]:
        classification = "both-targets-complete"
    return (
        last_complete,
        str(len(all_complete_sequences[Q8_WINDOW_L14_LABEL])),
        str(len(all_complete_sequences[Q8_WINDOW_L15_LABEL])),
        classification,
    )


def format_q8_partner_phase_audit_marker(marker: dict[str, str]) -> str:
    if not marker:
        return ""
    cuda_phase = (
        f" cuda_phase={marker['cuda_phase']}" if marker.get("cuda_phase") else ""
    )
    run_phase = (
        f" run_phase={marker['run_phase']}" if marker.get("run_phase") else ""
    )
    return (
        f"sequence={marker['sequence']} event={marker['event']} "
        f"stage={marker['stage']} binding_label={marker['binding_label']} "
        f"passed_label={marker['passed_label']} "
        f"weight_offset={marker['weight_offset']} "
        f"home_tier={marker['home_tier']} home_device={marker['home_device']} "
        f"partner_tier={marker['partner_tier']} "
        f"partner_device={marker['partner_device']} tokens={marker['tokens']} "
        f"in={marker['in']} out={marker['out']} "
        f"transfer_bytes={marker['transfer_bytes']} "
        f"result_bytes={marker['result_bytes']}"
        f"{run_phase}"
        f"{cuda_phase} cuda_error={marker['cuda_error']}"
    )


def first_gpu_layer_failure(text: str) -> str:
    match = re.search(r"ds4: gpu layer (\d+) ([^\n]+?) failed(?:\n|$)", text)
    if not match:
        return ""
    operation = re.sub(r"\s+", "-", match.group(2).strip())
    return f"layer={match.group(1)} operation={operation}"


def q8_partner_phase_audit_summary(
        text: str, pair: int) -> tuple[str, str, str]:
    markers = q8_partner_phase_audit_markers(text, pair)
    checkpoint_events = {
        "begin", "activation-complete", "activation-copy-failed",
        "pre-compute-sync-begin", "pre-compute-complete",
        "pre-compute-sync-failed",
        "compute-submitted", "compute-submit-failed", "compute-complete",
        "compute-sync-failed", "result-complete", "result-copy-failed",
    }
    failure_events = {
        "activation-copy-failed", "compute-submit-failed",
        "pre-compute-sync-failed", "compute-sync-failed",
        "result-copy-failed", "home-restore-failed",
    }
    checkpoints = [
        marker for marker in markers if marker["event"] in checkpoint_events
    ]
    first_failure_index = next(
        (index for index, marker in enumerate(markers)
         if marker["event"] in failure_events),
        None,
    )
    first_failure = (
        markers[first_failure_index] if first_failure_index is not None else {}
    )
    classification = ""
    if first_failure:
        event = first_failure["event"]
        if event == "pre-compute-sync-failed":
            classification = "pre-target-partner-error"
        elif event == "compute-sync-failed":
            sequence = first_failure["sequence"]
            pre_compute_completed = any(
                marker["sequence"] == sequence and
                marker["event"] == "pre-compute-complete"
                for marker in markers[:first_failure_index]
            )
            if pre_compute_completed:
                classification = "target-compute-or-concurrent-partner-work"
        elif event == "result-copy-failed":
            sequence = first_failure["sequence"]
            pre_compute_completed = any(
                marker["sequence"] == sequence and
                marker["event"] == "pre-compute-complete"
                for marker in markers[:first_failure_index]
            )
            compute_completed = any(
                marker["sequence"] == sequence and
                marker["event"] == "compute-complete"
                for marker in markers[:first_failure_index]
            )
            if pre_compute_completed and compute_completed:
                phase = first_failure.get("cuda_phase", "")
                if phase == "d2h":
                    classification = "result-d2h-pcie-host-bounce-transfer"
                elif phase == "h2d":
                    classification = "result-h2d-pcie-host-bounce-transfer"
                elif phase == "alloc":
                    classification = "result-host-bounce-allocation"
                else:
                    classification = "result-host-bounce-copy-after-compute"
        if not classification:
            classification = "inconclusive"
    return (
        format_q8_partner_phase_audit_marker(checkpoints[-1])
        if checkpoints else "",
        format_q8_partner_phase_audit_marker(first_failure),
        classification,
    )


def last_attention_phase_audit(text: str) -> str:
    last = ""
    pattern = re.compile(
        r"prefill attention row phase audit event=(\S+) phase=(\S+) "
        r"kind=(\S+) layer=(\d+) pos=(\d+) tokens=(\d+) "
        r"home=(\d+) partner=(\d+)"
    )
    for match in pattern.finditer(text):
        event, phase, kind, layer, pos, tokens, home, partner = match.groups()
        last = (
            f"{event}:{phase}:{kind}:layer{layer}:pos{pos}:tokens{tokens}:"
            f"home{home}:partner{partner}"
        )
    return last


def last_attention_end_fence(text: str) -> str:
    last = ""
    pattern = re.compile(
        r"prefill attention row end fence event=(\S+) target=(\S+) "
        r"kind=(\S+) layer=(\d+) pos=(\d+) tokens=(\d+) "
        r"home=(\d+) partner=(\d+)"
    )
    for match in pattern.finditer(text):
        event, target, kind, layer, pos, tokens, home, partner = match.groups()
        last = (
            f"{event}:{target}:{kind}:layer{layer}:pos{pos}:tokens{tokens}:"
            f"home{home}:partner{partner}"
        )
    return last


def last_attention_entry_fence(text: str) -> str:
    last = ""
    pattern = re.compile(
        r"prefill attention row entry fence event=(\S+) target=(\S+) "
        r"kind=(\S+) layer=(\d+) pos=(\d+) tokens=(\d+) "
        r"home=(\d+) partner=(\d+)"
    )
    for match in pattern.finditer(text):
        event, target, kind, layer, pos, tokens, home, partner = match.groups()
        last = (
            f"{event}:{target}:{kind}:layer{layer}:pos{pos}:tokens{tokens}:"
            f"home{home}:partner{partner}"
        )
    return last


def first_attention_audit_failure(text: str) -> str:
    fence_pattern = re.compile(
        r"prefill attention row (end|entry) fence event=(\S+) target=(\S+) "
        r"kind=(\S+) layer=(\d+) pos=(\d+) tokens=(\d+) "
        r"home=(\d+) partner=(\d+)"
    )
    phase_pattern = re.compile(
        r"prefill attention row phase audit event=(\S+) phase=(\S+) "
        r"kind=(\S+) layer=(\d+) pos=(\d+) tokens=(\d+) "
        r"home=(\d+) partner=(\d+)"
    )
    failure_events = {
        "submit-failed", "device-switch-failed", "sync-failed", "failed"
    }
    for line in text.splitlines():
        match = fence_pattern.search(line)
        if match:
            family, event, target, kind, layer, pos, tokens, home, partner = (
                match.groups()
            )
            if event in failure_events:
                return (
                    f"{family}-fence:{event}:{target}:{kind}:layer{layer}:"
                    f"pos{pos}:tokens{tokens}:home{home}:partner{partner}"
                )
        match = phase_pattern.search(line)
        if match:
            event, phase, kind, layer, pos, tokens, home, partner = match.groups()
            if event in failure_events:
                return (
                    f"phase-audit:{event}:{phase}:{kind}:layer{layer}:"
                    f"pos{pos}:tokens{tokens}:home{home}:partner{partner}"
                )
    return ""


def attention_row_boundary_marker_state(
        text: str, end_layer: int, entry_layer: int, pos: int) -> str:
    identity_end = (
        f"kind=mixed layer={end_layer} pos={pos} tokens=512 home=0 partner=2"
    )
    identity_entry = (
        f"kind=mixed layer={entry_layer} pos={pos} tokens=512 home=0 partner=2"
    )
    expected = [
        f"prefill attention row end fence event=begin target=pair {identity_end}",
        f"prefill attention row end fence event=complete target=partner {identity_end}",
        f"prefill attention row end fence event=complete target=home {identity_end}",
        f"prefill attention row end fence event=complete target=pair {identity_end}",
        f"prefill attention row audit dispatch=split {identity_end}",
        f"prefill attention row entry fence event=begin target=pair {identity_entry}",
        f"prefill attention row entry fence event=complete target=partner {identity_entry}",
        f"prefill attention row entry fence event=complete target=home {identity_entry}",
        f"prefill attention row entry fence event=complete target=pair {identity_entry}",
    ]
    for phase in (
            "query-copy", "partner-attention", "home-attention", "result-gather"):
        expected.extend([
            f"prefill attention row phase audit event=begin phase={phase} "
            f"{identity_entry}",
            f"prefill attention row phase audit event=complete phase={phase} "
            f"{identity_entry}",
        ])
    expected.append(f"prefill attention row audit dispatch=split {identity_entry}")

    audit_prefixes = (
        "prefill attention row end fence event=",
        "prefill attention row entry fence event=",
        "prefill attention row phase audit event=",
    )
    row_pattern = re.compile(
        r"prefill attention row audit dispatch=\S+ kind=\S+ layer=(\d+) "
        r"pos=\d+ tokens=\d+ home=(\d+) partner=(\d+)"
    )
    observed: list[str] = []
    for line in text.splitlines():
        if any(prefix in line for prefix in audit_prefixes):
            observed.append(line)
            continue
        match = row_pattern.search(line)
        if (match and int(match.group(1)) in {end_layer, entry_layer} and
                match.group(2) == "0" and match.group(3) == "2"):
            observed.append(line)

    for index, (actual, wanted) in enumerate(zip(observed, expected), 1):
        if wanted not in actual:
            return f"diverged:{index}/{len(expected)}"
    if len(observed) < len(expected):
        return f"prefix:{len(observed)}/{len(expected)}"
    if len(observed) > len(expected):
        return f"extra:{len(observed)}/{len(expected)}"
    return "complete"


def outcome(statuses: list[str]) -> str:
    if not statuses:
        return "not-run"
    if all(status == "passed" for status in statuses):
        return "passed"
    failed_statuses = {
        "failed-device-loss", "interrupted-prior-run-device-loss",
        "interrupted-no-result-device-loss",
    }
    if any(status in failed_statuses for status in statuses):
        return "failed"
    invalid_statuses = {
        "validation-failed", "environment-invalid",
        "passed-invalidated-watch", "completed-no-result-invalidated-watch",
    }
    if any(status in invalid_statuses for status in statuses):
        return "invalid"
    if any(status == "inconclusive-underloaded" for status in statuses):
        return "underloaded"
    return "incomplete"


def last_progress(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else {}


def post_health_state(path: Path) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return "unverified"
    text = path.read_text(errors="replace")
    error = re.search(
        r"ERR!|GPU is lost|Unknown Error|Unable to determine|"
        r"GPU Unavailable|Critical Xid",
        text,
        re.IGNORECASE,
    )
    if error:
        return "unhealthy"
    if len(re.findall(r"^GPU \d+:", text, re.MULTILINE)) == 4:
        return "healthy"
    return "unverified"


def healthy_post_snapshot(path: Path) -> bool:
    return post_health_state(path) == "healthy"


def inference(outcomes: dict[str, str], rows: list[dict[str, str]]) -> str:
    if any("power-limit-drift" in row.get("watch_status", "") for row in rows):
        return (
            "At least one arm changed power limit after its 250 W preflight. That arm "
            "is invalid for causal comparison; identify the external power-limit writer "
            "and rerun it."
        )
    if any("foreign-compute-process" in row.get("watch_status", "")
           for row in rows):
        return (
            "At least one arm overlapped an unexpected GPU compute process. That arm "
            "is invalid for causal comparison; identify the external launcher and "
            "rerun it without the foreign workload."
        )
    shadow_variants = (
        ("attention-row-query-shadow", "query-copy"),
        ("attention-row-partner-shadow", "partner-compute"),
        ("attention-row-gather-shadow", "result-gather"),
    )
    for variant, phase in shadow_variants:
        state = outcomes.get(variant, "not-run")
        if state == "failed":
            return (
                f"The production-transport `{phase}` shadow arm lost a device. "
                "Its accepted attention output was assigned to the unchanged home "
                "kernel, so the retained direct-P2P row phase (or its interaction "
                "with concurrent production work) is sufficient under this explicit "
                "completion boundary. Do not run broader shadow phases before "
                "inspecting this artifact."
            )
    for variant, phase in shadow_variants:
        state = outcomes.get(variant, "not-run")
        if state in {"invalid", "underloaded"}:
            return (
                f"The `{phase}` shadow arm is `{state}` rather than causal "
                "evidence. Inspect its activation, path, load, and health gates, "
                "then repeat only that arm in a fresh one-shot directory."
            )
    for variant, phase in shadow_variants:
        state = outcomes.get(variant, "not-run")
        if state == "incomplete":
            return (
                f"The `{phase}` shadow arm has no verified outcome. Preserve it and "
                "repeat that same single arm in a fresh directory; do not resume it."
            )
    passed_shadow_phases = [
        phase for variant, phase in shadow_variants
        if outcomes.get(variant, "not-run") == "passed"
    ]
    if passed_shadow_phases:
        phase = passed_shadow_phases[-1]
        next_phase = {
            "query-copy": "partner-compute",
            "partner-compute": "result-gather",
            "result-gather": "none",
        }[phase]
        if next_phase == "none":
            return (
                "The full direct query/partner-attention/result-gather shadow arm "
                "survived. The production row operations alone were not sufficient "
                "under the shadow arm's completion boundary; compare overlap and "
                "downstream-consumption semantics next."
            )
        return (
            f"The production-transport `{phase}` shadow arm survived with healthy "
            f"post-run GPUs. Run `{next_phase}` as a fresh one-shot arm; do not "
            "resume or combine expected-crash arms."
        )
    compute_rows = [
        row for row in rows
        if row.get("variant") == "attention-q8-global-compute-fence"
    ]
    if (outcomes.get("attention-q8-global-compute-fence") == "failed" and
            compute_rows):
        row = compute_rows[-1]
        classification = row.get("q8_compute_fence_classification", "")
        pre_gather_classification = row.get(
            "q8_pre_gather_fence_classification", ""
        )
        last = (row.get("q8_compute_fence_last_failure") or
                row.get("q8_compute_fence_last_armed", ""))
        if (classification == "all-observed-compute-fences-returned" and
                pre_gather_classification ==
                "mapped-host-marker-partial-write"):
            return (
                "Every observed pair-0 partner-Q8 compute synchronization returned. "
                "The next diagnostic mapped-host marker made its current sequence "
                "visible, but its complement remained the stale complement of the "
                "preceding high-bit activation marker; result D2H was never attempted. "
                "The first observed failure is therefore inside the diagnostic's own "
                "mapped-host marker/system-visibility boundary (or concurrent work), "
                "not the named Q8 compute binding. Remove that marker path before "
                "bracketing the production result gather."
            )
        return (
            "The global pair-0 partner-Q8 compute fence failed with "
            f"classification `{classification}`. The dynamically observed final "
            f"boundary record was `{last}`. This localizes where CUDA surfaced "
            "the reset without asserting that the named binding caused it."
        )
    direct_rows = [
        row for row in rows
        if row.get("variant") == "attention-q8-direct-gather-fence"
    ]
    if (outcomes.get("attention-q8-direct-gather-fence") == "failed" and
            direct_rows):
        row = direct_rows[-1]
        classification = row.get("q8_direct_gather_classification", "")
        compute_classification = row.get(
            "q8_compute_fence_classification", ""
        )
        if compute_classification != "all-observed-compute-fences-returned":
            last_compute = (
                row.get("q8_compute_fence_last_failure") or
                row.get("q8_compute_fence_last_armed", "")
            )
            return (
                "The marker-free direct-gather arm failed before its next result "
                "copy was armed, with compute-fence classification "
                f"`{compute_classification}` and final compute record "
                f"`{last_compute}`. No direct-gather boundary is attributed."
            )
        last = (row.get("q8_direct_gather_last_failure") or
                row.get("q8_direct_gather_last_armed", ""))
        if classification == "complete":
            return (
                "Every observed marker-free compute synchronization and direct "
                "result D2H/H2D gather returned. The reset surfaced after the last "
                f"confirmed boundary `{last}` or in concurrent work; no observed "
                "selected Q8 result-copy boundary failed."
            )
        return (
            "The marker-free direct-gather bracket failed with classification "
            f"`{classification}` after the immediately preceding partner-compute "
            "synchronization returned. Its final dynamic boundary record was "
            f"`{last}`. This identifies which synchronous result-copy boundary "
            "returned or failed without introducing mapped-host marker traffic."
        )
    attention = outcomes.get("attention-off", "not-run")
    attention_host_bounce = outcomes.get("attention-host-bounce", "not-run")
    attention_q8_host_bounce = outcomes.get(
        "attention-q8-host-bounce", "not-run"
    )
    attention_q8_async_completion = outcomes.get(
        "attention-q8-async-completion", "not-run"
    )
    attention_q8_pre_gather_fence = outcomes.get(
        "attention-q8-pre-gather-fence", "not-run"
    )
    attention_q8_activation_fence = outcomes.get(
        "attention-q8-activation-fence", "not-run"
    )
    attention_q8_phase_audit = outcomes.get(
        "attention-q8-phase-audit", "not-run"
    )
    attention_q8_targeted_phase_audit = outcomes.get(
        "attention-q8-targeted-phase-audit", "not-run"
    )
    attention_q8_l14_l15_phase_audit = outcomes.get(
        "attention-q8-l14-l15-phase-audit", "not-run"
    )
    attention_q8_l12_phase_audit = outcomes.get(
        "attention-q8-l12-phase-audit", "not-run"
    )
    query_dst = outcomes.get("attention-query-dst", "not-run")
    gather_dst = outcomes.get("attention-gather-dst", "not-run")
    both_dst = outcomes.get("attention-both-dst", "not-run")
    phase_audit = outcomes.get("attention-phase-audit", "not-run")
    end_fence = outcomes.get("attention-end-fence", "not-run")
    boundary_audit = outcomes.get("attention-row-boundary-audit", "not-run")
    production = outcomes.get("production", "not-run")
    bounce = outcomes.get("partner-bounce", "not-run")
    bounce_indexer = outcomes.get("bounce-indexer-off", "not-run")
    serialized = outcomes.get("partner-serialized", "not-run")
    indexer = outcomes.get("indexer-off", "not-run")
    attention_rows = [row for row in rows if row.get("variant") == "attention-off"]
    attention_host_bounce_rows = [
        row for row in rows
        if row.get("variant") == "attention-host-bounce"
    ]
    attention_host_bounce_failed_rows = [
        row for row in attention_host_bounce_rows
        if row.get("status") in {
            "failed-device-loss", "interrupted-prior-run-device-loss",
            "interrupted-no-result-device-loss",
        }
    ]
    attention_q8_host_bounce_rows = [
        row for row in rows
        if row.get("variant") == "attention-q8-host-bounce"
    ]
    attention_q8_host_bounce_failed_rows = [
        row for row in attention_q8_host_bounce_rows
        if row.get("status") in {
            "failed-device-loss", "interrupted-prior-run-device-loss",
            "interrupted-no-result-device-loss",
        }
    ]
    attention_q8_async_completion_rows = [
        row for row in rows
        if row.get("variant") == "attention-q8-async-completion"
    ]
    attention_q8_pre_gather_fence_rows = [
        row for row in rows
        if row.get("variant") == "attention-q8-pre-gather-fence"
    ]
    attention_q8_activation_fence_rows = [
        row for row in rows
        if row.get("variant") == "attention-q8-activation-fence"
    ]
    attention_q8_phase_audit_rows = [
        row for row in rows
        if row.get("variant") == "attention-q8-phase-audit"
    ]
    attention_q8_phase_audit_problem_rows = [
        row for row in attention_q8_phase_audit_rows
        if row.get("status") not in {"passed", "completed-no-result"}
    ]
    attention_q8_targeted_phase_audit_rows = [
        row for row in rows
        if row.get("variant") == "attention-q8-targeted-phase-audit"
    ]
    attention_q8_targeted_phase_audit_problem_rows = [
        row for row in attention_q8_targeted_phase_audit_rows
        if row.get("status") not in {"passed", "completed-no-result"}
    ]
    attention_q8_l14_l15_phase_audit_rows = [
        row for row in rows
        if row.get("variant") == "attention-q8-l14-l15-phase-audit"
    ]
    attention_q8_l14_l15_phase_audit_problem_rows = [
        row for row in attention_q8_l14_l15_phase_audit_rows
        if row.get("status") not in {"passed", "completed-no-result"}
    ]
    attention_q8_l12_phase_audit_rows = [
        row for row in rows
        if row.get("variant") == "attention-q8-l12-phase-audit"
    ]
    attention_q8_l12_phase_audit_problem_rows = [
        row for row in attention_q8_l12_phase_audit_rows
        if row.get("status") not in {"passed", "completed-no-result"}
    ]

    def l12_occurrence_mapping_verified(row: dict[str, str]) -> bool:
        return row.get("q8_phase_audit_occurrence_mapping") == "verified"

    def measured_host_bounce_checkpoint(row: dict[str, str]) -> bool:
        checkpoint = row.get("attention_host_bounce_checkpoint", "")
        return (
            checkpoint.startswith(("begin:", "complete:", "failed:")) and
            ":pos512:tokens512:home0:partner2" in checkpoint
        )

    def completed_measured_attention_host_bounce(
            row: dict[str, str]) -> bool:
        checkpoint = row.get("attention_host_bounce_checkpoint", "")
        return (
            checkpoint.startswith("complete:") and
            measured_host_bounce_checkpoint(row) and
            row.get("pair0_attention_query_copy_transport") == "host-bounce" and
            row.get("pair0_attention_gather_copy_transport") == "host-bounce" and
            row.get("pair0_attention_cache_copy_transport") == "host-bounce" and
            set(row.get(
                "pair0_attention_cache_host_bounce_classes", ""
            ).split(",")) == {"raw", "attn-comp", "index"}
        )

    def completed_measured_q8_host_bounce(row: dict[str, str]) -> bool:
        # Pair 0 reaches 64 calls during the untimed warmup.  The next durable
        # 64-call checkpoint (128) is the first proof that host-bounce Q8 work
        # completed in measured prefill.
        return (
            row.get("pair0_q8_transport") == "host-bounce" and
            int(row.get("pair0_q8_complete_checkpoint_calls", "0") or 0) >= 128
        )

    def pair0_device_loss(row: dict[str, str]) -> bool:
        for item in row.get("lost_devices", "").split(","):
            match = re.match(r"^([0-9]+)(?:@|$)", item.strip())
            if match and int(match.group(1)) in {0, 1}:
                return True
        return False

    def full_production_load(row: dict[str, str]) -> bool:
        try:
            return float(row.get("prefill_tps", "")) >= 500.0
        except (TypeError, ValueError):
            return False

    def indexer_observation(candidate_rows: list[dict[str, str]]) -> str:
        calls = sum(
            int(row.get("pair0_indexer_begin_calls", "0") or 0)
            for row in candidate_rows
        )
        traffic = sum(
            int(row.get("pair0_indexer_begin_bytes", "0") or 0)
            for row in candidate_rows
        )
        if calls:
            return (
                " Pair-0 indexer transport retained its direct-peer configuration "
                f"and durably logged {calls} begun dispatch(es) carrying "
                f"{traffic} bytes before the arm ended."
            )
        return (
            " Pair-0 indexer transport retained its direct-peer configuration, but "
            "no pair-0 indexer dispatch was durably logged before the arm ended; it "
            "cannot be cited as executed direct pair-0 traffic in this observed "
            "interval."
        )

    def topk_evidence(candidate_rows: list[dict[str, str]]) -> str:
        transports: set[str] = set()
        for row in candidate_rows:
            transports.update(filter(None, row.get(
                "pair0_attention_topk_copy_transport", ""
            ).split(",")))
        if transports and transports <= {"none", "partner-local"}:
            if transports == {"partner-local"}:
                return (
                    " Recorded row-audit markers show that top-k remained "
                    "partner-local, so no cross-device top-k payload transfer "
                    "occurred."
                )
            if transports == {"none"}:
                return (
                    " Recorded row-audit markers show that no top-k payload "
                    "transfer was required."
                )
            return (
                " Recorded row-audit markers show that top-k was either not "
                "required or remained partner-local; no cross-device top-k "
                "payload transfer occurred."
            )
        if transports:
            return (
                " Recorded row-audit markers observed top-k transport modes `" +
                ",".join(sorted(transports)) + "`."
            )
        return (
            " No top-k transfer was observed in the available durable row-audit "
            "records; because no top-k transport marker was captured, this is "
            "not proof that none occurred."
        )

    attention_host_bounce_failure_reached = any(
        measured_host_bounce_checkpoint(row)
        for row in attention_host_bounce_failed_rows
    )
    attention_host_bounce_pair0_failure_reached = any(
        measured_host_bounce_checkpoint(row) and pair0_device_loss(row)
        for row in attention_host_bounce_failed_rows
    )
    attention_host_bounce_complete_pair0_failure_reached = any(
        row.get("attention_host_bounce_checkpoint", "").startswith("complete:") and
        measured_host_bounce_checkpoint(row) and pair0_device_loss(row)
        for row in attention_host_bounce_failed_rows
    )
    attention_host_bounce_pass_verified = (
        bool(attention_host_bounce_rows) and
        all(
             row.get("attention_host_bounce_checkpoint", "").startswith(
                 "complete:"
             ) and measured_host_bounce_checkpoint(row) and
             full_production_load(row)
            for row in attention_host_bounce_rows
        )
    )
    attention_q8_host_bounce_pass_verified = (
        bool(attention_q8_host_bounce_rows) and
        all(
            completed_measured_attention_host_bounce(row) and
            completed_measured_q8_host_bounce(row) and
            full_production_load(row)
            for row in attention_q8_host_bounce_rows
        )
    )
    attention_completed_without_result = (
        bool(attention_rows) and
        any(row.get("status") == "completed-no-result" for row in attention_rows) and
        all(row.get("status") in {"passed", "completed-no-result"}
            for row in attention_rows)
    )
    end_problem_rows = [
        row for row in rows
        if (row.get("variant") == "attention-end-fence" and
            row.get("status") not in {"passed", "completed-no-result"})
    ]
    end_problem_marker = next(
        (row.get("attention_end_fence_last", "")
         for row in reversed(end_problem_rows)
         if row.get("attention_end_fence_last")),
        "",
    )
    boundary_rows = [
        row for row in rows
        if row.get("variant") == "attention-row-boundary-audit"
    ]
    boundary_problem_rows = [
        row for row in boundary_rows
        if row.get("status") not in {"passed", "completed-no-result"}
    ]
    boundary_last = boundary_problem_rows[-1] if boundary_problem_rows else {}
    boundary_end_marker = boundary_last.get("attention_end_fence_last", "")
    boundary_entry_marker = boundary_last.get("attention_entry_fence_last", "")
    boundary_phase_marker = boundary_last.get("attention_phase_audit_last", "")
    boundary_first_failure = boundary_last.get(
        "attention_audit_first_failure", ""
    )
    boundary_survived_without_result = (
        bool(boundary_rows) and
        all(row.get("status") in {"passed", "completed-no-result"}
            for row in boundary_rows) and
        all(row.get("attention_row_boundary_marker_state") == "complete"
            for row in boundary_rows)
    )
    if attention_q8_activation_fence != "not-run":
        invalid_statuses = {
            "validation-failed", "environment-invalid",
            "passed-invalidated-watch", "completed-no-result-invalidated-watch",
        }
        statuses = {
            row.get("status", "") for row in attention_q8_activation_fence_rows
        }
        device_loss_statuses = {
            "failed-device-loss", "interrupted-prior-run-device-loss",
            "interrupted-no-result-device-loss",
        }
        evidence_rows = [
            row for row in attention_q8_activation_fence_rows
            if row.get("status") in device_loss_statuses
        ]
        all_classifications = {
            row.get("q8_pre_activation_fence_classification", "")
            for row in attention_q8_activation_fence_rows
        }
        classifications = {
            row.get("q8_pre_activation_fence_classification", "")
            for row in evidence_rows
        }
        failure = next((
            row.get("q8_pre_activation_fence_last_failure", "")
            for row in reversed(evidence_rows)
            if row.get("q8_pre_activation_fence_last_failure")
        ), "")
        armed = next((
            row.get("q8_pre_activation_fence_last_armed", "")
            for row in reversed(evidence_rows)
            if row.get("q8_pre_activation_fence_last_armed")
        ), "")
        returned = next((
            row.get("q8_pre_activation_fence_last_returned", "")
            for row in reversed(evidence_rows)
            if row.get("q8_pre_activation_fence_last_returned")
        ), "")
        checkpoint = next((
            row.get("q8_pre_activation_fence_last_checkpoint", "")
            for row in reversed(evidence_rows)
            if row.get("q8_pre_activation_fence_last_checkpoint")
        ), "")
        if (statuses & invalid_statuses or
                attention_q8_activation_fence == "invalid" or
                "invalid-activation-fence-record" in all_classifications):
            return (
                "The pre-activation-fence arm failed its exact enable, tagged marker, "
                "sequence, pair mapping, binding metadata, counter, or production-path "
                "validation. Its activation record is invalid for boundary inference."
            )
        if len(classifications - {""}) > 1:
            return (
                "Repeated pre-activation-fence arms ended at different validated API "
                "boundaries (`" + "`, `".join(sorted(classifications - {""})) +
                "`). Preserve the records separately; their combination does not "
                "identify one failure locus."
            )
        classification = next(iter(classifications - {""}), "")
        if classification == "activation-source-d2h-failed":
            return (
                "The activation source D2H call was the first API in this diagnostic "
                "to report failure; destination synchronization and activation H2D "
                "were not attempted. A synchronous D2H can surface earlier source-side "
                "work, so this observation does not prove that D2H traffic caused the "
                "reset. Last record: " + failure
            )
        if classification == "activation-destination-switch-or-setup-failed":
            return (
                "Activation D2H completed, then destination context selection failed "
                "before the pre-H2D device fence or H2D. This localizes the first "
                "reported error to destination switch/setup, without identifying the "
                "underlying hardware or software cause. Last record: " + failure
            )
        if classification == "activation-pre-h2d-device-sync-failed":
            return (
                "Activation D2H and destination selection completed, but the explicit "
                "destination-wide pre-H2D synchronization first reported failure. The "
                "current activation H2D was not attempted; the synchronization may be "
                "surfacing prior destination work or a poisoned context and is not "
                "itself proven causal. Last record: " + failure
            )
        if classification == "activation-marker-channel-failure":
            return (
                "Activation D2H and destination synchronization completed before the "
                "tagged marker launch/event/validation channel failed. No current "
                "activation H2D was attempted. This identifies the observation-channel "
                "boundary, not the reset cause. Last record: " + failure
            )
        if classification == "activation-fence-confirmed-h2d-failed":
            return (
                "The source D2H, destination-wide fence, and exact tagged marker were "
                "confirmed before activation H2D was invoked; that synchronous H2D "
                "then reported failure. H2D is the first failing API for this call, "
                "but the record does not prove that H2D initiated the endpoint reset. "
                "Last record: " + failure
            )
        if classification == (
                "activation-fence-confirmed-h2d-return-not-observed-"
                "trailing-checkpoint"):
            return (
                "The final armed record and same-sequence durable sparse checkpoint "
                "confirm source D2H, destination synchronization, and the exact tagged "
                "marker immediately before activation H2D. No matching returned "
                "breadcrumb was observed before the device-loss watcher ended the "
                "process. A kill can occur after API success but before logging, so "
                "this does not prove H2D failed or caused the reset. Last checkpoint: " +
                checkpoint + "; last armed: " + armed
            )
        if classification == "activation-fence-confirmed-h2d-return-not-observed":
            return (
                "The final fflush-only armed record confirms source D2H, destination "
                "synchronization, and the exact tagged marker before activation H2D, "
                "but no matching returned breadcrumb was observed before watcher "
                "termination. This narrows the observation interval without proving "
                "that H2D failed to return or caused the reset. Last armed: " + armed
            )
        if classification == "activation-h2d-returned-subsequent-locus-unresolved":
            return (
                "Every observed activation arm has a matching returned breadcrumb, so "
                "the latest synchronous activation H2D returned successfully and no "
                "recorded activation H2D is a no-return event. The "
                "device-loss run has no later activation arm observed, leaving the "
                "subsequent locus unresolved. Any teardown inability to synchronize the "
                "partner does not retract the earlier return evidence or identify the "
                "reset cause. "
                "Last armed: " + armed +
                "; last returned: " + returned
            )
        if classification in {
                "activation-audit-state-failed",
                "activation-source-or-staging-setup-failed"}:
            return (
                "The activation diagnostic failed in its audit state or source staging/"
                "setup before a current activation D2H/H2D boundary was established. "
                "This is valid software diagnostic evidence, not an identified reset "
                "cause. Last record: " + failure
            )
        if attention_q8_activation_fence == "passed":
            return (
                "Every selected activation copy in the completed arm crossed source "
                "D2H, destination synchronization, the exact tagged marker, and a "
                "successful synchronous H2D return. The added device-wide fence and "
                "per-call logging perturb overlap, so survival does not clear the "
                "unfenced production path."
            )
        return (
            "The pre-activation-fence arm has no validated record resolving the "
            "device-loss boundary. Sparse checkpoint absence alone is not negative "
            "evidence."
        )
    if attention_q8_pre_gather_fence != "not-run":
        invalid_fence_statuses = {
            "validation-failed", "environment-invalid",
            "passed-invalidated-watch", "completed-no-result-invalidated-watch",
        }
        fence_statuses = {
            row.get("status", "") for row in attention_q8_pre_gather_fence_rows
        }
        if fence_statuses & invalid_fence_statuses or \
                attention_q8_pre_gather_fence == "invalid":
            return (
                "The pre-gather-fence arm failed its exact event/marker/counter, "
                "production-path, environment, or post-run validation. It is "
                "invalid for boundary inference; do not treat a malformed positive-"
                "looking record as evidence."
            )
        if "inconclusive-underloaded" in fence_statuses:
            return (
                "The pre-gather-fence arm completed below the 500 prefill tok/s "
                "load floor. Its records are retained, but the underloaded arm is "
                "not a valid full-load trigger comparison."
            )
        device_loss_statuses = {
            "failed-device-loss", "interrupted-prior-run-device-loss",
            "interrupted-no-result-device-loss",
        }
        evidence_rows = [
            row for row in attention_q8_pre_gather_fence_rows
            if row.get("status") in device_loss_statuses
        ]
        classifications = {
            row.get("q8_pre_gather_fence_classification", "")
            for row in evidence_rows
        }
        failure_context = next((
            row.get("q8_pre_gather_fence_last_failure", "")
            for row in reversed(evidence_rows)
            if row.get("q8_pre_gather_fence_last_failure")
        ), "")
        armed_context = next((
            row.get("q8_pre_gather_fence_last_armed", "")
            for row in reversed(evidence_rows)
            if row.get("q8_pre_gather_fence_last_armed")
        ), "")
        returned_context = next((
            row.get("q8_pre_gather_fence_last_returned", "")
            for row in reversed(evidence_rows)
            if row.get("q8_pre_gather_fence_last_returned")
        ), "")
        checkpoint_context = next((
            row.get("q8_pre_gather_fence_last_checkpoint", "")
            for row in reversed(evidence_rows)
            if row.get("q8_pre_gather_fence_last_checkpoint")
        ), "")
        if "invalid-fence-record" in classifications:
            return (
                "The device-loss arm contains a truncated or internally "
                "inconsistent Q8 pre-gather-fence record. Pair, physical-device, "
                "sequence, complement, event, D2H/H2D attempt/completion, breadcrumb "
                "order, and counter consistency are required before interpreting it. "
                "Last record: " +
                failure_context
            )
        if "pre-gather-stream-failure" in classifications:
            return (
                "The dedicated post-marker event synchronization failed before the "
                "result D2H was attempted. For this recorded call, the pending CUDA "
                "failure therefore arose in the partner default-stream completion "
                "domain (the Q8 compute/marker chain or earlier concurrent device "
                "work), not from attempting its result D2H. This identifies an API "
                "observation boundary, not the electrical or software root cause. "
                "Last record: " + failure_context
            )
        if "pre-gather-state-failure" in classifications:
            return (
                "The pre-gather diagnostic found an internally inconsistent event/"
                "sequence state before synchronizing and before attempting result "
                "D2H. This is a valid diagnostic failure record but indicates a "
                "software audit-state invariant failure, not a resolved GPU fault "
                "boundary. Last record: " + failure_context
            )
        if "marker-channel-failure" in classifications:
            return (
                "The dedicated post-marker event synchronized successfully, but the "
                "exact mapped-host sequence/complement was invalid, and no result "
                "D2H was attempted. The partner stream crossed the event boundary; "
                "this is a marker-channel integrity failure and excludes this "
                "call's result D2H attempt as the trigger. Last record: " +
                failure_context
            )
        if (
            "post-compute-confirmed-result-gather-failed-before-d2h-attempt"
            in classifications
        ):
            return (
                "The dedicated event synchronized and the exact marker matched "
                "before result-gather setup, which then failed before D2H was "
                "attempted. The current Q8 compute is positively confirmed, and "
                "this call attempted no result D2H; the host-bounce allocation/"
                "setup boundary is the first observed failing API interval. This "
                "does not by itself identify the endpoint-reset root cause. Last "
                "record: " + failure_context
            )
        if "post-compute-confirmed-result-d2h-failed" in classifications:
            return (
                "The dedicated event synchronized and the exact marker matched "
                "before result D2H was attempted; that D2H API was invoked but did "
                "not complete successfully. The current Q8 compute is positively "
                "confirmed and D2H is the first observed failing result-gather API, "
                "but this does not by itself prove the electrical or software root "
                "cause of the endpoint reset. Last record: " +
                failure_context
            )
        if (
            "post-compute-confirmed-result-d2h-complete-result-h2d-not-attempted"
            in classifications
        ):
            return (
                "The dedicated event synchronized, the exact marker matched, and "
                "the result D2H completed successfully, but result-gather failed "
                "before destination H2D was attempted. This moves the first observed "
                "failing interval after D2H completion and before H2D invocation. "
                "It does not prove that the preceding D2H traffic was physically "
                "irrelevant to a latent endpoint reset. Last record: " +
                failure_context
            )
        if "post-compute-confirmed-result-h2d-failed" in classifications:
            return (
                "The dedicated event synchronized, the exact marker matched, and "
                "the result D2H completed successfully. Destination H2D was then "
                "invoked but did not complete successfully, making H2D the first "
                "observed failing result-gather API for this recorded call. This API "
                "boundary does not by itself prove the reset's physical or software "
                "cause. Last record: " + failure_context
            )
        if (
            "post-compute-confirmed-result-h2d-complete-later-gather-failure"
            in classifications
        ):
            return (
                "The dedicated event synchronized, the exact marker matched, and "
                "both result D2H and destination H2D completed successfully before a "
                "later result-gather step reported failure. This defensive record "
                "moves the first observed failure after both copy calls without "
                "proving the cause of the endpoint reset. Last record: " +
                failure_context
            )
        if (
            "post-compute-confirmed-before-result-gather-armed-status-not-observed"
            in classifications
        ):
            return (
                "A trailing sparse checkpoint proves that the current pair-0 partner "
                "stream synchronized and its exact marker matched before result "
                "gather. No matching armed breadcrumb was observed before "
                "corroborated device loss ended the run, so whether "
                "that gather was attempted or returned is not established. This "
                "narrows the observation interval to after confirmed compute and "
                "marker validation without identifying the reset cause. Last "
                "checkpoint: " + checkpoint_context
            )
        if (
            "post-compute-confirmed-result-gather-return-not-observed"
            in classifications
        ):
            return (
                "The exact final armed record proves the current pair-0 partner "
                "stream crossed the post-compute event/marker boundary immediately "
                "before result gather. Every earlier armed sequence, if any, has a "
                "matching successful return, but no matching returned breadcrumb for "
                "the current gather was "
                "observed before corroborated device loss ended the run. A SIGKILL "
                "can land after helper success but before that "
                "breadcrumb is emitted, so this identifies the subsequent result-"
                "gather/reset observation interval without proving that the CUDA "
                "copy failed to return or caused the reset. Last armed: " + armed_context +
                "; last returned: " + returned_context
            )
        if (
            "last-confirmed-gather-returned-subsequent-locus-unresolved"
            in classifications
        ):
            return (
                "The exact final armed and returned records prove that the latest "
                "confirmed pair-0 partner-Q8 result gather returned successfully. "
                "Corroborated device loss occurred afterward and before the "
                "next armed boundary, so the subsequent locus remains unresolved; "
                "the completed gather is not a no-return event and these records do "
                "not identify the reset cause. Last armed: " + armed_context +
                "; last returned: " + returned_context
            )
        if attention_q8_pre_gather_fence == "passed":
            return (
                "Every pair-0 partner-Q8 call in the completed full-load arm crossed "
                "the pre-result-D2H event and exact-marker boundary, with zero "
                "pre-gather or post-confirmation gather failures. Event "
                "synchronization and per-call armed/returned logging perturb timing, "
                "so survival does not clear the unfenced path; compare it with the "
                "async-completion control while preserving that caveat."
            )
        if "prior-pre-gather-calls-confirmed-current-call-unresolved" in {
                row.get("q8_pre_gather_fence_classification", "")
                for row in evidence_rows}:
            return (
                "Periodic checkpoints confirm earlier pre-result-D2H boundaries, "
                "but no validated failure record resolves the current call. "
                "Checkpoint absence between periodic emissions is not negative "
                "evidence."
            )
        return (
            "The pre-gather fence armed but produced no validated record resolving "
            "the failing call. The result is inconclusive."
        )
    if attention_q8_async_completion != "not-run":
        invalid_async_statuses = {
            "validation-failed", "environment-invalid",
            "passed-invalidated-watch", "completed-no-result-invalidated-watch",
        }
        async_statuses = {
            row.get("status", "")
            for row in attention_q8_async_completion_rows
        }
        if async_statuses & invalid_async_statuses:
            return (
                "At least one async-completion repeat failed its exact marker/"
                "count/complement, production-path, environment, or post-run "
                "health validation. Invalid evidence takes precedence over any "
                "positive-looking record from another repeat; do not infer a fault "
                "boundary from this mixed set."
            )
        if "inconclusive-underloaded" in async_statuses:
            return (
                "At least one async-completion repeat completed below the 500 "
                "prefill tok/s load floor. Its boundary records are retained, but "
                "underloaded evidence takes precedence over any positive-looking "
                "record from another repeat and is invalid as a full-load trigger "
                "comparison."
            )
        if attention_q8_async_completion == "invalid":
            return (
                "The async-completion arm returned but failed its exact marker/count/"
                "complement or production-path validation. Do not infer a fault "
                "boundary from it; inspect the summary-state mismatch."
            )
        device_loss_statuses = {
            "failed-device-loss", "interrupted-prior-run-device-loss",
            "interrupted-no-result-device-loss",
        }
        evidence_rows = [
            row for row in attention_q8_async_completion_rows
            if row.get("status") in device_loss_statuses
        ]
        classifications = {
            row.get("q8_async_completion_classification", "")
            for row in evidence_rows
        }
        failure_context = next((
            row.get("q8_async_completion_last_failure", "")
            for row in reversed(evidence_rows)
            if row.get("q8_async_completion_last_failure")
        ), "")
        if ("invalid-failure-record" in classifications or
                "invalid-checkpoint-record" in classifications):
            return (
                "The device-loss arm contains a truncated or internally inconsistent "
                "Q8 async-completion record. It is invalid for boundary inference; "
                "invalid evidence takes precedence over a positive-looking record "
                "from another repeat, and no logged interpretation token is trusted "
                "without pair, physical-device, sequence, complement, counter, and "
                "event consistency. Last record: " + failure_context
            )
        if "current-call-post-compute-confirmed" in classifications:
            return (
                "The non-layer-targeted pair-0 Q8 marker positively confirms that "
                "a recorded Q8 call in the device-loss arm crossed its post-compute "
                "default-stream boundary before its API failure surfaced. CUDA "
                "therefore executed that stream "
                "through the marker after the Q8 compute submission. This narrows the "
                "first positively observed failing boundary for that recorded call to "
                "after the marker, but "
                "does not prove an electrical or software root cause and does not "
                "exclude the compute load as a trigger for a latent endpoint fault. "
                "The record is not assumed to be the terminal process call because "
                "the caller can recover from some copy failures. Last record: " +
                failure_context
            )
        if "current-call-marker-observed-event-unavailable" in classifications:
            return (
                "The mapped pair-0 Q8 slot contained the recorded call's exact "
                "sequence and complement after the API failure surfaced, but the "
                "dedicated CUDA event could not be queried on the unhealthy "
                "device/context. This is a positive hardware breadcrumb that the "
                "marker write became host-visible after compute submission; without "
                "a successful CUDA completion primitive it is not formal "
                "same-stream completion proof and does not identify root cause. The "
                "record is not assumed to be the terminal process call. Last record: "
                + failure_context
            )
        if "current-call-post-compute-event-confirmed-marker-invalid" in classifications:
            return (
                "The dedicated event positively confirms that the partner default "
                "stream crossed the post-compute marker submission for a recorded "
                "Q8 call, but the mapped-host sequence/complement was invalid. This is "
                "positive post-compute boundary evidence and a marker-channel "
                "integrity failure, not evidence that Q8 compute failed and not a "
                "valid clean arm. The record is not assumed to be the terminal "
                "process call. Last record: " + failure_context
            )
        if "current-call-inconclusive" in classifications:
            return (
                "The Q8 audit captured an API failure but no matching positive marker "
                "for that recorded call. Because the CUDA context/device was "
                "unhealthy, absence is inconclusive: it is not evidence that the Q8 "
                "compute itself failed. "
                "Use the durable prior-call checkpoint only as the last confirmed "
                "boundary. Last record: " + failure_context
            )
        if attention_q8_async_completion == "passed":
            return (
                "Every pair-0 partner-Q8 call in the completed full-load arm had "
                "begun=submitted=confirmed with a valid final complement and a "
                "healthy synchronized partner context. The added mapped-host marker "
                "and event are a timing/PCIe perturbation, so survival does not clear "
                "the original path; compare it with the otherwise identical "
                "attention-q8-host-bounce no-marker control."
            )
        if "prior-calls-confirmed-current-call-unresolved" in classifications:
            return (
                "The arm durably confirms earlier pair-0 Q8 post-compute boundaries, "
                "but it captured no validated positive failure record. The process's "
                "terminal boundary remains unresolved; a missing marker is not "
                "negative evidence."
            )
        return (
            "The async-completion arm armed but produced no positive boundary record "
            "that resolves the failing call. Its result is inconclusive, not evidence "
            "against partner-Q8 completion."
        )
    if boundary_audit == "passed" or boundary_survived_without_result:
        recovery_note = (
            " The benchmark and healthy post-run snapshot completed even though "
            "the wrapper omitted its final result record."
            if boundary_survived_without_result and boundary_audit != "passed"
            else ""
        )
        return (
            "The combined attention-row boundary audit and healthy post-run snapshot "
            "passed." + recovery_note + " "
            "It host-confirmed pair-0 completion after layer 17 attention, again "
            "immediately before layer 18 query copy, and after every layer 18 "
            "row-split phase. Because the added boundaries perturb overlap and "
            "instantaneous load, this narrows the trigger to the removed timing "
            "envelope but does not identify a layer-specific software defect."
        )
    if boundary_audit == "invalid":
        return (
            "The combined attention-row boundary audit process returned but failed "
            "its "
            "production-path or marker-order validation. It is invalid for causal "
            "inference; inspect the missing durable marker before another run."
        )
    if boundary_audit in {"failed", "incomplete"}:
        if (boundary_first_failure.startswith("end-fence:") or
                not boundary_end_marker.startswith("complete:pair:")):
            return (
                "The combined audit did not host-confirm the layer-17 end fence. "
                "Its first failure-class marker and last durable end-fence marker "
                "define the observation boundary; "
                "this run does not move the prior bracket into layer 18."
            )
        if (boundary_first_failure.startswith("entry-fence:") or
                (not boundary_first_failure and
                 boundary_entry_marker.startswith((
                     "failed:", "sync-failed:", "device-switch-failed:",
                     "submit-failed:")))):
            return (
                "The layer-17 attention end fence completed on both devices, but the "
                "layer-18 row-launch/pre-query fence failed before layer-18 query copy "
                "was submitted. "
                "The pending error therefore entered the CUDA observation window after "
                "the layer-17 attention result gather and by the layer-18 pre-query "
                "boundary. That interval includes layer-17 tail/MoE/Q8 work and "
                "upstream layer-18 QKV/cache/indexer preparation; it does not prove a "
                "layer-18 kernel bug. This is not a transformer-layer entry fence."
            )
        if boundary_entry_marker.startswith("complete:pair:"):
            if (boundary_first_failure.startswith("phase-audit:") or
                    (not boundary_first_failure and
                     boundary_phase_marker.startswith((
                         "submit-failed:", "device-switch-failed:",
                         "sync-failed:")))):
                return (
                    "Both the layer-17 end fence and layer-18 row-launch/pre-query "
                    "fence completed. The first failure-class marker is inside the "
                    "named layer-18 row-split phase, so a secondary cleanup failure "
                    "cannot overwrite that boundary. The entry fence excludes earlier "
                    "queued work from being the already-pending CUDA error. The marker "
                    "is an observation boundary, not by itself proof of a layer-specific "
                    "hardware or kernel defect."
                )
            if boundary_phase_marker.startswith("complete:result-gather:"):
                if "lost-device" in boundary_last.get("watch_status", ""):
                    return (
                        "The combined audit host-confirmed both boundary fences and all "
                        "layer-18 row-split phases before the telemetry watcher "
                        "independently detected device loss. The first observed error "
                        "is therefore downstream of layer-18 attention completion, "
                        "while a delayed physical effect from preceding traffic remains "
                        "possible."
                    )
                if boundary_audit == "failed":
                    return (
                        "The combined audit host-confirmed both boundary fences and all "
                        "layer-18 row-split phases before the benchmark later returned "
                        "failure. No lost-device watcher record corroborates a GPU "
                        "failure, so a user interruption or ordinary process failure "
                        "must not be assigned a hardware boundary."
                    )
                return (
                    "The combined audit host-confirmed both boundary fences and all "
                    "layer-18 row-split phases before the arm ended without a validated "
                    "outcome. With no lost-device watcher record, Ctrl-C or another "
                    "interruption is not evidence of a GPU failure."
                )
            return (
                "The layer-17 end and layer-18 row-launch/pre-query fences completed, "
                "and the last "
                "durable layer-18 phase marker is the current observation boundary. "
                "Inspect that exact marker and GPU health before narrowing further."
            )
        return (
            "Layer-17 attention completion was host-confirmed, but no complete "
            "layer-18 row-launch/pre-query boundary was recorded. The remaining "
            "interval is after layer-17 attention and through upstream layer-18 "
            "QKV/cache/indexer preparation, before query-copy admission; inspect the "
            "last marker and GPU health without assigning a layer-specific cause."
        )
    if end_fence == "passed":
        return (
            "The full production path survived after one completion boundary was "
            "added only after the selected pair's final row-split result gather. "
            "Every row-split copy and kernel before that boundary retained production "
            "overlap. This is diagnostic evidence that overlap with later layer-tail "
            "or stage-handoff work is part of the trigger, not a proposal to disable "
            "row splitting."
        )
    if end_fence in {"failed", "incomplete"}:
        if end_problem_marker.startswith("complete:pair:"):
            later_state = (
                "before this arm later failed"
                if end_fence == "failed"
                else "although this arm has no validated final outcome"
            )
            return (
                "The end-only fence completed on both partner and home "
                f"{later_state}. All queued pair-0 work through the selected final "
                "row-split result gather was therefore host-confirmed complete. The "
                "first subsequent CUDA observation lies in downstream layer-tail or "
                "stage-handoff work. This does not prove that the preceding traffic or "
                "load could not trigger a delayed physical fault."
            )
        if end_problem_marker.startswith(("failed:partner:",
                                          "sync-failed:partner:",
                                          "device-switch-failed:partner:")):
            return (
                "The partner synchronization was the first failing end-fence "
                "boundary. An error was already pending in pair-0 work through the "
                "selected result gather. Home synchronization was deliberately not "
                "attempted after that first failure because a poisoned context would "
                "make its result secondary."
            )
        if end_problem_marker.startswith(("failed:home:",
                                          "sync-failed:home:",
                                          "device-switch-failed:home:")):
            return (
                "The partner end-fence completed, but the following home boundary "
                "failed. This localizes the first observed error to queued home-side "
                "work or completion of the partner-to-home result gather, before "
                "downstream layer-tail or stage-handoff work."
            )
        state = "failed" if end_fence == "failed" else "has no validated final outcome"
        return (
            f"The end-only row-split fence arm {state}. Its last durable partner, "
            "home, or pair fence record remains the observation boundary; determine "
            "GPU health and inspect that marker before causal attribution."
        )
    if end_fence == "invalid":
        return (
            "The end-only row-split fence process returned but failed its production-"
            "path validation. Inspect the missing fence marker before causal inference."
        )
    if phase_audit == "passed":
        return (
            "All production row-split operations and traffic survived after adding "
            "four completion checkpoints to only the selected pair/layer/frontier "
            "dispatch. This does not make row splitting dispensable. The checkpoints "
            "deliberately perturb overlap and instantaneous load, making that broader "
            "timing/power/traffic envelope part of the trigger and preserving the "
            "checkpoint log for the next narrowing step."
        )
    if phase_audit == "failed":
        return (
            "The targeted row-split phase-audit arm failed. The last durable "
            "query-copy, partner-attention, home-attention, or result-gather event "
            "in its production log, if present, is the observation boundary for "
            "the device loss."
        )
    if phase_audit == "invalid":
        return (
            "The targeted row-split phase-audit process returned but failed its "
            "production-path validation. Its result is invalid for causal inference; "
            "inspect the missing marker before running another arm."
        )
    if phase_audit == "incomplete":
        return (
            "The targeted row-split phase-audit arm has no validated final outcome. "
            "Its last durable query-copy, partner-attention, home-attention, or "
            "result-gather record, if present, remains the observation boundary; "
            "determine GPU health before classifying the arm as passed or failed."
        )
    if attention_q8_l12_phase_audit == "passed":
        if not any(
                l12_occurrence_mapping_verified(row)
                for row in attention_q8_l12_phase_audit_rows):
            return (
                "The layer-12 arm reports completion, but its durable records do "
                "not prove warmup occurrence 1 was skipped and measured occurrence "
                "2 was selected. Treat the result as validation-inconsistent rather "
                "than as survival of the intended first measured target."
            )
        return (
            "The exact occurrence-2 layer-12 audit completed at full production "
            "load with a healthy post-run snapshot. Its clean pre-compute, compute, "
            "and result-gather boundaries show that this narrowly fenced first "
            "measured call can survive; they do not prove the unfenced production "
            "overlap is safe or that layer 12 cannot contribute a delayed physical "
            "effect."
        )
    if attention_q8_l12_phase_audit == "invalid":
        return (
            "The exact occurrence-2 layer-12 arm failed environment, production-"
            "path, tuple, occurrence, or marker validation. It is invalid for causal "
            "inference; inspect the missing hard gate before another GPU run."
        )
    if attention_q8_l12_phase_audit == "underloaded":
        return (
            "The exact occurrence-2 layer-12 arm completed below the required "
            "production-load floor. It cannot be used to exclude the full-load "
            "trigger."
        )
    if attention_q8_l12_phase_audit in {"failed", "incomplete"}:
        problem_rows = attention_q8_l12_phase_audit_problem_rows
        corroborated = [
            row for row in problem_rows
            if (
                row.get("status") in {
                    "failed-device-loss", "interrupted-prior-run-device-loss",
                    "interrupted-no-result-device-loss",
                } or row.get("watch_status") == "lost-device-detected" or
                row.get("post_health") == "unhealthy"
            )
        ]
        if not corroborated:
            return (
                "The layer-12 occurrence-gated arm ended without an independent "
                "lost-device watcher record or unhealthy post-run snapshot. Its "
                "last marker is retained, but the result is not causal GPU-loss "
                "evidence. It was configured to skip warmup occurrence 1 and select "
                "measured occurrence 2, but configuration is not proof that the "
                "target was reached."
                + indexer_observation(problem_rows)
            )
        row = corroborated[-1]
        if not l12_occurrence_mapping_verified(row):
            return (
                "Device loss is independently corroborated, but the durable log "
                "does not prove both the warmup skip and the measured occurrence-2 "
                "selection for the exact layer-12 tuple. Do not assign the loss to "
                "that target or interpret any unverified phase marker as the first "
                "measured call."
                + indexer_observation(problem_rows)
            )
        scope = (
            " The durable ordering proves occurrence 1 was the skipped warmup call "
            "and occurrence 2 was the first measured call of exact tuple "
            "`tensor:blk.12.attn_output_b.weight@143236281600`; pair-0 attention "
            "and Q8 payloads remain host-bounced, the 50/50 row split and all "
            "partner compute remain enabled, and pair 1 remains direct."
            + indexer_observation(problem_rows)
        )
        classification = row.get("q8_phase_audit_classification", "")
        checkpoint = row.get("q8_phase_audit_last_checkpoint", "")
        if classification == "pre-target-partner-error":
            return (
                "The first measured layer-12 target reached its activation-copy "
                "boundary, but the partner pre-compute synchronization found an "
                "already-pending CUDA error. This excludes layer-12 Q8 compute and "
                "its result gather as the operation that first created that pending "
                "error; earlier or concurrent layer-12 attention/partner work remains "
                "inside the trigger window." + scope
            )
        if classification == "target-compute-or-concurrent-partner-work":
            return (
                "The layer-12 partner pre-compute boundary was clean and its Q8 "
                "projection was submitted, but the post-compute synchronization "
                "failed. The first synchronous error lies in that compute interval "
                "or concurrent partner work, before the result gather." + scope
            )
        if classification.startswith("result-"):
            return (
                "The exact layer-12 Q8 compute completed behind a clean partner "
                "synchronization before CUDA first reported failure in the result "
                "host-bounce path. This sharply separates the projection from the "
                "first observed copy boundary, but does not prove that the transfer "
                "electrically initiated the endpoint loss." + scope
            )
        if "event=result-complete" in checkpoint:
            return (
                "The exact first measured layer-12 activation, pre-compute boundary, "
                "Q8 compute, and result gather all completed before the pair later "
                "became unhealthy. The failure moved past this narrowly fenced call; "
                "a delayed physical effect or different concurrent work remains in "
                "scope." + scope
            )
        return (
            "The exact first measured layer-12 arm lost the pair without a complete "
            "classified phase-failure marker. Preserve its final durable checkpoint "
            "as the observation boundary instead of assigning the fault to the "
            "result-copy line." + scope
        )
    if attention_q8_l14_l15_phase_audit in {"failed", "incomplete"}:
        problem_rows = attention_q8_l14_l15_phase_audit_problem_rows
        corroborated_problem_rows = [
            row for row in problem_rows
            if (
                row.get("status") in {
                    "failed-device-loss", "interrupted-prior-run-device-loss",
                    "interrupted-no-result-device-loss",
                } or row.get("watch_status") == "lost-device-detected" or
                row.get("post_health") == "unhealthy"
            )
        ]
        classifications = sorted({
            row.get("q8_window_classification", "")
            for row in corroborated_problem_rows
            if row.get("q8_window_classification", "") not in {
                "", "inconclusive"
            }
        })
        window_scope = (
            " The cumulative window targets the exact tuples "
            "`tensor:blk.14.attn_output_b.weight@143571266304` and "
            "`tensor:blk.15.attn_output_b.weight@143723876608`. Pair-0 "
            "attention and Q8 payloads remain host-bounced, the 50/50 row split "
            "and all partner compute remain enabled, and pair 1 remains direct."
            + indexer_observation(problem_rows)
        )
        if not corroborated_problem_rows:
            return (
                "The cumulative layer-14/layer-15 arm ended without a watcher record "
                "or unhealthy post-run snapshot corroborating device loss. Its last "
                "durable completed binding and any later partial-chain marker are "
                "preserved, but a Ctrl-C or ordinary process failure is not causal GPU-"
                "loss evidence."
                + window_scope
            )
        if len(classifications) > 1:
            return (
                "Repeated cumulative layer-14/layer-15 arms produced different "
                "first-boundary classifications (`" + "`, `".join(classifications) +
                "`). Preserve each repeat separately; the combined result does not "
                "support one causal boundary." + window_scope
            )
        classified_row = next((
            row for row in reversed(corroborated_problem_rows)
            if row.get("q8_window_classification", "") not in {
                "", "inconclusive"
            }
        ), {})
        classification = classified_row.get(
            "q8_window_classification", "inconclusive"
        )
        if classification in {"layer14-activation-copy", "before-layer14-compute"}:
            return (
                "The cumulative audit surfaced its first synchronous error before "
                "layer 14 partner compute was submitted. The boundary includes the "
                "layer-14 activation-copy helper and earlier/concurrent partner work; "
                "it does not assign the fault to layer-14 compute or result gather."
                + window_scope
            )
        if classification == "layer14-compute-submit":
            return (
                "The layer-14 pre-compute boundary completed, but submission of the "
                "layer-14 attention-output projection failed. The first observed "
                "error is after the clean pre-boundary and before a completed target "
                "kernel; it does not identify the physical endpoint."
                + window_scope
            )
        if classification == "layer14-compute-or-concurrent-partner-work":
            return (
                "The layer-14 pre-compute boundary completed and its projection was "
                "submitted, but the post-compute partner synchronization failed. The "
                "first synchronous error lies in that compute interval or concurrent "
                "partner work, before the layer-14 result copy."
                + window_scope
            )
        if classification.startswith("layer14-result-"):
            return (
                "Layer 14 completed both compute boundaries before its result-copy "
                "helper failed. This localizes where CUDA first surfaced the error, "
                "but does not prove that the transfer caused the endpoint loss or "
                "distinguish GPU, PCIe, host-memory, power, and driver causes."
                + window_scope
            )
        if classification == "between-layer14-result-and-layer15-compute":
            return (
                "A layer-14 result chain completed before the layer-15 activation or "
                "pre-compute boundary failed. The first observed error is therefore "
                "after completed layer-14 result handling and before layer-15 compute "
                "was submitted; intervening/concurrent partner work remains in scope."
                + window_scope
            )
        if classification == "layer15-compute-submit-after-layer14":
            return (
                "Layer 14 completed, and the layer-15 pre-compute boundary was reached, "
                "but layer-15 projection submission failed. This separates the failure "
                "from the completed layer-14 result path without assigning a physical "
                "root cause."
                + window_scope
            )
        if classification == "layer15-compute-or-concurrent-partner-work":
            return (
                "Layer 14 completed and layer-15 compute was submitted after a clean "
                "pre-compute boundary, but layer-15 post-compute synchronization "
                "failed. The first synchronous error lies in layer-15 compute or "
                "concurrent partner work, not its result copy."
                + window_scope
            )
        if classification.startswith("layer15-result-"):
            return (
                "Both layer-14 and layer-15 compute boundaries completed before the "
                "layer-15 result-copy helper failed. This is the strongest software "
                "observation boundary for that transfer call, but it is still not proof "
                "that PCIe transfer caused the physical endpoint loss."
                + window_scope
            )
        if classification == "both-targets-complete":
            return (
                "At least one complete layer-14/layer-15 target pair crossed every "
                "activation, compute, and result boundary before the arm later lost a "
                "device or ended. The first observed failure moved downstream of the "
                "instrumented window, arguing against one unique layer-14 or layer-15 "
                "binding defect; a cumulative workload/overlap trigger or delayed "
                "physical effect remains possible."
                + window_scope
            )
        if classification == "measured-target-window-not-reached":
            first_layer = classified_row.get("first_gpu_layer_failure", "")
            layer_note = (
                f" The first engine failure was `{first_layer}`."
                if first_layer else ""
            )
            return (
                "Both exact layer-14/layer-15 chains completed only during the "
                "512-token warmup. The measured 32K pipeline then failed before its "
                "first layer-14 marker, so the earlier summary's claim that failure "
                "moved downstream of the instrumented window was incorrect."
                + layer_note +
                " The measured target window was never reached; this result instead "
                "supports phase/scheduling or cumulative-state dependence and "
                "requires bracketing the earlier measured layer-12 boundary."
                + window_scope
            )
        if classification.startswith("layer14-partial-"):
            return (
                "A prior target chain completed, but the final durable observation is "
                "inside a later layer-14 chain with no explicit phase-failure marker. "
                "The failure therefore did not demonstrably move downstream of the "
                "instrumented window; the exact last checkpoint remains the boundary."
                + window_scope
            )
        if classification == "layer14-complete-layer15-not-complete":
            return (
                "The last durable completed target is layer 14, while layer 15 did not "
                "complete and emitted no classified phase-failure marker. The current "
                "observation window is after layer-14 result completion and within or "
                "before layer-15 handling; do not relabel the disappearance as a proven "
                "layer-15 transfer failure."
                + window_scope
            )
        if classification == "layer15-partial-without-confirmed-layer14":
            return (
                "The final durable observation is a partial layer-15 chain without "
                "enough completed layer-14 chains to establish the intended paired "
                "boundary. Preserve the raw ordering, but do not make a layer-specific "
                "causal inference from this arm."
                + window_scope
            )
        return (
            "The cumulative layer-14/layer-15 arm ended without a classified target "
            "failure. Its separately recorded last completed binding and first partial "
            "chain are the observation boundary; the result is otherwise inconclusive."
            + window_scope
        )
    if attention_q8_l14_l15_phase_audit == "passed":
        return (
            "The cumulative layer-14/layer-15 Q8 audit completed all 65 chains for "
            "each exact binding at the validated production-load floor. Survival means "
            "the workload can complete when both adjacent overlap windows are perturbed; "
            "it does not prove either window caused prior failures, identify the root "
            "cause, or establish a production mitigation."
            + indexer_observation(attention_q8_l14_l15_phase_audit_rows)
        )
    if attention_q8_l14_l15_phase_audit == "underloaded":
        return (
            "The cumulative layer-14/layer-15 Q8 audit completed below the required "
            "500 prefill tok/s floor. Its marker chains remain useful, but underloaded "
            "survival cannot identify the trigger."
        )
    if attention_q8_l14_l15_phase_audit == "invalid":
        return (
            "The cumulative layer-14/layer-15 Q8 audit failed paired-target marker, "
            "production-path, throughput, or health validation and is invalid for "
            "causal comparison."
        )
    if attention_q8_targeted_phase_audit in {"failed", "incomplete"}:
        classified_row = next((
            row for row in attention_q8_targeted_phase_audit_problem_rows
            if row.get("q8_phase_audit_classification") not in {
                "", "inconclusive"
            }
        ), {})
        classification = classified_row.get(
            "q8_phase_audit_classification", "inconclusive"
        )
        target_scope = (
            " The audit target is exactly `tensor:blk.14.attn_output_b.weight` "
            "at weight offset 143571266304. Pair-0 attention and Q8 payloads "
            "remained host-bounced, row splitting and partner compute remained "
            "enabled, the pair-0 indexer route was outside the host-bounce "
            "override, and pair 1 remained direct."
            + indexer_observation(attention_q8_targeted_phase_audit_problem_rows)
        )
        if classification == "pre-target-partner-error":
            return (
                "The targeted Q8 audit failed its pre-compute partner "
                "synchronization before the layer-14 attention-output projection "
                "was submitted. The first synchronous error therefore surfaced "
                "before the target submission and is consistent with earlier "
                "partner-side work or an already poisoned/lost partner device; it "
                "does not prove that queued work caused the fault or assign it to "
                "the target projection or result D2H copy."
                + target_scope
            )
        if classification == "target-compute-or-concurrent-partner-work":
            return (
                "The targeted Q8 audit completed the pre-compute partner boundary, "
                "submitted the layer-14 attention-output projection, then failed its "
                "post-compute partner synchronization before result gather. The first "
                "synchronous failure is in that projection or partner work submitted "
                "concurrently after the pre-compute boundary, not in its result D2H "
                "copy."
                + target_scope
            )
        if classification == "result-d2h-pcie-host-bounce-transfer":
            return (
                "The targeted Q8 audit completed both pre- and post-compute partner "
                "synchronizations for the layer-14 attention-output projection, then "
                "recorded the first failure in its result-gather D2H call. This "
                "localizes the first synchronous failure to that partner-to-host "
                "transfer call, while leaving the GPU endpoint, PCIe path, host-memory "
                "path, and driver as unresolved physical causes."
                + target_scope
            )
        if classification in {
                "result-h2d-pcie-host-bounce-transfer",
                "result-host-bounce-allocation",
                "result-host-bounce-copy-after-compute"}:
            detail = {
                "result-h2d-pcie-host-bounce-transfer":
                    "home-device H2D leg of the target result transfer",
                "result-host-bounce-allocation":
                    "host-bounce allocation/setup for the target result transfer",
                "result-host-bounce-copy-after-compute":
                    "target host-bounce result helper, with its exact leg unrecorded",
            }[classification]
            return (
                "The targeted Q8 audit completed both target compute boundaries "
                f"before the first synchronous failure in the {detail}."
                + target_scope
            )
        completed_target = any(
            "event=result-complete" in row.get(
                "q8_phase_audit_last_checkpoint", ""
            )
            for row in attention_q8_targeted_phase_audit_problem_rows
        )
        if completed_target:
            return (
                "At least one targeted layer-14 attention-output sequence completed "
                "its pre-compute boundary, projection, post-compute boundary, and "
                "result gather before the arm later ended without a classified target "
                "phase failure. The first observed failure lies outside that recorded "
                "target sequence, although a delayed physical effect remains possible."
                + target_scope
            )
        return (
            "The targeted Q8 phase-audit arm ended without a complete marker chain "
            "or a classified target failure. Its last durable targeted checkpoint is "
            "the observation boundary; the result is otherwise inconclusive."
            + target_scope
        )
    if attention_q8_targeted_phase_audit == "passed":
        return (
            "The targeted layer-14 attention-output Q8 phase-audit arm completed at "
            "the validated production-load floor without a durable target failure. "
            "Because only the exact target projection was synchronized and durably "
            "logged, this run shows that the full workload can survive with that "
            "target overlap window perturbed. It does not by itself prove that the "
            "target window caused the prior failures, identify a software defect, or "
            "establish a production mitigation."
            + indexer_observation(attention_q8_targeted_phase_audit_rows)
        )
    if attention_q8_targeted_phase_audit == "underloaded":
        return (
            "The targeted layer-14 attention-output Q8 phase-audit arm completed "
            "below the required 500 prefill tok/s floor. Its markers remain useful, "
            "but survival under reduced load cannot identify the trigger."
        )
    if attention_q8_targeted_phase_audit == "invalid":
        return (
            "The targeted Q8 phase-audit arm failed target-marker, production-path, "
            "throughput, or health validation and is invalid for causal comparison."
        )
    if attention_q8_phase_audit in {"failed", "incomplete"}:
        classified_row = next((
            row for row in attention_q8_phase_audit_problem_rows
            if row.get("q8_phase_audit_classification") not in {"", "inconclusive"}
        ), {})
        classification = classified_row.get(
            "q8_phase_audit_classification", "inconclusive"
        )
        common_scope = (
            " The transport cut covers pair-0 attention-owned and Q8-partner "
            "payload routes only; it does not assert that every possible direct "
            "pair-0 peer route was removed."
            + indexer_observation(attention_q8_phase_audit_problem_rows)
            + " Pair-1 traffic retained direct-peer transport."
        )
        if classification == "pre-target-partner-error":
            return (
                "The Q8 phase audit failed its explicit pre-compute partner "
                "synchronization before the audited projection was submitted. The "
                "first synchronous error therefore surfaced before that submission "
                "and is consistent with earlier partner-side work or an already "
                "poisoned/lost partner device; it does not prove that queued work "
                "caused the fault or assign it to the audited projection/result copy."
                + common_scope
            )
        if classification == "target-compute-or-concurrent-partner-work":
            return (
                "The Q8 phase audit completed its pre-compute partner boundary, "
                "recorded compute submission, then failed the post-compute partner "
                "synchronization before result gather. The first synchronous failure "
                "is therefore in the audited compute or partner work submitted "
                "concurrently after the pre-compute boundary, not in the result D2H "
                "host-bounce copy."
                + common_scope
            )
        if classification == "result-d2h-pcie-host-bounce-transfer":
            return (
                "The Q8 phase audit durably completed the partner compute "
                "synchronization for the same sequence, then recorded the first "
                "failure in the result-gather D2H copy. This localizes the first "
                "synchronous failure to the partner-to-host PCIe leg of the pinned-"
                "host-bounce result transfer rather than to the audited partner "
                "compute. It does not by itself distinguish the GPU endpoint, PCIe "
                "path, host-memory path, or driver as the physical root cause."
                + common_scope
            )
        if classification in {
                "result-h2d-pcie-host-bounce-transfer",
                "result-host-bounce-allocation",
                "result-host-bounce-copy-after-compute"}:
            detail = {
                "result-h2d-pcie-host-bounce-transfer":
                    "home-device H2D leg of the pinned-host-bounce result transfer",
                "result-host-bounce-allocation":
                    "host-bounce allocation/setup for the result transfer",
                "result-host-bounce-copy-after-compute":
                    "host-bounce result-copy helper, with its D2H/H2D leg unrecorded",
            }[classification]
            return (
                "The Q8 phase audit durably completed the partner compute "
                "synchronization for the same sequence before the result copy "
                f"failed. The first synchronous failure is in the {detail}, not in "
                "the audited partner compute."
                + common_scope
            )
        return (
            "The Q8 phase-audit arm ended without a marker sequence that separates "
            "partner compute synchronization from result-copy failure. Its last "
            "durable checkpoint and first failure context are the observation "
            "boundary; the current evidence is otherwise inconclusive."
            + common_scope
        )
    if attention_q8_phase_audit == "passed":
        return (
            "The Q8 phase-audit arm completed without a durable Q8 phase failure. "
            "Because its explicit partner synchronization and per-phase fsync markers "
            "perturb production timing, this is diagnostic survival rather than a "
            "causal pass."
            + indexer_observation(attention_q8_phase_audit_rows)
        )
    if attention_q8_phase_audit == "underloaded":
        return (
            "The Q8 phase-audit arm completed below the required production-load "
            "floor. Its checkpoint sequence remains diagnostic evidence, but survival "
            "under reduced load cannot identify the trigger."
        )
    if attention_q8_phase_audit == "invalid":
        return (
            "The Q8 phase-audit arm failed production-path, marker, throughput, or "
            "health validation and is invalid for causal comparison. Preserve its raw "
            "phase markers, but do not treat it as a safe pass."
        )
    if (attention_q8_host_bounce == "passed" and
            attention_q8_host_bounce_pass_verified):
        return (
            "Pair 0 completed the full production-shaped arm with durable measured "
            "completion evidence for both attention-owned and Q8 partner "
            "host-bounce routes. The 50/50 attention row split and both partner "
            "compute paths remained enabled. This transport cut covers only pair-0 "
            "attention-owned and Q8-partner payload routes, not every possible direct "
            "pair-0 peer route. Pair-1 traffic remained direct peer."
            + indexer_observation(attention_q8_host_bounce_rows) +
            " This implicates one of the "
            "removed pair-0 direct-peer transport paths or the timing/load envelope "
            "changed by staging both through host memory, but is not by itself a "
            "power-matched causal proof or a production mitigation."
        )
    if attention_q8_host_bounce == "failed":
        failure_context = next((
            row.get("host_bounce_failure_context", "")
            for row in attention_q8_host_bounce_failed_rows
            if row.get("host_bounce_failure_context", "")
        ), "")
        context_sentence = (
            f" The first contextual host-bounce failure was `{failure_context}`."
            if failure_context else ""
        )
        verified_pair0_failure = any(
            pair0_device_loss(row) and
            completed_measured_attention_host_bounce(row) and
            completed_measured_q8_host_bounce(row)
            for row in attention_q8_host_bounce_failed_rows
        )
        if verified_pair0_failure:
            return (
                "The failing arm durably completed measured pair-0 host-bounce "
                "checkpoints for both attention-owned traffic and Q8 partner "
                "traffic before pair-0 device loss. Thus direct pair-0 production "
                "attention-owned and Q8-partner payload transport is unnecessary "
                "for this observed loss. This transport cut did not claim to remove "
                "every possible direct pair-0 route. Pair-1 traffic remained direct "
                "peer and was not removed."
                + indexer_observation(attention_q8_host_bounce_failed_rows) +
                " Partner attention and Q8 compute, aggregate load, pair-1 traffic, "
                "or a delayed physical effect therefore remain."
                + topk_evidence(attention_q8_host_bounce_failed_rows)
                + context_sentence
            )
        if not any(pair0_device_loss(row)
                   for row in attention_q8_host_bounce_failed_rows):
            return (
                "The combined host-bounce arm failed, but its durable loss record "
                "does not identify physical GPU 0 or 1 in pair 0. Pair 1 retained "
                "direct peer traffic, so this is real failure evidence but cannot "
                "show that either cut pair-0 payload route was unnecessary."
                + context_sentence
            )
        return (
            "Pair-0 device loss was corroborated in the combined host-bounce arm, "
            "but the durable records do not prove that both transport cuts completed "
            "during measured prefill. A 64-call Q8 checkpoint or an attention "
            "checkpoint at pos=0 is warmup evidence only. Do not infer that the "
            "cut pair-0 attention/Q8 payload routes were unnecessary. The cut covered "
            "only attention-owned and Q8-partner payload routes, not every possible "
            "direct pair-0 route. Pair-1 traffic retained direct peer transport."
            + indexer_observation(attention_q8_host_bounce_failed_rows)
            + context_sentence
        )
    if attention_q8_host_bounce == "passed":
        return (
            "The combined pair-0 host-bounce arm was recorded as passed, but it "
            "lacks durable measured completion evidence for both host-bounce routes "
            "at the required >=500 prefill tok/s load. Treat it as inconclusive; "
            "warmup-only or underloaded evidence does not validate the transport "
            "cut. Pair-1 traffic remained direct."
            + indexer_observation(attention_q8_host_bounce_rows)
        )
    if attention_q8_host_bounce == "invalid":
        return (
            "The combined pair-0 host-bounce arm failed its production-path, "
            "transport-marker, throughput, or health validation. It is invalid for "
            "causal inference; pair 1 remained direct throughout this arm."
        )
    if attention_q8_host_bounce == "underloaded":
        return (
            "The combined pair-0 host-bounce arm completed below the required "
            "500 prefill tok/s production-load floor. It is deliberately classified "
            "as inconclusive-underloaded: survival at substantially reduced work rate "
            "cannot show that either cut pair-0 direct payload family is causal. "
            "Pair-1 traffic remained direct."
            + indexer_observation(attention_q8_host_bounce_rows)
        )
    if attention_q8_host_bounce == "incomplete":
        return (
            "The combined pair-0 host-bounce arm has no verified outcome. Missing "
            "durable measured completion for either transport family is not proof "
            "that both pair-0 attention/Q8 payload routes were removed from the "
            "failing phase. Pair-1 traffic remained direct."
            + indexer_observation(attention_q8_host_bounce_rows)
        )
    if (attention_host_bounce == "passed" and
            attention_host_bounce_pass_verified):
        return (
            "Pair 0 completed the full 32K production-shaped row split after only "
            "its attention-owned query, mirrored-cache, and result copies were staged "
            "through pinned host memory. Any non-split top-k copy uses the same staging."
            + topk_evidence(attention_host_bounce_rows) + " Q8 partner transport, "
            "50/50 attention rows, partner attention arithmetic, and both indexer row "
            "splits remained enabled. This implicates the direct attention P2P/BAR1 "
            "path or its timing envelope, but host staging changes throughput and "
            "instantaneous load and is not by itself a power-matched proof or "
            "production mitigation."
        )
    if attention_host_bounce == "failed":
        if not attention_host_bounce_failure_reached:
            return (
                "A device loss was corroborated, but no durable measured pos=512 "
                "attention host-bounce checkpoint proves that the failed arm reached "
                "the transport cut. This is a real failure but inconclusive for "
                "direct attention P2P/BAR1; inspect the last phase, watch event, and "
                "post-run health before attributing the failure."
            )
        if not attention_host_bounce_pair0_failure_reached:
            return (
                "The failed arm reached the measured forced-host-bounce submission "
                "boundary, but its durable device-loss evidence does not identify "
                "physical GPU 0 or 1, the pair under test. Because pair 1 retained "
                "direct peer traffic, this failure is real but inconclusive for "
                "pair-0 direct attention P2P/BAR1."
            )
        if attention_host_bounce_complete_pair0_failure_reached:
            return (
                "A durable event=complete checkpoint proves that the measured layer "
                "at pos=512 completed end-to-end through the forced attention "
                "host-bounce route before pair-0 device loss."
                + topk_evidence(attention_host_bounce_failed_rows) +
                " Pair-0 attention-owned direct P2P/BAR1 "
                "transfer is therefore not necessary for the observed pair-0 "
                "device loss in this arm. Q8 direct peer traffic, partner attention "
                "execution, pair-1 direct traffic, aggregate load, or a delayed "
                "physical effect remains."
            )
        return (
            "A durable measured pos=512 begin/failed checkpoint proves only that the "
            "arm reached the forced-host-bounce submission boundary; it does not "
            "claim that the pos=512 attention operation completed."
            + topk_evidence(attention_host_bounce_failed_rows) +
            " Pair-0 attention-owned direct P2P/BAR1 transfer is not "
            "necessary for the observed pair-0 device loss because that route was "
            "disabled by configuration, but the completion boundary is unknown. Q8 "
            "direct peer traffic, partner attention execution, pair-1 direct traffic, "
            "aggregate load, or a delayed physical effect remains."
        )
    if attention_host_bounce == "passed":
        return (
            "The attention host-bounce arm was recorded as passed, but no durable "
            "complete pos=512 checkpoint verifies that the measured transport cut "
            "completed. Treat this record as inconclusive rather than as a safe pass "
            "for direct attention P2P/BAR1."
        )
    if attention_host_bounce == "invalid":
        return (
            "The attention host-bounce arm returned but failed its production-path, "
            "transport-marker, or >=500 prefill tok/s validation. It is invalid for "
            "causal inference and is neither a safe pass nor a reproduced device loss."
        )
    if attention_host_bounce == "incomplete":
        return (
            "The attention host-bounce arm has no verified outcome. A missing result, "
            "user interruption, or unverified post-run GPU snapshot is not a device-loss "
            "failure and not a safe pass; repeat this one-arm diagnostic in a fresh "
            "directory after checking its durable watch marker."
        )
    if attention == "passed" or attention_completed_without_result:
        result_note = (
            "The benchmark reached its final decode frontier and emitted results, "
            "although the older wrapper omitted its final result record. "
            if attention_completed_without_result else ""
        )
        return (
            result_note +
            "Pair 0 survived with only its prefill attention row split "
            "disabled while full dense admission, direct partner projections, "
            "mirrored attention-cache updates, pair-0 decode-indexer splitting, and "
            "both pair-1 prefill splits remained. Pair-0 prefill indexer splitting "
            "also stayed home because its split ownership only feeds split attention. "
            "The reproduced failure was already observed at mixed layer 17/position "
            "512, before indexed attention begins. Together, this makes the pair-0 "
            "mixed prefill split-attention ownership/execution path or its interaction "
            "a necessary condition in this pass."
        )
    if attention == "failed":
        return (
            "Pair 0 still failed with its prefill attention row split disabled. "
            "That split is therefore not necessary for the failure; the remaining "
            "common path is full partner projection work/traffic, aggregate load, "
            "power delivery, or the shared PCIe/root-complex path."
        )
    if attention == "invalid":
        return (
            "The attention-off process returned but failed its production-path "
            "validation. It is invalid for causal inference; inspect the missing "
            "marker rather than treating the wrapper failure as a GPU failure."
        )
    ownership = {
        "query-only destination scheduling": query_dst,
        "gather-only destination scheduling": gather_dst,
        "combined destination scheduling": both_dst,
        "source-scheduled production control": production,
    }
    ownership_requested = any(
        state != "not-run" for name, state in ownership.items()
        if name != "source-scheduled production control"
    )
    if ownership_requested and any(
            state == "invalid" for state in ownership.values()):
        return (
            "At least one attention peer-copy scheduling arm returned but failed "
            "its production-path, direct-peer, or >=500 prefill tok/s validation. "
            "The matrix is invalid for causal inference; this is not evidence for "
            "or against that scheduling mode."
        )
    unresolved_ownership = [
        name for name, state in ownership.items()
        if state in {"not-run", "incomplete"}
    ]
    if ownership_requested and unresolved_ownership:
        return (
            "The attention peer-copy scheduling matrix has no verified outcome for "
            + ", ".join(unresolved_ownership)
            + ". A missing result, interrupted wrapper, or unverified post-run GPU "
            "snapshot is not a failed arm and is not a safe pass. Resume the same "
            "directory only to advance arms that never started. An arm already retained "
            "as incomplete is immutable evidence and must be repeated in a fresh full "
            "matrix before making a factorial comparison."
        )
    if ownership_requested and production == "passed":
        return (
            "The source-scheduled production control passed, so this matrix did not "
            "reproduce the known fault under its current boot and workload history. "
            "Candidate pass/fail differences from this matrix cannot establish a "
            "necessary trigger; repeat from a clean boot and preserve the full matrix."
        )
    candidate_ownership = (query_dst, gather_dst, both_dst)
    if (ownership_requested and production == "failed" and
            all(state == "failed" for state in candidate_ownership)):
        return (
            "The query-only, gather-only, combined destination-scheduled arms, and "
            "source-scheduled production control all recorded failures under the full "
            "32K direct-P2P row-split workload. Changing the CUDA submission context, "
            "default stream, event dependencies, and required peer-access direction "
            "was not sufficient to prevent the fault in this matrix. This does not "
            "identify a physical copy engine or clear the attention/cache path."
        )
    if ownership_requested and production == "failed" and both_dst == "passed":
        if query_dst == "failed" and gather_dst == "failed":
            return (
                "Only the combined destination-scheduled arm passed while both "
                "single-copy arms and the source-scheduled production control failed. "
                "The arm preserves transfer directions, byte counts, row splitting, "
                "and direct P2P, but jointly changes CUDA submission context/default "
                "stream, event dependencies, and peer-access direction for both "
                "copies. Confirm the apparent pass and the control from fresh boots; "
                "this matrix does not identify a physical copy engine."
            )
        return (
            "The combined destination-scheduled arm passed while the source-scheduled "
            "production control failed. Direct P2P, transfer directions, byte counts, "
            "row splitting, and partner computation were retained. The changed bundle "
            "is CUDA submission context/default stream, event dependencies, and "
            "peer-access direction; its components are not independently isolated. "
            "Confirm the apparent pass and the failed control from fresh boots."
        )
    if (ownership_requested and production == "failed" and
            query_dst == "passed" and gather_dst == "passed"):
        return (
            "Both single-copy destination-scheduled arms passed while the combined "
            "destination-scheduled arm and source-scheduled production control failed. "
            "That non-additive pattern points to scheduling/history sensitivity rather "
            "than isolating either logical transfer. Confirm each apparent pass, the "
            "combined failure, and the control from fresh boots before attribution."
        )
    if ownership_requested and production == "failed" and query_dst == "passed":
        return (
            "Query-copy destination scheduling passed while the source-scheduled "
            "production control failed; result gather retained production scheduling. "
            "This is a promising query-side scheduling/peer-access axis without "
            "disabling row splitting or direct P2P, not identification of a physical "
            "copy engine. Confirm both the apparent pass and failed control from fresh "
            "boots before attribution."
        )
    if ownership_requested and production == "failed" and gather_dst == "passed":
        return (
            "Only result-gather destination scheduling passed while the source-scheduled "
            "production control failed; query copy retained production scheduling. "
            "This is a promising gather-side scheduling/peer-access axis without "
            "disabling row splitting or direct P2P, not identification of a physical "
            "copy engine. Confirm both the apparent pass and failed control from fresh "
            "boots before attribution."
        )
    if ownership_requested and production == "failed":
        return (
            "The full attention peer-copy scheduling matrix and failed production "
            "control are complete, but the pass/fail pattern does not isolate one copy. "
            "It changes CUDA submission context/default stream, event dependencies, "
            "and peer-access direction without changing transfer direction or bytes. "
            "Repeat any apparent pass and the control from fresh boots before causal "
            "attribution."
        )
    if production == "not-run":
        return (
            "The production control has not run yet. Resume the same directory to "
            "advance to it; earlier failures remain evidence, but causal comparison "
            "requires the final control."
        )
    if production == "incomplete":
        return (
            "The production control has no verified outcome. The retained incomplete "
            "record is not overwritten by resume; repeat the full matrix in a fresh "
            "directory before causal comparison."
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
    full_legacy_matrix = (
        "attention-off", "partner-bounce", "bounce-indexer-off",
        "partner-serialized", "indexer-off", "production",
    )
    if all(outcomes.get(variant) == "failed" for variant in full_legacy_matrix):
        return (
            "Every workload-preserving arm failed. The evidence does not isolate direct "
            "P2P, indexer rows, or overlap; partner computation, aggregate pair load, "
            "power delivery, or the shared PCIe/root-complex path remains."
        )
    return (
        "The completed arms do not yet isolate a necessary trigger. Resume the same "
        "directory only to advance variants that never started; repeat any retained "
        "incomplete arm in a fresh matrix."
    )


def main() -> None:
    if len(sys.argv) != 2:
        fail("usage: summarize-sm75-small-bar1-pair-isolation.py OUTPUT_DIR")
    root = Path(sys.argv[1]).resolve()
    production = root / "production"
    if not production.is_dir():
        fail(f"production directory not found: {production}")
    manifest_path = root / "manifest.txt"
    manifest = read_kv(manifest_path) if manifest_path.exists() else {}
    boundary_end_layer = int(manifest.get("attention_row_boundary_end_layer", "17"))
    boundary_entry_layer = int(
        manifest.get("attention_row_boundary_entry_layer", "18")
    )
    boundary_pos = int(manifest.get("attention_row_boundary_pos", "512"))

    rows: list[dict[str, str]] = []
    statuses: defaultdict[str, list[str]] = defaultdict(list)
    stems = {
        path.name.removesuffix(suffix)
        for suffix in (".result", ".started")
        for path in production.glob(f"*{suffix}")
    }
    for stem in sorted(stems):
        result_path = production / f"{stem}.result"
        started_path = production / f"{stem}.started"
        has_result = result_path.exists()
        values = read_kv(result_path if has_result else started_path)
        variant = values.get("variant", "unknown")
        progress_values = last_progress(production / f"{stem}-progress.csv")
        log_path = production / f"{stem}.log"
        log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
        bench_path = production / f"{stem}.csv"
        bench_row: dict[str, str] = {}
        if bench_path.exists():
            with bench_path.open(newline="") as handle:
                bench_rows = list(csv.DictReader(handle))
            if bench_rows:
                bench_row = bench_rows[-1]
        post_health_path = root / "health" / f"{stem}-post.log"
        post_health = post_health_state(post_health_path)
        post_health_ok = post_health == "healthy"
        watch_path = root / "telemetry" / f"{stem}-watch-event.txt"
        watch_present = watch_path.exists()
        watch_values = read_kv(watch_path) if watch_present else {}
        watch_status = watch_values.get("status", "")
        if has_result:
            status = values.get("status", "unknown")
            if status == "passed":
                if watch_present:
                    status = "passed-invalidated-watch"
                elif not post_health_ok:
                    status = "passed-unverified-health"
            elif status == "interrupted-prior-run" and (
                    watch_status == "lost-device-detected" or
                    post_health == "unhealthy"):
                status = "interrupted-prior-run-device-loss"
            elif status == "failed":
                status = (
                    "failed-device-loss"
                    if (watch_status == "lost-device-detected" or
                        post_health == "unhealthy")
                    else "failed-unverified"
                )
        elif (watch_status == "lost-device-detected" or
              post_health == "unhealthy"):
            status = "interrupted-no-result-device-loss"
        elif (bench_row and progress_values.get("phase") == "decode" and
              progress_values.get("event") == "frontier-complete" and
              progress_values.get("current") == progress_values.get("total") and
              post_health_ok and not watch_present):
            status = "completed-no-result"
        elif (bench_row and progress_values.get("phase") == "decode" and
              progress_values.get("event") == "frontier-complete" and
              progress_values.get("current") == progress_values.get("total")):
            status = (
                "completed-no-result-invalidated-watch" if watch_present else
                "completed-no-result-unverified-health"
            )
        else:
            status = "interrupted-no-result"
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
        query_schedule, gather_schedule = attention_copy_schedules(log_text, 0)
        query_transport, gather_transport = attention_copy_transports(log_text, 0)
        cache_transport, topk_transport = attention_aux_transports(log_text, 0)
        q8_phase_last, q8_phase_first_failure, q8_phase_classification = (
            q8_partner_phase_audit_summary(log_text, 0)
        )
        (q8_async_last, q8_async_last_failure, q8_async_classification,
         q8_async_begun, q8_async_submitted, q8_async_confirmed,
         q8_async_synchronized) = q8_partner_async_completion_summary(log_text)
        (q8_fence_last, q8_fence_last_failure, q8_fence_last_armed,
         q8_fence_last_returned, q8_fence_classification,
         q8_fence_attempted, q8_fence_confirmed, q8_fence_failed,
         q8_fence_gather_failed, q8_fence_armed_count,
         q8_fence_returned_count) = q8_partner_pre_gather_fence_summary(
             log_text, status
         )
        (q8_activation_last, q8_activation_last_failure,
         q8_activation_last_armed, q8_activation_last_returned,
         q8_activation_classification, q8_activation_attempted,
         q8_activation_confirmed, q8_activation_returned,
         q8_activation_failed, q8_activation_armed_count,
         q8_activation_returned_count) = q8_partner_pre_activation_fence_summary(
             log_text, status
         )
        (q8_compute_last_armed, q8_compute_last_returned,
         q8_compute_last_failure, q8_compute_classification,
         q8_compute_counts) = q8_partner_compute_fence_summary(log_text)
        (q8_direct_last_armed, q8_direct_last_returned,
         q8_direct_last_failure, q8_direct_classification,
         q8_direct_counts) = q8_partner_direct_gather_summary(log_text)
        if variant == "attention-q8-l14-l15-phase-audit":
            (q8_window_last_complete, q8_window_l14_complete,
             q8_window_l15_complete, q8_window_classification) = (
                q8_partner_phase_audit_window_summary(log_text, 0)
            )
        else:
            (q8_window_last_complete, q8_window_l14_complete,
             q8_window_l15_complete, q8_window_classification) = ("", "", "", "")
        row = {
            "variant": variant,
            "status": status,
            "exit_status": values.get("exit_status", ""),
            "last_phase": values.get("last_phase", progress_values.get("phase", "")),
            "last_event": values.get("last_event", progress_values.get("event", "")),
            "last_current": values.get("last_current", progress_values.get("current", "")),
            "last_total": values.get("last_total", progress_values.get("total", "")),
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
            "q8_async_completion_last_checkpoint": q8_async_last,
            "q8_async_completion_last_failure": q8_async_last_failure,
            "q8_async_completion_classification": q8_async_classification,
            "q8_async_completion_begun": q8_async_begun,
            "q8_async_completion_submitted": q8_async_submitted,
            "q8_async_completion_confirmed": q8_async_confirmed,
            "q8_async_completion_partners_synchronized": q8_async_synchronized,
            "q8_pre_gather_fence_last_checkpoint": q8_fence_last,
            "q8_pre_gather_fence_last_failure": q8_fence_last_failure,
            "q8_pre_gather_fence_last_armed": q8_fence_last_armed,
            "q8_pre_gather_fence_armed_count": q8_fence_armed_count,
            "q8_pre_gather_fence_last_returned": q8_fence_last_returned,
            "q8_pre_gather_fence_returned_count": q8_fence_returned_count,
            "q8_pre_gather_fence_classification": q8_fence_classification,
            "q8_pre_gather_fence_attempted": q8_fence_attempted,
            "q8_pre_gather_fence_confirmed": q8_fence_confirmed,
            "q8_pre_gather_fence_failed": q8_fence_failed,
            "q8_pre_gather_fence_result_gather_failed_after_confirmed": (
                q8_fence_gather_failed
            ),
            "q8_pre_activation_fence_last_checkpoint": q8_activation_last,
            "q8_pre_activation_fence_last_failure": q8_activation_last_failure,
            "q8_pre_activation_fence_last_armed": q8_activation_last_armed,
            "q8_pre_activation_fence_last_returned": q8_activation_last_returned,
            "q8_pre_activation_fence_classification": (
                q8_activation_classification
            ),
            "q8_pre_activation_fence_attempted": q8_activation_attempted,
            "q8_pre_activation_fence_confirmed": q8_activation_confirmed,
            "q8_pre_activation_fence_returned": q8_activation_returned,
            "q8_pre_activation_fence_failed": q8_activation_failed,
            "q8_pre_activation_fence_armed_count": q8_activation_armed_count,
            "q8_pre_activation_fence_returned_count": (
                q8_activation_returned_count
            ),
            "q8_compute_fence_last_armed": q8_compute_last_armed,
            "q8_compute_fence_last_returned": q8_compute_last_returned,
            "q8_compute_fence_last_failure": q8_compute_last_failure,
            "q8_compute_fence_classification": q8_compute_classification,
            "q8_compute_fence_armed_returned_count": q8_compute_counts,
            "q8_direct_gather_last_armed": q8_direct_last_armed,
            "q8_direct_gather_last_returned": q8_direct_last_returned,
            "q8_direct_gather_last_failure": q8_direct_last_failure,
            "q8_direct_gather_classification": q8_direct_classification,
            "q8_direct_gather_armed_returned_failure_count": q8_direct_counts,
            "q8_phase_audit_last_checkpoint": q8_phase_last,
            "q8_phase_audit_first_failure": q8_phase_first_failure,
            "q8_phase_audit_classification": q8_phase_classification,
            "q8_phase_audit_skipped_occurrence": (
                q8_partner_phase_audit_occurrence_summary(log_text, "skipped")
            ),
            "q8_phase_audit_selected_occurrence": (
                q8_partner_phase_audit_occurrence_summary(log_text, "selected")
            ),
            "q8_phase_audit_occurrence_mapping": (
                q8_l12_occurrence_mapping_state(log_text)
                if variant == "attention-q8-l12-phase-audit" else ""
            ),
            "q8_window_last_complete": q8_window_last_complete,
            "q8_window_l14_complete": q8_window_l14_complete,
            "q8_window_l15_complete": q8_window_l15_complete,
            "q8_window_classification": q8_window_classification,
            "pair0_attention_query_copy_schedule": query_schedule,
            "pair0_attention_gather_copy_schedule": gather_schedule,
            "pair0_attention_query_copy_transport": query_transport,
            "pair0_attention_gather_copy_transport": gather_transport,
            "pair0_attention_cache_copy_transport": cache_transport,
            "pair0_attention_cache_host_bounce_classes": (
                attention_host_bounce_cache_classes(log_text, 0)
            ),
            "pair0_attention_topk_copy_transport": topk_transport,
            "attention_host_bounce_checkpoint": (
                last_attention_host_bounce_checkpoint(log_text, 0)
            ),
            "host_bounce_failure_context": (
                first_host_bounce_failure_context(log_text)
            ),
            "first_gpu_layer_failure": first_gpu_layer_failure(log_text),
            "attention_phase_audit_last": last_attention_phase_audit(log_text),
            "attention_end_fence_last": last_attention_end_fence(log_text),
            "attention_entry_fence_last": last_attention_entry_fence(log_text),
            "attention_audit_first_failure": first_attention_audit_failure(log_text),
            "attention_row_boundary_marker_state": (
                attention_row_boundary_marker_state(
                    log_text, boundary_end_layer, boundary_entry_layer, boundary_pos
                )
                if variant == "attention-row-boundary-audit" else ""
            ),
            "post_health": post_health,
            "watch_status": watch_status,
            "lost_devices": watch_values.get("lost_devices", ""),
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
        "prefill attention row execution/copy scheduling/transport, and pair-0 "
        "prefill/decode-indexer rows. Destination-scheduled attention copies retain "
        "the same transfer direction and byte count; they change the initiating CUDA "
        "context/default stream, event dependencies, and required peer-access "
        "direction. CUDA chooses the physical transfer engine, which this audit does "
        "not identify. The async-completion arm adds a mapped-host marker and a "
        "dedicated event after every selected partner-Q8 compute submission. A "
        "matching marker is positive boundary evidence; a missing marker after "
        "device/context loss is explicitly inconclusive. The pre-gather-fence arm "
        "synchronizes that event and validates the exact marker immediately before "
        "result D2H submission. Its success checkpoints are periodic, so a missing "
        "per-call checkpoint is not negative evidence. A compact, fflush-only "
        "armed record positively confirms the pre-gather event/marker boundary on "
        "every call, and its matching returned record proves the synchronous copy "
        "helper returned successfully. A missing returned record means only that its "
        "breadcrumb was not observed before termination; it is not proof that a "
        "CUDA copy API failed to return.",
        "The pre-activation-fence arm additionally stages activation through host "
        "memory, synchronizes the destination device, and confirms a high-bit-tagged "
        "marker on its default stream before synchronous activation H2D. Armed/"
        "returned records carry the complete binding identity. These records identify "
        "where CUDA first surfaced an error; they do not establish what caused an "
        "endpoint reset. The shared marker slot is later reused, so its teardown value "
        "is not required to equal the last activation tag.",
        "",
        "| Variant | Outcome | Prefill tok/s | Decode tok/s | Last phase | Last event | "
        "Q8 transport | Serialized | Q8 async classification | Q8 async counts | "
        "Q8 async last checkpoint | Q8 async last failure | "
        "Q8 async partners synchronized | Q8 pre-gather classification | "
        "Q8 pre-gather counts | Q8 pre-gather last checkpoint | "
        "Q8 pre-gather armed/returned count | Q8 pre-gather last armed | "
        "Q8 pre-gather last returned | "
        "Q8 pre-gather last failure | Q8 pre-activation classification | "
        "Q8 pre-activation counts | Q8 pre-activation last checkpoint | "
        "Q8 pre-activation armed/returned count | Q8 pre-activation last armed | "
        "Q8 pre-activation last returned | Q8 pre-activation last failure | "
        "Q8 direct-gather classification | Q8 direct-gather counts | "
        "Q8 direct-gather last boundary | "
        "Pair-0 Q8 begun bytes* | "
        "Q8 phase last checkpoint | Q8 phase first failure | "
        "Q8 phase classification | Q8 skipped occurrence | "
        "Q8 selected occurrence | Q8 occurrence mapping | "
        "Q8 window last complete | "
        "Q8 window L14 complete | Q8 window L15 complete | "
        "Q8 window classification | "
        "First GPU-layer failure | "
        "Pair-0 indexer begun bytes | Query copy schedule | Gather copy schedule | "
        "Query copy transport | Gather copy transport | Cache copy transport | "
        "Top-k copy transport | Host-bounce checkpoint | Host-bounce failure context | "
        "Attention phase checkpoint | "
        "Attention end fence | Attention entry fence | First attention-audit failure | "
        "Boundary marker sequence | Post health | Watch event | Lost devices |",
        "| " + " | ".join(["---"] * 59) + " |",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['status']} | {row.get('prefill_tps', '')} | "
            f"{row.get('decode_tps', '')} | {row['last_phase']} | {row['last_event']} | "
            f"{row.get('pair0_q8_transport', '')} | "
            f"{row.get('pair0_q8_serialized', '')} | "
            f"{row.get('q8_async_completion_classification', '')} | "
            f"{row.get('q8_async_completion_begun', '')}/"
            f"{row.get('q8_async_completion_submitted', '')}/"
            f"{row.get('q8_async_completion_confirmed', '')} | "
            f"{row.get('q8_async_completion_last_checkpoint', '')} | "
            f"{row.get('q8_async_completion_last_failure', '')} | "
            f"{row.get('q8_async_completion_partners_synchronized', '')} | "
            f"{row.get('q8_pre_gather_fence_classification', '')} | "
            f"{row.get('q8_pre_gather_fence_attempted', '')}/"
            f"{row.get('q8_pre_gather_fence_confirmed', '')}/"
            f"{row.get('q8_pre_gather_fence_failed', '')}/"
            f"{row.get('q8_pre_gather_fence_result_gather_failed_after_confirmed', '')} | "
            f"{row.get('q8_pre_gather_fence_last_checkpoint', '')} | "
            f"{row.get('q8_pre_gather_fence_armed_count', '')}/"
            f"{row.get('q8_pre_gather_fence_returned_count', '')} | "
            f"{row.get('q8_pre_gather_fence_last_armed', '')} | "
            f"{row.get('q8_pre_gather_fence_last_returned', '')} | "
            f"{row.get('q8_pre_gather_fence_last_failure', '')} | "
            f"{row.get('q8_pre_activation_fence_classification', '')} | "
            f"{row.get('q8_pre_activation_fence_attempted', '')}/"
            f"{row.get('q8_pre_activation_fence_confirmed', '')}/"
            f"{row.get('q8_pre_activation_fence_returned', '')}/"
            f"{row.get('q8_pre_activation_fence_failed', '')} | "
            f"{row.get('q8_pre_activation_fence_last_checkpoint', '')} | "
            f"{row.get('q8_pre_activation_fence_armed_count', '')}/"
            f"{row.get('q8_pre_activation_fence_returned_count', '')} | "
            f"{row.get('q8_pre_activation_fence_last_armed', '')} | "
            f"{row.get('q8_pre_activation_fence_last_returned', '')} | "
            f"{row.get('q8_pre_activation_fence_last_failure', '')} | "
            f"{row.get('q8_direct_gather_classification', '')} | "
            f"{row.get('q8_direct_gather_armed_returned_failure_count', '')} | "
            f"{row.get('q8_direct_gather_last_failure') or row.get('q8_direct_gather_last_returned') or row.get('q8_direct_gather_last_armed', '')} | "
            f"{row.get('pair0_q8_begin_checkpoint_bytes', '0')} | "
            f"{row.get('q8_phase_audit_last_checkpoint', '')} | "
            f"{row.get('q8_phase_audit_first_failure', '')} | "
            f"{row.get('q8_phase_audit_classification', '')} | "
            f"{row.get('q8_phase_audit_skipped_occurrence', '')} | "
            f"{row.get('q8_phase_audit_selected_occurrence', '')} | "
            f"{row.get('q8_phase_audit_occurrence_mapping', '')} | "
            f"{row.get('q8_window_last_complete', '')} | "
            f"{row.get('q8_window_l14_complete', '0')} | "
            f"{row.get('q8_window_l15_complete', '0')} | "
            f"{row.get('q8_window_classification', '')} | "
            f"{row.get('first_gpu_layer_failure', '')} | "
            f"{row.get('pair0_indexer_begin_bytes', '0')} | "
            f"{row.get('pair0_attention_query_copy_schedule', '')} | "
            f"{row.get('pair0_attention_gather_copy_schedule', '')} | "
            f"{row.get('pair0_attention_query_copy_transport', '')} | "
            f"{row.get('pair0_attention_gather_copy_transport', '')} | "
            f"{row.get('pair0_attention_cache_copy_transport', '')} | "
            f"{row.get('pair0_attention_topk_copy_transport', '')} | "
            f"{row.get('attention_host_bounce_checkpoint', '')} | "
            f"{row.get('host_bounce_failure_context', '')} | "
            f"{row.get('attention_phase_audit_last', '')} | "
            f"{row.get('attention_end_fence_last', '')} | "
            f"{row.get('attention_entry_fence_last', '')} | "
            f"{row.get('attention_audit_first_failure', '')} | "
            f"{row.get('attention_row_boundary_marker_state', '')} | "
            f"{row.get('post_health', '')} | "
            f"{row.get('watch_status', '')} | "
            f"{row.get('lost_devices', '')} |"
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
    if (len(sys.argv) == 3 and
            sys.argv[1] == "--validate-q8-async-completion-log"):
        log_path = Path(sys.argv[2])
        state = q8_partner_async_completion_marker_state(
            log_path.read_text(errors="replace")
        )
        if state != "complete":
            fail(f"Q8 async-completion marker state is {state}")
    elif (len(sys.argv) == 3 and
            sys.argv[1] == "--validate-q8-pre-gather-fence-log"):
        log_path = Path(sys.argv[2])
        state = q8_partner_pre_gather_fence_marker_state(
            log_path.read_text(errors="replace")
        )
        if state != "complete":
            fail(f"Q8 pre-gather-fence marker state is {state}")
    elif (len(sys.argv) == 3 and
            sys.argv[1] == "--validate-q8-pre-activation-fence-log"):
        log_path = Path(sys.argv[2])
        state = q8_partner_pre_activation_fence_marker_state(
            log_path.read_text(errors="replace")
        )
        if state != "complete":
            fail(f"Q8 pre-activation-fence marker state is {state}")
    elif (len(sys.argv) == 3 and
            sys.argv[1] == "--validate-q8-compute-fence-log"):
        log_path = Path(sys.argv[2])
        state = q8_partner_compute_fence_marker_state(
            log_path.read_text(errors="replace")
        )
        if state != "complete":
            fail(f"Q8 compute-fence marker state is {state}")
    elif (len(sys.argv) == 3 and
            sys.argv[1] == "--validate-q8-direct-gather-log"):
        log_path = Path(sys.argv[2])
        state = q8_partner_direct_gather_marker_state(
            log_path.read_text(errors="replace")
        )
        if state != "complete":
            fail(f"Q8 direct-gather marker state is {state}")
    elif (len(sys.argv) == 4 and
            sys.argv[1] == "--validate-q8-l14-l15-log"):
        log_path = Path(sys.argv[2])
        state = q8_partner_phase_audit_window_marker_state(
            log_path.read_text(errors="replace"),
            int(sys.argv[3]),
        )
        if state != "complete":
            fail(f"Q8 layer-14/layer-15 marker sequence is {state}")
    elif (len(sys.argv) == 6 and
            sys.argv[1] == "--validate-attention-row-boundary-log"):
        log_path = Path(sys.argv[2])
        state = attention_row_boundary_marker_state(
            log_path.read_text(errors="replace"),
            int(sys.argv[3]),
            int(sys.argv[4]),
            int(sys.argv[5]),
        )
        if state != "complete":
            fail(f"attention row-boundary marker sequence is {state}")
    else:
        main()
