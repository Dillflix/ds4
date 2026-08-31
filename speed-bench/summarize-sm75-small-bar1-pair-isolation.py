#!/usr/bin/env python3

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path


VARIANT_ORDER = (
    "attention-off", "attention-host-bounce", "attention-q8-host-bounce",
    "attention-q8-phase-audit", "attention-q8-targeted-phase-audit",
    "attention-q8-l14-l15-phase-audit",
    "attention-query-dst", "attention-gather-dst",
    "attention-both-dst", "attention-phase-audit", "attention-end-fence",
    "attention-row-boundary-audit", "partner-bounce", "bounce-indexer-off",
    "partner-serialized", "indexer-off", "production"
)

Q8_WINDOW_L14_LABEL = "tensor:blk.14.attn_output_b.weight"
Q8_WINDOW_L14_OFFSET = "143571266304"
Q8_WINDOW_L15_LABEL = "tensor:blk.15.attn_output_b.weight"
Q8_WINDOW_L15_OFFSET = "143723876608"
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
    for index, line in enumerate(lines):
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
    failure_events = {
        "activation-copy-failed", "pre-compute-sync-failed",
        "compute-submit-failed", "compute-sync-failed",
        "result-copy-failed", "home-restore-failed",
    }
    complete_markers = [
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
    last_complete = (
        format_q8_partner_phase_audit_marker(complete_markers[-1])
        if complete_markers else ""
    )
    first_failure_index = next((
        index for index, marker in enumerate(exact_markers)
        if marker["event"] in failure_events
    ), None)
    first_failure = (
        exact_markers[first_failure_index]
        if first_failure_index is not None else {}
    )
    classification = "inconclusive"
    if first_failure:
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
            prior_markers = exact_markers[:first_failure_index]
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
    elif exact_markers and exact_markers[-1]["event"] != "result-complete":
        if exact_markers[-1]["binding_label"] == Q8_WINDOW_L14_LABEL:
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
        str(len(complete_sequences[Q8_WINDOW_L14_LABEL])),
        str(len(complete_sequences[Q8_WINDOW_L15_LABEL])),
        classification,
    )


def format_q8_partner_phase_audit_marker(marker: dict[str, str]) -> str:
    if not marker:
        return ""
    cuda_phase = (
        f" cuda_phase={marker['cuda_phase']}" if marker.get("cuda_phase") else ""
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
        f"{cuda_phase} cuda_error={marker['cuda_error']}"
    )


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
    attention = outcomes.get("attention-off", "not-run")
    attention_host_bounce = outcomes.get("attention-host-bounce", "not-run")
    attention_q8_host_bounce = outcomes.get(
        "attention-q8-host-bounce", "not-run"
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
            "q8_phase_audit_last_checkpoint": q8_phase_last,
            "q8_phase_audit_first_failure": q8_phase_first_failure,
            "q8_phase_audit_classification": q8_phase_classification,
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
        "not identify.",
        "",
        "| Variant | Outcome | Prefill tok/s | Decode tok/s | Last phase | Last event | "
        "Q8 transport | Serialized | Pair-0 Q8 begun bytes* | "
        "Q8 phase last checkpoint | Q8 phase first failure | "
        "Q8 phase classification | Q8 window last complete | "
        "Q8 window L14 complete | Q8 window L15 complete | "
        "Q8 window classification | "
        "Pair-0 indexer begun bytes | Query copy schedule | Gather copy schedule | "
        "Query copy transport | Gather copy transport | Cache copy transport | "
        "Top-k copy transport | Host-bounce checkpoint | Host-bounce failure context | "
        "Attention phase checkpoint | "
        "Attention end fence | Attention entry fence | First attention-audit failure | "
        "Boundary marker sequence | Post health | Watch event | Lost devices |",
        "| --- | --- | ---: | ---: | --- | --- | --- | --- | ---: | --- | --- | --- | --- | ---: | ---: | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['status']} | {row.get('prefill_tps', '')} | "
            f"{row.get('decode_tps', '')} | {row['last_phase']} | {row['last_event']} | "
            f"{row.get('pair0_q8_transport', '')} | "
            f"{row.get('pair0_q8_serialized', '')} | "
            f"{row.get('pair0_q8_begin_checkpoint_bytes', '0')} | "
            f"{row.get('q8_phase_audit_last_checkpoint', '')} | "
            f"{row.get('q8_phase_audit_first_failure', '')} | "
            f"{row.get('q8_phase_audit_classification', '')} | "
            f"{row.get('q8_window_last_complete', '')} | "
            f"{row.get('q8_window_l14_complete', '0')} | "
            f"{row.get('q8_window_l15_complete', '0')} | "
            f"{row.get('q8_window_classification', '')} | "
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
    if (len(sys.argv) == 4 and
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
