#!/usr/bin/env python3

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path


VARIANT_ORDER = (
    "attention-off", "attention-phase-audit", "attention-end-fence",
    "attention-row-boundary-audit", "partner-bounce", "bounce-indexer-off",
    "partner-serialized", "indexer-off", "production"
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
    if any(status in {"failed", "interrupted-prior-run"} for status in statuses):
        return "failed"
    if any(status == "validation-failed" for status in statuses):
        return "invalid"
    return "incomplete"


def last_progress(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else {}


def healthy_post_snapshot(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(errors="replace")
    return (len(re.findall(r"^GPU \d+:", text, re.MULTILINE)) == 4 and
            not re.search(
                r"ERR!|GPU is lost|Unknown Error|Unable to determine",
                text,
                re.IGNORECASE,
            ))


def inference(outcomes: dict[str, str], rows: list[dict[str, str]]) -> str:
    if any("power-limit-drift" in row.get("watch_status", "") for row in rows):
        return (
            "At least one arm changed power limit after its 250 W preflight. That arm "
            "is invalid for causal comparison; identify the external power-limit writer "
            "and rerun it."
        )
    attention = outcomes.get("attention-off", "not-run")
    phase_audit = outcomes.get("attention-phase-audit", "not-run")
    end_fence = outcomes.get("attention-end-fence", "not-run")
    boundary_audit = outcomes.get("attention-row-boundary-audit", "not-run")
    production = outcomes.get("production", "not-run")
    bounce = outcomes.get("partner-bounce", "not-run")
    bounce_indexer = outcomes.get("bounce-indexer-off", "not-run")
    serialized = outcomes.get("partner-serialized", "not-run")
    indexer = outcomes.get("indexer-off", "not-run")
    attention_rows = [row for row in rows if row.get("variant") == "attention-off"]
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
        "directory until all requested variants have durable results."
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
        post_health_ok = healthy_post_snapshot(
            root / "health" / f"{stem}-post.log"
        )
        if has_result:
            status = values.get("status", "unknown")
            if status == "passed" and not post_health_ok:
                status = "passed-unverified-health"
        elif (bench_row and progress_values.get("phase") == "decode" and
              progress_values.get("event") == "frontier-complete" and
              progress_values.get("current") == progress_values.get("total") and
              post_health_ok):
            status = "completed-no-result"
        elif (bench_row and progress_values.get("phase") == "decode" and
              progress_values.get("event") == "frontier-complete" and
              progress_values.get("current") == progress_values.get("total")):
            status = "completed-no-result-unverified-health"
        else:
            status = "interrupted-no-result"
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
            "post_health": "healthy" if post_health_ok else "unverified",
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
        "Pair-0 indexer begun bytes | Attention phase checkpoint | "
        "Attention end fence | Attention entry fence | First attention-audit failure | "
        "Boundary marker sequence | Post health | Watch event |",
        "| --- | --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['status']} | {row.get('prefill_tps', '')} | "
            f"{row.get('decode_tps', '')} | {row['last_phase']} | {row['last_event']} | "
            f"{row.get('pair0_q8_transport', '')} | "
            f"{row.get('pair0_q8_serialized', '')} | "
            f"{row.get('pair0_q8_begin_checkpoint_bytes', '0')} | "
            f"{row.get('pair0_indexer_begin_bytes', '0')} | "
            f"{row.get('attention_phase_audit_last', '')} | "
            f"{row.get('attention_end_fence_last', '')} | "
            f"{row.get('attention_entry_fence_last', '')} | "
            f"{row.get('attention_audit_first_failure', '')} | "
            f"{row.get('attention_row_boundary_marker_state', '')} | "
            f"{row.get('post_health', '')} | "
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
    if (len(sys.argv) == 6 and
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
