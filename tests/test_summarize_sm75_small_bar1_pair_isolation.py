#!/usr/bin/env python3

import csv
import pathlib
import subprocess
import sys
import tempfile
import unittest


REPO = pathlib.Path(__file__).resolve().parents[1]
SUMMARIZER = REPO / "speed-bench/summarize-sm75-small-bar1-pair-isolation.py"


def write_healthy_post(root: pathlib.Path, stem: str) -> None:
    health = root / "health"
    health.mkdir(exist_ok=True)
    (health / f"{stem}-post.log").write_text(
        "GPU 0: ok\nGPU 1: ok\nGPU 2: ok\nGPU 3: ok\n"
    )


def write_unhealthy_post(root: pathlib.Path, stem: str, device: int = 1) -> None:
    health = root / "health"
    health.mkdir(exist_ok=True)
    (health / f"{stem}-post.log").write_text(
        "".join(
            f"GPU {index}: {'GPU is lost' if index == device else 'ok'}\n"
            for index in range(4)
        )
    )


def write_lost_watch(root: pathlib.Path, stem: str, lost_devices: str) -> None:
    telemetry = root / "telemetry"
    telemetry.mkdir(exist_ok=True)
    (telemetry / f"{stem}-watch-event.txt").write_text(
        "status=lost-device-detected\n"
        f"lost_devices={lost_devices}\n"
    )


def complete_row_boundary_log() -> str:
    end_identity = "kind=mixed layer=17 pos=512 tokens=512 home=0 partner=2"
    entry_identity = "kind=mixed layer=18 pos=512 tokens=512 home=0 partner=2"
    lines = [
        f"ds4: CUDA prefill attention row end fence event=begin target=pair {end_identity}",
        f"ds4: CUDA prefill attention row end fence event=complete target=partner {end_identity}",
        f"ds4: CUDA prefill attention row end fence event=complete target=home {end_identity}",
        f"ds4: CUDA prefill attention row end fence event=complete target=pair {end_identity}",
        f"ds4: CUDA prefill attention row audit dispatch=split {end_identity} q_bytes=1 result_bytes=1",
        f"ds4: CUDA prefill attention row entry fence event=begin target=pair {entry_identity}",
        f"ds4: CUDA prefill attention row entry fence event=complete target=partner {entry_identity}",
        f"ds4: CUDA prefill attention row entry fence event=complete target=home {entry_identity}",
        f"ds4: CUDA prefill attention row entry fence event=complete target=pair {entry_identity}",
    ]
    for phase in (
            "query-copy", "partner-attention", "home-attention", "result-gather"):
        lines.extend([
            f"ds4: CUDA prefill attention row phase audit event=begin phase={phase} "
            f"{entry_identity}",
            f"ds4: CUDA prefill attention row phase audit event=complete phase={phase} "
            f"{entry_identity}",
        ])
    lines.append(
        f"ds4: CUDA prefill attention row audit dispatch=split {entry_identity} "
        "q_bytes=1 result_bytes=1"
    )
    return "\n".join(lines) + "\n"


def q8_phase_marker(
        sequence: int, event: str, stage: str, *, home_tier: int = 0,
        home_device: int = 0, partner_tier: int = 2,
        partner_device: int = 1, cuda_error: str = "none",
        binding_label: str = "tensor:blk.14.attn_output_b.weight",
        weight_offset: int = 143571266304) -> str:
    return (
        "ds4: CUDA q8 partner phase audit "
        f"sequence={sequence} event={event} stage={stage} "
        f"binding_label={binding_label} "
        f"passed_label=attn_output_b weight_offset={weight_offset} "
        f"home_tier={home_tier} home_device={home_device} "
        f"partner_tier={partner_tier} partner_device={partner_device} "
        "tokens=512 in=8192 out=4096 transfer_bytes=8388608 "
        f"result_bytes=8388608 cuda_error={cuda_error}\n"
    )


def q8_phase_chain(
        sequence: int, *,
        binding_label: str = "tensor:blk.14.attn_output_b.weight",
        weight_offset: int = 143571266304) -> str:
    return "".join(
        q8_phase_marker(
            sequence, event, stage,
            binding_label=binding_label,
            weight_offset=weight_offset,
        )
        for event, stage in (
            ("begin", "activation-prepare"),
            ("activation-complete", "activation-copy"),
            ("pre-compute-sync-begin", "pre-compute-sync"),
            ("pre-compute-complete", "pre-compute-sync"),
            ("compute-submitted", "compute"),
            ("compute-complete", "compute-sync"),
            ("result-complete", "result-gather"),
        )
    )


def q8_l12_measured_selection_prefix() -> str:
    binding = "tensor:blk.12.attn_output_b.weight"
    offset = 143236281600
    return (
        "ds4-bench: starting untimed CUDA warm-up frontier 512\n"
        "ds4: CUDA q8 partner phase audit skipped occurrence=1 "
        f"binding_label={binding} weight_offset={offset}\n"
        "ds4-bench: completed untimed CUDA warm-up frontier 512\n"
        "ds4: prefill fault breadcrumb event=chunk-begin "
        "range_start=0 range_tokens=32768 chunk_start=0 chunk_tokens=2048\n"
        "ds4: CUDA q8 partner phase audit selected occurrence=2 sequence=1 "
        f"binding_label={binding} weight_offset={offset}\n"
    )


def q8_async_enabled() -> str:
    return (
        "ds4: CUDA q8 partner async completion audit enabled: "
        "logical_pairs=0 marker=partner-default-stream-mapped-host "
        "event=dedicated-post-marker interpretation=positive-only\n"
    )


def q8_async_checkpoint(sequence: int) -> str:
    complement = (~sequence) & ((1 << 64) - 1)
    return (
        "ds4: CUDA q8 partner async completion checkpoint "
        "home_tier=0 partner_tier=2 "
        f"begun={sequence} submitted={sequence} confirmed={sequence} "
        f"sequence={sequence} complement={complement} "
        "evidence=post-compute-confirmed\n"
    )


def q8_async_summary(
        begun: int, submitted: int, confirmed: int, *, synchronized: str = "yes",
        sequence: int | None = None, complement: int | None = None) -> str:
    sequence = begun if sequence is None else sequence
    complement = (
        ((~sequence) & ((1 << 64) - 1))
        if complement is None else complement
    )
    return (
        "ds4: CUDA q8 partner async completion summary "
        "home_tier=0 partner_tier=2 "
        f"begun={begun} submitted={submitted} confirmed={confirmed} "
        f"last_sequence={sequence} last_complement={complement} "
        f"partners_synchronized={synchronized}\n"
    )


def q8_async_failure(
        sequence: int, *, marker_matches: str, event_status: str,
        interpretation: str) -> str:
    marker_sequence = sequence if marker_matches == "yes" else sequence - 1
    marker_complement = (~marker_sequence) & ((1 << 64) - 1)
    return (
        "ds4: CUDA q8 partner async completion failure "
        f"stage=result-gather current_sequence={sequence} "
        f"marker_sequence={marker_sequence} marker_complement={marker_complement} "
        f"marker_matches={marker_matches} event_status={event_status} "
        f"interpretation={interpretation} begun={sequence} "
        f"submitted={sequence} confirmed={sequence - 1} "
        "home_tier=0 home_device=0 partner_tier=2 partner_device=1 "
        "tokens=512 in=8192 out=4096 "
        "binding_label=tensor:blk.0.attn_output_b.weight "
        "passed_label=attn_output_b weight_offset=141234898176\n"
    )


def q8_pre_gather_enabled() -> str:
    return (
        q8_async_enabled() +
        "ds4: CUDA q8 partner pre-gather fence audit enabled: "
        "logical_pairs=0 boundary=post-marker-event-sync "
        "marker=exact-before-result-d2h\n"
    )


def q8_pre_gather_checkpoint(sequence: int) -> str:
    complement = (~sequence) & ((1 << 64) - 1)
    return (
        "ds4: CUDA q8 partner pre-gather fence checkpoint "
        "event=complete stage=pre-result-d2h "
        f"current_sequence={sequence} marker_sequence={sequence} "
        f"marker_complement={complement} marker_matches=yes "
        "event_status=complete result_d2h_attempted=no result_d2h_completed=no "
        "result_h2d_attempted=no result_h2d_completed=no "
        "interpretation=post-compute-confirmed-before-result-d2h "
        f"attempted={sequence} confirmed={sequence} failed=0 "
        "result_gather_failed_after_confirmed=0 "
        "home_tier=0 home_device=0 partner_tier=2 partner_device=1 "
        "tokens=512 in=8192 out=4096 "
        "binding_label=tensor:blk.0.attn_output_b.weight "
        "passed_label=attn_output_b weight_offset=141234898176\n"
    )


def q8_pre_gather_armed(sequence: int) -> str:
    complement = (~sequence) & ((1 << 64) - 1)
    return (
        "ds4: CUDA q8 partner pre-gather armed "
        f"current_sequence={sequence} marker_sequence={sequence} "
        f"marker_complement={complement} home_tier=0 home_device=0 "
        "partner_tier=2 partner_device=1\n"
    )


def q8_pre_gather_returned(sequence: int) -> str:
    return (
        "ds4: CUDA q8 partner pre-gather returned "
        f"current_sequence={sequence} result_gather_status=success "
        "home_tier=0 home_device=0 partner_tier=2 partner_device=1\n"
    )


def q8_pre_gather_completed_calls(count: int) -> str:
    return "".join(
        q8_pre_gather_armed(sequence) + q8_pre_gather_returned(sequence)
        for sequence in range(1, count + 1)
    )


def q8_pre_gather_checkpointed_calls(count: int) -> str:
    lines = []
    for sequence in range(1, count + 1):
        if sequence == 1 or sequence % 64 == 0:
            lines.append(q8_pre_gather_checkpoint(sequence))
        lines.append(q8_pre_gather_armed(sequence))
        lines.append(q8_pre_gather_returned(sequence))
    return "".join(lines)


def q8_pre_gather_failure(
        sequence: int, event: str, *, d2h_attempted_override: str | None = None,
        d2h_completed_override: str | None = None,
        h2d_attempted_override: str | None = None,
        h2d_completed_override: str | None = None,
        attempted_override: int | None = None
) -> str:
    if event == "state-invalid":
        attempted = attempted_override or sequence
        marker_sequence = sequence - 1
        event_status = "not-synchronized"
        d2h_attempted = "no"
        d2h_completed = "no"
        h2d_attempted = "no"
        h2d_completed = "no"
        interpretation = "failure-surfaced-before-result-d2h"
        confirmed = attempted - 1
        failed = 1
        gather_failed = 0
    elif event == "sync-failed":
        attempted = sequence
        marker_sequence = sequence - 1
        event_status = "unspecified-launch-failure"
        d2h_attempted = "no"
        d2h_completed = "no"
        h2d_attempted = "no"
        h2d_completed = "no"
        interpretation = "failure-surfaced-before-result-d2h"
        confirmed = sequence - 1
        failed = 1
        gather_failed = 0
    elif event == "marker-invalid":
        attempted = sequence
        marker_sequence = sequence - 1
        event_status = "complete"
        d2h_attempted = "no"
        d2h_completed = "no"
        h2d_attempted = "no"
        h2d_completed = "no"
        interpretation = "post-compute-event-confirmed-marker-invalid"
        confirmed = sequence - 1
        failed = 1
        gather_failed = 0
    elif event == "result-gather-failed":
        attempted = sequence
        marker_sequence = sequence
        event_status = "complete"
        d2h_attempted = d2h_attempted_override or "yes"
        d2h_completed = d2h_completed_override or "no"
        h2d_attempted = h2d_attempted_override or "no"
        h2d_completed = h2d_completed_override or "no"
        interpretation = (
            "failure-surfaced-after-confirmed-result-h2d-complete"
            if h2d_completed == "yes" else
            "failure-surfaced-after-confirmed-result-h2d-attempt"
            if h2d_attempted == "yes" else
            "failure-surfaced-after-confirmed-result-d2h-complete-before-result-h2d"
            if d2h_completed == "yes" else
            "failure-surfaced-after-confirmed-result-d2h-attempt"
            if d2h_attempted == "yes" else
            "failure-surfaced-after-confirmed-before-result-d2h"
        )
        confirmed = sequence
        failed = 0
        gather_failed = 1
    else:
        raise AssertionError(f"unknown event: {event}")
    complement = (~marker_sequence) & ((1 << 64) - 1)
    marker_matches = "yes" if marker_sequence == sequence else "no"
    stage = "result-gather" if event == "result-gather-failed" else "pre-result-d2h"
    return (
        "ds4: CUDA q8 partner pre-gather fence failure "
        f"event={event} stage={stage} current_sequence={sequence} "
        f"marker_sequence={marker_sequence} marker_complement={complement} "
        f"marker_matches={marker_matches} event_status={event_status} "
        f"result_d2h_attempted={d2h_attempted} "
        f"result_d2h_completed={d2h_completed} "
        f"result_h2d_attempted={h2d_attempted} "
        f"result_h2d_completed={h2d_completed} interpretation={interpretation} "
        f"attempted={attempted} confirmed={confirmed} failed={failed} "
        f"result_gather_failed_after_confirmed={gather_failed} "
        "home_tier=0 home_device=0 partner_tier=2 partner_device=1 "
        "tokens=512 in=8192 out=4096 "
        "binding_label=tensor:blk.0.attn_output_b.weight "
        "passed_label=attn_output_b weight_offset=141234898176\n"
    )


def q8_pre_gather_summary(sequence: int) -> str:
    return q8_async_summary(sequence, sequence, sequence).rstrip("\n") + (
        f" pre_gather_attempted={sequence} "
        f"pre_gather_confirmed={sequence} pre_gather_failed=0 "
        "pre_gather_result_gather_failed_after_confirmed=0 "
        f"pre_gather_last_sequence={sequence}\n"
    )


Q8_ACTIVATION_TAG = 1 << 63


def q8_activation_identity(
        *, home_device: int = 0, partner_device: int = 1,
        binding_label: str = "tensor:blk.0.attn_output_b.weight") -> str:
    return (
        f"home_tier=0 home_device={home_device} partner_tier=2 "
        f"partner_device={partner_device} bytes=8388608 tokens=512 "
        f"in=8192 out=4096 binding_label={binding_label} "
        "passed_label=attn_output_b weight_offset=141234898176"
    )


def q8_activation_marker(sequence: int) -> tuple[int, int]:
    marker = sequence | Q8_ACTIVATION_TAG
    return marker, (~marker) & ((1 << 64) - 1)


def q8_activation_enabled() -> str:
    return (
        "ds4: CUDA q8 partner pre-activation fence audit enabled: "
        "logical_pairs=0 boundary=post-source-d2h-destination-sync "
        "marker=destination-default-stream-tagged-exact-before-activation-h2d\n"
    )


def q8_activation_armed(sequence: int, **identity: object) -> str:
    marker, complement = q8_activation_marker(sequence)
    return (
        "ds4: CUDA q8 partner pre-activation armed "
        f"current_sequence={sequence} marker_sequence={marker} "
        f"marker_complement={complement} activation_d2h_status=complete "
        "destination_sync_status=complete " +
        q8_activation_identity(**identity) + "\n"
    )


def q8_activation_returned(sequence: int, **identity: object) -> str:
    return (
        "ds4: CUDA q8 partner pre-activation returned "
        f"current_sequence={sequence} activation_h2d_status=success " +
        q8_activation_identity(**identity) + "\n"
    )


def q8_activation_checkpoint(sequence: int, **identity: object) -> str:
    marker, complement = q8_activation_marker(sequence)
    return (
        "ds4: CUDA q8 partner pre-activation fence checkpoint "
        "event=marker-confirmed stage=pre-activation-h2d "
        f"current_sequence={sequence} marker_sequence={marker} "
        f"marker_complement={complement} marker_matches=yes cuda_status=complete "
        "destination_sync_status=complete activation_d2h_attempted=yes "
        "activation_d2h_completed=yes activation_h2d_attempted=no "
        "activation_h2d_completed=no interpretation=activation-d2h-and-"
        "destination-stream-confirmed-before-h2d "
        f"attempted={sequence} confirmed={sequence} returned={sequence - 1} "
        "failed=0 " + q8_activation_identity(**identity) + "\n"
    )


def q8_activation_completed_calls(count: int) -> str:
    lines: list[str] = []
    for sequence in range(1, count + 1):
        lines.append(q8_activation_armed(sequence))
        if sequence == 1 or sequence % 64 == 0:
            lines.append(q8_activation_checkpoint(sequence))
        lines.append(q8_activation_returned(sequence))
    return "".join(lines)


def q8_activation_failure(sequence: int, event: str, **identity: object) -> str:
    specs = {
        "bounce-free-failed": (
            "activation-staging", "cudaErrorUnknown", "not-attempted",
            "no", "no", "no", "no",
            "failure-surfaced-by-bounce-free-before-activation-d2h", False,
        ),
        "activation-d2h-failed": (
            "activation-d2h", "cudaErrorUnknown", "not-attempted",
            "yes", "no", "no", "no",
            "failure-surfaced-by-activation-d2h", False,
        ),
        "destination-switch-failed": (
            "pre-activation-h2d", "cudaSetDevice-failed", "not-attempted",
            "yes", "yes", "no", "no",
            "failure-after-activation-d2h-before-destination-sync", False,
        ),
        "destination-sync-failed": (
            "pre-activation-h2d", "cudaErrorUnknown", "failed",
            "yes", "yes", "no", "no",
            "failure-surfaced-by-pre-h2d-destination-device-sync", False,
        ),
        "marker-launch-failed": (
            "pre-activation-marker", "cudaErrorUnknown", "complete",
            "yes", "yes", "no", "no",
            "failure-surfaced-by-pre-h2d-marker-launch", False,
        ),
        "activation-h2d-failed": (
            "activation-h2d", "cudaErrorUnknown", "complete",
            "yes", "yes", "yes", "no",
            "failure-surfaced-by-activation-h2d-after-confirmed-boundary", True,
        ),
    }
    (stage, cuda_status, sync_status, d2h_attempted, d2h_completed,
     h2d_attempted, h2d_completed, interpretation, confirmed_marker) = specs[event]
    if confirmed_marker:
        marker, complement = q8_activation_marker(sequence)
        marker_matches = "yes"
        confirmed = sequence
    else:
        marker = complement = 0
        marker_matches = "no"
        confirmed = sequence - 1
    return (
        "ds4: CUDA q8 partner pre-activation fence failure "
        f"event={event} stage={stage} current_sequence={sequence} "
        f"marker_sequence={marker} marker_complement={complement} "
        f"marker_matches={marker_matches} cuda_status={cuda_status} "
        f"destination_sync_status={sync_status} "
        f"activation_d2h_attempted={d2h_attempted} "
        f"activation_d2h_completed={d2h_completed} "
        f"activation_h2d_attempted={h2d_attempted} "
        f"activation_h2d_completed={h2d_completed} "
        f"interpretation={interpretation} attempted={sequence} "
        f"confirmed={confirmed} returned={sequence - 1} failed=1 " +
        q8_activation_identity(**identity) + "\n"
    )


def q8_activation_sequence_domain_exhausted() -> str:
    sequence = Q8_ACTIVATION_TAG
    return (
        "ds4: CUDA q8 partner pre-activation fence failure "
        "event=state-invalid stage=pre-activation-marker "
        f"current_sequence={sequence} marker_sequence=0 marker_complement=0 "
        "marker_matches=no cuda_status=sequence-domain-exhausted "
        "destination_sync_status=not-attempted "
        "activation_d2h_attempted=no activation_d2h_completed=no "
        "activation_h2d_attempted=no activation_h2d_completed=no "
        "interpretation=failure-before-activation-d2h "
        f"attempted={sequence} confirmed={sequence - 1} "
        f"returned={sequence - 1} failed=1 " +
        q8_activation_identity() + "\n"
    )


def q8_activation_summary(
        sequence: int, *, confirmed: int | None = None,
        returned: int | None = None, failed: int = 0,
        last_sequence: int | None = None, shared: int = 777,
        shared_complement: int | None = None,
        partners_synchronized: str = "yes") -> str:
    # A later untagged post-compute marker may legitimately own the shared slot.
    confirmed = sequence if confirmed is None else confirmed
    returned = sequence if returned is None else returned
    last_sequence = confirmed if last_sequence is None else last_sequence
    shared_complement = (
        (~shared) & ((1 << 64) - 1)
        if shared_complement is None else shared_complement
    )
    return (
        "ds4: CUDA q8 partner pre-activation fence summary "
        f"home_tier=0 partner_tier=2 attempted={sequence} confirmed={confirmed} "
        f"returned={returned} failed={failed} last_sequence={last_sequence} "
        f"shared_slot_sequence={shared} "
        f"shared_slot_complement={shared_complement} "
        f"partners_synchronized={partners_synchronized}\n"
    )


class SummarizeSmallBar1PairIsolationTest(unittest.TestCase):
    def test_identifies_prefill_attention_rows_as_necessary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            statuses = {"attention-off": "passed", "production": "failed"}
            for slot, (variant, status) in enumerate(statuses.items(), 1):
                stem = f"r1-s{slot}-{variant}"
                (production / f"{stem}.result").write_text(
                    f"variant={variant}\nstatus={status}\nexit_status="
                    f"{0 if status == 'passed' else 1}\n"
                    "last_phase=measured-prefill\nlast_event=chunk-start\n"
                )
                (production / f"{stem}.log").write_text("")
                if status == "passed":
                    write_healthy_post(root, stem)
                else:
                    write_unhealthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("prefill attention row split disabled", report)
            self.assertIn("pair-0 decode-indexer splitting", report)
            self.assertIn("Pair-0 prefill indexer splitting also stayed home", report)
            self.assertIn("necessary condition", report)

    def test_rejects_power_limit_drift_as_causal_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            telemetry = root / "telemetry"
            production.mkdir()
            telemetry.mkdir()
            stem = "r1-s1-attention-off"
            (production / f"{stem}.result").write_text(
                "variant=attention-off\nstatus=failed\nexit_status=126\n"
                "last_phase=measured-prefill\nlast_event=chunk-start\n"
            )
            (production / f"{stem}.log").write_text("")
            (telemetry / f"{stem}-watch-event.txt").write_text(
                "status=power-limit-drift\nrequired_power_limit_w=250\n"
            )

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("invalid for causal comparison", report)
            self.assertIn("external power-limit writer", report)

    def test_recovers_completed_attention_off_without_result_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-off"
            (production / f"{stem}.started").write_text(
                "variant=attention-off\nrepeat=1\n"
            )
            (production / f"{stem}.log").write_text("")
            (production / f"{stem}.csv").write_text(
                "ctx_tokens,prefill_tps,gen_tps\n32768,487.24,16.89\n"
            )
            (production / f"{stem}-progress.csv").write_text(
                "realtime_sec,realtime_nsec,phase,event,current,total\n"
                "1,0,decode,frontier-complete,256,256\n"
            )
            write_healthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("completed-no-result", report)
            self.assertIn("older wrapper omitted its final result record", report)
            self.assertIn("mixed prefill split-attention", report)

    def test_interrupted_prior_run_without_device_loss_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-production"
            (production / f"{stem}.result").write_text(
                "variant=production\nstatus=interrupted-prior-run\n"
                "exit_status=125\nlast_phase=measured-prefill\n"
                "last_event=chunk-start\n"
            )
            (production / f"{stem}.log").write_text("")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["status"], "interrupted-prior-run")
            report = (root / "summary.md").read_text()
            self.assertIn("production control has no verified outcome", report)
            self.assertIn("fresh directory", report)

    def test_generic_failed_result_without_device_loss_is_unverified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-production"
            (production / f"{stem}.result").write_text(
                "variant=production\nstatus=failed\nexit_status=1\n"
                "last_phase=measured-prefill\nlast_event=chunk-start\n"
            )
            (production / f"{stem}.log").write_text("")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["status"], "failed-unverified")
            report = (root / "summary.md").read_text()
            self.assertIn("production control has no verified outcome", report)

    def test_lost_device_watch_corroborates_interrupted_prior_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            telemetry = root / "telemetry"
            production.mkdir()
            telemetry.mkdir()
            stem = "r1-s1-production"
            (production / f"{stem}.result").write_text(
                "variant=production\nstatus=interrupted-prior-run\n"
                "exit_status=125\nlast_phase=measured-prefill\n"
                "last_event=chunk-start\n"
            )
            (production / f"{stem}.log").write_text("")
            (telemetry / f"{stem}-watch-event.txt").write_text(
                "status=lost-device-detected\n"
            )

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["status"], "interrupted-prior-run-device-loss"
            )

    def test_lost_device_watch_corroborates_started_only_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            telemetry = root / "telemetry"
            production.mkdir()
            telemetry.mkdir()
            stem = "r1-s1-production"
            (production / f"{stem}.started").write_text(
                "variant=production\nrepeat=1\n"
            )
            (production / f"{stem}.log").write_text("")
            (telemetry / f"{stem}-watch-event.txt").write_text(
                "status=lost-device-detected\n"
            )

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["status"], "interrupted-no-result-device-loss"
            )

    def test_unhealthy_post_corroborates_interrupted_prior_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            health = root / "health"
            production.mkdir()
            health.mkdir()
            stem = "r1-s1-production"
            (production / f"{stem}.result").write_text(
                "variant=production\nstatus=interrupted-prior-run\n"
                "exit_status=125\nlast_phase=measured-prefill\n"
                "last_event=chunk-start\n"
            )
            (production / f"{stem}.log").write_text("")
            (health / f"{stem}-post.log").write_text(
                "GPU 0: ok\nGPU 1: GPU is lost\nGPU 2: ok\nGPU 3: ok\n"
            )

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["status"], "interrupted-prior-run-device-loss"
            )
            self.assertEqual(row["post_health"], "unhealthy")

    def test_empty_post_snapshot_does_not_imply_device_loss(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            health = root / "health"
            production.mkdir()
            health.mkdir()
            stem = "r1-s1-production"
            (production / f"{stem}.result").write_text(
                "variant=production\nstatus=interrupted-prior-run\n"
                "exit_status=125\nlast_phase=measured-prefill\n"
                "last_event=chunk-start\n"
            )
            (production / f"{stem}.log").write_text("")
            (health / f"{stem}-post.log").write_text("")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["status"], "interrupted-prior-run")
            self.assertEqual(row["post_health"], "unverified")

    def test_validation_failure_is_not_reported_as_device_loss(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-phase-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-phase-audit\nstatus=validation-failed\n"
                "exit_status=127\nlast_phase=decode\n"
                "last_event=frontier-complete\n"
            )
            (production / f"{stem}.log").write_text("")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("failed its production-path validation", report)
            self.assertIn("invalid for causal inference", report)
            self.assertNotIn("observation boundary for the device loss", report)

    def test_completed_no_result_does_not_override_failed_repeat(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            health = root / "health"
            production.mkdir()
            health.mkdir()
            completed = "r1-s1-attention-off"
            (production / f"{completed}.started").write_text(
                "variant=attention-off\nrepeat=1\n"
            )
            (production / f"{completed}.log").write_text("")
            (production / f"{completed}.csv").write_text(
                "ctx_tokens,prefill_tps,gen_tps\n32768,487.24,16.89\n"
            )
            (production / f"{completed}-progress.csv").write_text(
                "realtime_sec,realtime_nsec,phase,event,current,total\n"
                "1,0,decode,frontier-complete,256,256\n"
            )
            (health / f"{completed}-post.log").write_text(
                "GPU 0: ok\nGPU 1: ok\nGPU 2: ok\nGPU 3: ok\n"
            )
            failed = "r2-s2-attention-off"
            (production / f"{failed}.result").write_text(
                "variant=attention-off\nstatus=failed\nexit_status=1\n"
            )
            (production / f"{failed}.log").write_text("")
            write_unhealthy_post(root, failed)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("Pair 0 still failed", report)
            self.assertNotIn("Pair 0 survived", report)

    def test_reports_targeted_attention_phase_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-phase-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-phase-audit\nstatus=passed\nexit_status=0\n"
                "last_phase=decode\nlast_event=frontier-complete\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA prefill attention row phase audit event=complete "
                "phase=result-gather kind=mixed layer=17 pos=512 tokens=512 "
                "home=0 partner=2\n"
            )
            write_healthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("All production row-split operations and traffic survived", report)
            self.assertIn("timing/power/traffic envelope", report)
            self.assertIn(
                "complete:result-gather:mixed:layer17:pos512:tokens512:home0:partner2",
                report,
            )

    def test_passed_result_without_post_health_is_not_survival(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-phase-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-phase-audit\nstatus=passed\nexit_status=0\n"
                "last_phase=decode\nlast_event=frontier-complete\n"
            )
            (production / f"{stem}.log").write_text("")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("passed-unverified-health", report)
            self.assertIn("has no validated final outcome", report)
            self.assertNotIn(
                "All production row-split operations and traffic survived",
                report,
            )

    def test_watch_marker_invalidates_passed_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            telemetry = root / "telemetry"
            production.mkdir()
            telemetry.mkdir()
            stem = "r1-s1-attention-query-dst"
            (production / f"{stem}.result").write_text(
                "variant=attention-query-dst\nstatus=passed\nexit_status=0\n"
                "last_phase=decode\nlast_event=frontier-complete\n"
            )
            (production / f"{stem}.log").write_text("")
            write_healthy_post(root, stem)
            (telemetry / f"{stem}-watch-event.txt").write_text(
                "status=foreign-compute-process\n"
            )

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["status"], "passed-invalidated-watch")
            report = (root / "summary.md").read_text()
            self.assertIn("unexpected GPU compute process", report)
            self.assertIn("invalid for causal comparison", report)

    def test_recovers_started_only_phase_audit_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-phase-audit"
            (production / f"{stem}.started").write_text(
                "variant=attention-phase-audit\nrepeat=1\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA prefill attention row phase audit event=begin "
                "phase=partner-attention kind=mixed layer=17 pos=512 tokens=512 "
                "home=0 partner=2\n"
            )
            (production / f"{stem}-progress.csv").write_text(
                "realtime_sec,realtime_nsec,phase,event,current,total\n"
                "1,0,measured-prefill,chunk-start,512,32768\n"
            )

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("interrupted-no-result", report)
            self.assertIn(
                "begin:partner-attention:mixed:layer17:pos512:tokens512:home0:partner2",
                report,
            )
            self.assertIn("has no validated final outcome", report)

    def test_reports_passed_attention_end_fence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-end-fence"
            (production / f"{stem}.result").write_text(
                "variant=attention-end-fence\nstatus=passed\nexit_status=0\n"
                "last_phase=decode\nlast_event=frontier-complete\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA prefill attention row end fence event=complete "
                "target=pair kind=mixed layer=21 pos=512 tokens=512 "
                "home=0 partner=2\n"
            )
            write_healthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("full production path survived", report)
            self.assertIn("not a proposal to disable row splitting", report)
            self.assertIn(
                "complete:pair:mixed:layer21:pos512:tokens512:home0:partner2",
                report,
            )

    def test_recovers_started_only_attention_end_fence_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-end-fence"
            (production / f"{stem}.started").write_text(
                "variant=attention-end-fence\nrepeat=1\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA prefill attention row end fence event=begin "
                "target=pair kind=mixed layer=21 pos=512 tokens=512 "
                "home=0 partner=2\n"
                "ds4: CUDA prefill attention row end fence event=complete "
                "target=partner kind=mixed layer=21 pos=512 tokens=512 "
                "home=0 partner=2\n"
                "ds4: CUDA prefill attention row end fence event=sync-failed "
                "target=home kind=mixed layer=21 pos=512 tokens=512 "
                "home=0 partner=2\n"
                "ds4: CUDA prefill attention row end fence event=failed "
                "target=home kind=mixed layer=21 pos=512 tokens=512 "
                "home=0 partner=2\n"
            )
            write_unhealthy_post(root, stem)
            (production / f"{stem}-progress.csv").write_text(
                "realtime_sec,realtime_nsec,phase,event,current,total\n"
                "1,0,measured-prefill,prefill_chunk,0,32768\n"
            )

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("interrupted-no-result", report)
            self.assertIn(
                "failed:home:mixed:layer21:pos512:tokens512:home0:partner2",
                report,
            )
            self.assertIn("partner end-fence completed", report)
            self.assertIn("following home boundary failed", report)

    def test_distinguishes_failure_after_completed_attention_end_fence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-end-fence"
            (production / f"{stem}.result").write_text(
                "variant=attention-end-fence\nstatus=failed\nexit_status=1\n"
                "last_phase=measured-prefill\nlast_event=prefill_chunk\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA prefill attention row end fence event=complete "
                "target=pair kind=mixed layer=21 pos=512 tokens=512 "
                "home=0 partner=2\n"
            )
            write_unhealthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("completed on both partner and home", report)
            self.assertIn("host-confirmed complete", report)
            self.assertIn("first subsequent CUDA observation lies", report)
            self.assertIn("delayed physical fault", report)

    def test_distinguishes_partner_attention_end_fence_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-end-fence"
            (production / f"{stem}.result").write_text(
                "variant=attention-end-fence\nstatus=failed\nexit_status=1\n"
                "last_phase=measured-prefill\nlast_event=prefill_chunk\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA prefill attention row end fence event=sync-failed "
                "target=partner kind=mixed layer=21 pos=512 tokens=512 "
                "home=0 partner=2\n"
                "ds4: CUDA prefill attention row end fence event=failed "
                "target=partner kind=mixed layer=21 pos=512 tokens=512 "
                "home=0 partner=2\n"
            )
            write_unhealthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("partner synchronization was the first failing", report)
            self.assertIn("Home synchronization was deliberately not attempted", report)

    def test_layer_boundary_audit_localizes_failure_before_l18_query_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-row-boundary-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-row-boundary-audit\nstatus=failed\n"
                "exit_status=1\nlast_phase=measured-prefill\n"
                "last_event=prefill_chunk\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA prefill attention row end fence event=complete "
                "target=pair kind=mixed layer=17 pos=512 tokens=512 "
                "home=0 partner=2\n"
                "ds4: CUDA prefill attention row entry fence event=begin "
                "target=pair kind=mixed layer=18 pos=512 tokens=512 "
                "home=0 partner=2\n"
                "ds4: CUDA prefill attention row entry fence event=sync-failed "
                "target=partner kind=mixed layer=18 pos=512 tokens=512 "
                "home=0 partner=2\n"
                "ds4: CUDA prefill attention row entry fence event=failed "
                "target=partner kind=mixed layer=18 pos=512 tokens=512 "
                "home=0 partner=2\n"
            )

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn(
                "failed:partner:mixed:layer18:pos512:tokens512:home0:partner2",
                report,
            )
            self.assertIn("before layer-18 query copy was submitted", report)
            self.assertIn("layer-17 tail/MoE/Q8 work", report)
            self.assertIn("does not prove a layer-18 kernel bug", report)

    def test_layer_boundary_audit_reports_failure_inside_l18_phase(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-row-boundary-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-row-boundary-audit\nstatus=failed\n"
                "exit_status=1\nlast_phase=measured-prefill\n"
                "last_event=prefill_chunk\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA prefill attention row end fence event=complete "
                "target=pair kind=mixed layer=17 pos=512 tokens=512 "
                "home=0 partner=2\n"
                "ds4: CUDA prefill attention row entry fence event=complete "
                "target=pair kind=mixed layer=18 pos=512 tokens=512 "
                "home=0 partner=2\n"
                "ds4: CUDA prefill attention row phase audit event=sync-failed "
                "phase=partner-attention kind=mixed layer=18 pos=512 tokens=512 "
                "home=0 partner=2\n"
            )

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn(
                "sync-failed:partner-attention:mixed:layer18:pos512:tokens512:"
                "home0:partner2",
                report,
            )
            self.assertIn("layer-18 row-launch/pre-query fence completed", report)
            self.assertIn("first failure-class marker is inside", report)
            self.assertIn("not by itself proof of a layer-specific", report)

    def test_reports_passed_layer_boundary_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-row-boundary-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-row-boundary-audit\nstatus=passed\n"
                "exit_status=0\nlast_phase=decode\n"
                "last_event=frontier-complete\n"
            )
            (production / f"{stem}.log").write_text(complete_row_boundary_log())
            write_healthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["attention_entry_fence_last"],
                "complete:pair:mixed:layer18:pos512:tokens512:home0:partner2",
            )
            self.assertEqual(row["attention_row_boundary_marker_state"], "complete")
            report = (root / "summary.md").read_text()
            self.assertIn("healthy post-run snapshot passed", report)
            self.assertIn("immediately before layer 18 query copy", report)
            self.assertIn("does not identify a layer-specific software defect", report)

    def test_recovers_completed_boundary_audit_without_result_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-row-boundary-audit"
            (production / f"{stem}.started").write_text(
                "variant=attention-row-boundary-audit\n"
            )
            (production / f"{stem}.log").write_text(complete_row_boundary_log())
            (production / f"{stem}.csv").write_text(
                "pp_tokens,prefill_tps,gen_tps\n32768,600.0,16.0\n"
            )
            (production / f"{stem}-progress.csv").write_text(
                "realtime_sec,realtime_nsec,phase,event,current,total\n"
                "1,0,decode,frontier-complete,256,256\n"
            )
            write_healthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["status"], "completed-no-result")
            report = (root / "summary.md").read_text()
            self.assertIn("wrapper omitted its final result record", report)
            self.assertIn("healthy post-run snapshot passed", report)

    def test_ctrl_c_after_complete_phases_is_not_called_gpu_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-row-boundary-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-row-boundary-audit\nstatus=failed\n"
                "exit_status=130\nlast_phase=interrupted\nlast_event=prefill_chunk\n"
            )
            (production / f"{stem}.log").write_text(complete_row_boundary_log())

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("no lost-device watcher record", report)
            self.assertIn("not evidence of a GPU failure", report)
            self.assertNotIn("first observed error is downstream", report)

    def test_lost_device_after_complete_phases_is_downstream_observation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            telemetry = root / "telemetry"
            telemetry.mkdir()
            stem = "r1-s1-attention-row-boundary-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-row-boundary-audit\nstatus=failed\n"
                "exit_status=143\nlast_phase=measured-prefill\n"
                "last_event=prefill_chunk\n"
            )
            (production / f"{stem}.log").write_text(complete_row_boundary_log())
            (telemetry / f"{stem}-watch-event.txt").write_text(
                "status=lost-device-detected\n"
            )

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("independently detected device loss", report)
            self.assertIn("downstream of layer-18 attention completion", report)

    def test_validates_exact_row_boundary_marker_state_machine(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log_path = pathlib.Path(temporary) / "boundary.log"
            complete = complete_row_boundary_log()
            command = [
                sys.executable,
                str(SUMMARIZER),
                "--validate-attention-row-boundary-log",
                str(log_path),
                "17",
                "18",
                "512",
            ]
            log_path.write_text(complete)
            subprocess.run(command, check=True)

            mutations = {
                "wrong-kind": complete.replace(
                    "phase audit event=begin phase=query-copy kind=mixed",
                    "phase audit event=begin phase=query-copy kind=indexed",
                    1,
                ),
                "wrong-position": complete.replace(
                    "entry fence event=begin target=pair kind=mixed layer=18 pos=512",
                    "entry fence event=begin target=pair kind=mixed layer=18 pos=1024",
                    1,
                ),
                "unexpected-l17-entry": complete.replace(
                    "prefill attention row audit dispatch=split kind=mixed layer=17",
                    "prefill attention row entry fence event=begin target=pair "
                    "kind=mixed layer=17 pos=512 tokens=512 home=0 partner=2\n"
                    "ds4: CUDA prefill attention row audit dispatch=split "
                    "kind=mixed layer=17",
                    1,
                ),
                "missing-l18-dispatch": complete.rsplit("\n", 2)[0] + "\n",
                "duplicate-phase": complete.replace(
                    "ds4: CUDA prefill attention row phase audit event=complete "
                    "phase=query-copy",
                    "ds4: CUDA prefill attention row phase audit event=complete "
                    "phase=query-copy kind=mixed layer=18 pos=512 tokens=512 "
                    "home=0 partner=2\n"
                    "ds4: CUDA prefill attention row phase audit event=complete "
                    "phase=query-copy",
                    1,
                ),
            }
            for name, mutated in mutations.items():
                with self.subTest(name=name):
                    log_path.write_text(mutated)
                    result = subprocess.run(
                        command, text=True, capture_output=True, check=False
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("marker sequence is", result.stderr)

    def test_first_phase_failure_survives_secondary_cleanup_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-row-boundary-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-row-boundary-audit\nstatus=failed\n"
                "exit_status=1\nlast_phase=measured-prefill\n"
                "last_event=prefill_chunk\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA prefill attention row end fence event=complete "
                "target=pair kind=mixed layer=17 pos=512 tokens=512 "
                "home=0 partner=2\n"
                "ds4: CUDA prefill attention row entry fence event=complete "
                "target=pair kind=mixed layer=18 pos=512 tokens=512 "
                "home=0 partner=2\n"
                "ds4: CUDA prefill attention row phase audit event=sync-failed "
                "phase=query-copy kind=mixed layer=18 pos=512 tokens=512 "
                "home=0 partner=2\n"
                "ds4: CUDA prefill attention row phase audit "
                "event=device-switch-failed phase=home-attention kind=mixed "
                "layer=18 pos=512 tokens=512 home=0 partner=2\n"
            )
            write_unhealthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["attention_audit_first_failure"],
                "phase-audit:sync-failed:query-copy:mixed:layer18:pos512:"
                "tokens512:home0:partner2",
            )
            self.assertEqual(
                row["attention_phase_audit_last"],
                "device-switch-failed:home-attention:mixed:layer18:pos512:"
                "tokens512:home0:partner2",
            )
            report = (root / "summary.md").read_text()
            self.assertIn("phase-audit:sync-failed:query-copy", report)
            self.assertIn("secondary cleanup failure cannot overwrite", report)

    def test_identifies_direct_or_overlap_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            cases = {
                "partner-bounce": ("passed", "decode", "frontier-complete"),
                "bounce-indexer-off": ("failed", "decode", "token-start"),
                "partner-serialized": ("passed", "decode", "frontier-complete"),
                "indexer-off": ("failed", "decode", "token-start"),
                "production": ("failed", "measured-prefill", "chunk-complete"),
            }
            for slot, (variant, (status, phase, event)) in enumerate(cases.items(), 1):
                stem = f"r1-s{slot}-{variant}"
                (production / f"{stem}.result").write_text(
                    f"variant={variant}\nstatus={status}\nexit_status="
                    f"{0 if status == 'passed' else 1}\nlast_phase={phase}\n"
                    f"last_event={event}\nlast_current=16\nlast_total=256\n"
                )
                (production / f"{stem}.log").write_text(
                    "ds4: CUDA q8 partner transfer audit event=begin "
                    "home_tier=0 partner_tier=2 calls=64 bytes=4096 "
                    "activation_bytes=32 result_bytes=32 tokens=1 "
                    f"transport={'host-bounce' if 'bounce' in variant else 'peer'} "
                    f"serialized={'yes' if variant == 'partner-serialized' else 'no'}\n"
                    "ds4: CUDA decode indexer row audit event=begin layer=2 "
                    "home_tier=0 partner_tier=2 n_comp=1024 transfer_bytes=256\n"
                )
                if status == "passed":
                    write_healthy_post(root, stem)
                else:
                    write_unhealthy_post(root, stem)
            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 5)
            production_row = next(row for row in rows
                                  if row["variant"] == "production")
            self.assertEqual(production_row["last_phase"], "measured-prefill")
            self.assertEqual(production_row["pair0_q8_begin_checkpoint_bytes"], "4096")
            self.assertEqual(production_row["pair0_indexer_begin_bytes"], "256")
            bounce_row = next(row for row in rows
                              if row["variant"] == "partner-bounce")
            self.assertEqual(bounce_row["pair0_q8_transport"], "host-bounce")
            serialized_row = next(row for row in rows
                                  if row["variant"] == "partner-serialized")
            self.assertEqual(serialized_row["pair0_q8_serialized"], "yes")
            report = (root / "summary.md").read_text()
            self.assertIn("direct P2P/BAR1 traffic", report)
            self.assertIn("instantaneous load", report)

    def test_identifies_indexer_requirement_across_transports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            statuses = {
                "partner-bounce": "failed",
                "bounce-indexer-off": "passed",
                "partner-serialized": "failed",
                "indexer-off": "passed",
                "production": "failed",
            }
            for slot, (variant, status) in enumerate(statuses.items(), 1):
                stem = f"r1-s{slot}-{variant}"
                (production / f"{stem}.result").write_text(
                    f"variant={variant}\nstatus={status}\nexit_status=1\n"
                    "last_phase=measured-prefill\nlast_event=chunk-start\n"
                    "last_current=2048\nlast_total=32768\n"
                )
                (production / f"{stem}.log").write_text("")
                if status == "passed":
                    write_healthy_post(root, stem)
                else:
                    write_unhealthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("under both direct-peer and host-bounce", report)
            self.assertIn("indexer path or its interaction", report)

    def test_reports_query_destination_stream_survival(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            variants = (
                ("attention-query-dst", "passed"),
                ("attention-gather-dst", "failed"),
                ("attention-both-dst", "failed"),
                ("production", "failed"),
            )
            for slot, (variant, status) in enumerate(variants, 1):
                stem = f"r1-s{slot}-{variant}"
                (production / f"{stem}.result").write_text(
                    f"variant={variant}\nstatus={status}\nexit_status="
                    f"{'0' if status == 'passed' else '1'}\n"
                    "last_phase=decode\nlast_event=frontier-complete\n"
                )
                (production / f"{stem}.log").write_text(
                    "ds4: CUDA prefill attention row audit dispatch=split "
                    "kind=mixed layer=17 pos=512 tokens=512 home=0 partner=2 "
                    "q_bytes=65536 result_bytes=65536 "
                    "query_copy_stream=destination gather_copy_stream=source\n"
                    if variant == "attention-query-dst" else ""
                )
                if status == "passed":
                    write_healthy_post(root, stem)
                else:
                    write_unhealthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("Query-copy destination scheduling passed", report)
            self.assertIn("scheduling/peer-access axis", report)
            self.assertIn("not identification of a physical copy engine", report)
            with (root / "summary.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            query_row = next(
                row for row in rows
                if row["variant"] == "attention-query-dst"
            )
            self.assertEqual(
                query_row["pair0_attention_query_copy_schedule"], "destination"
            )
            self.assertEqual(
                query_row["pair0_attention_gather_copy_schedule"], "source"
            )

    def test_reports_attention_host_bounce_transport_and_survival(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-host-bounce"
            (production / f"{stem}.result").write_text(
                "variant=attention-host-bounce\nstatus=passed\nexit_status=0\n"
                "last_phase=decode\nlast_event=frontier-complete\n"
            )
            (production / f"{stem}.csv").write_text(
                "ctx_tokens,prefill_tps,gen_tps\n32768,575.00,15.00\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA prefill attention cache mirror transport=host-bounce "
                "home_tier=0 partner_tier=2 class=raw event=complete "
                "bytes=33554432\n"
                "ds4: CUDA prefill attention host-bounce checkpoint "
                "event=complete kind=mixed layer=17 pos=512 tokens=512 "
                "home=0 partner=2\n"
                "ds4: CUDA prefill attention row audit dispatch=split "
                "kind=indexed layer=27 pos=16384 tokens=512 home=0 partner=2 "
                "q_bytes=33554432 result_bytes=33554432 "
                "query_copy_stream=source gather_copy_stream=source "
                "query_copy_transport=host-bounce "
                "gather_copy_transport=host-bounce "
                "topk_copy_transport=partner-local\n"
            )
            write_healthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["pair0_attention_query_copy_schedule"], "source")
            self.assertEqual(row["pair0_attention_gather_copy_schedule"], "source")
            self.assertEqual(
                row["pair0_attention_query_copy_transport"], "host-bounce"
            )
            self.assertEqual(
                row["pair0_attention_gather_copy_transport"], "host-bounce"
            )
            self.assertEqual(
                row["pair0_attention_cache_copy_transport"], "host-bounce"
            )
            self.assertEqual(
                row["pair0_attention_topk_copy_transport"], "partner-local"
            )
            self.assertEqual(
                row["attention_host_bounce_checkpoint"],
                "complete:mixed:layer17:pos512:tokens512:home0:partner2",
            )
            report = (root / "summary.md").read_text()
            self.assertIn("attention-owned query, mirrored-cache", report)
            self.assertIn("top-k remained partner-local", report)
            self.assertIn("no cross-device top-k payload transfer", report)
            self.assertIn("direct attention P2P/BAR1 path", report)
            self.assertIn("not by itself a power-matched proof", report)

    def test_reports_attention_host_bounce_device_loss(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-host-bounce"
            (production / f"{stem}.result").write_text(
                "variant=attention-host-bounce\nstatus=failed\nexit_status=1\n"
                "last_phase=measured-prefill\nlast_event=chunk-start\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA prefill attention host-bounce checkpoint "
                "event=complete kind=mixed layer=0 pos=0 tokens=512 "
                "home=0 partner=2\n"
            )
            write_unhealthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("no durable measured pos=512", report)
            self.assertIn("real failure but inconclusive", report)
            self.assertNotIn("not a necessary trigger", report)

    def test_reports_attention_host_bounce_device_loss_after_checkpoint(
            self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-host-bounce"
            (production / f"{stem}.result").write_text(
                "variant=attention-host-bounce\nstatus=failed\nexit_status=1\n"
                "last_phase=measured-prefill\nlast_event=chunk-start\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA prefill attention host-bounce checkpoint "
                "event=begin kind=mixed layer=17 pos=512 tokens=512 "
                "home=0 partner=2\n"
            )
            write_unhealthy_post(root, stem)
            write_lost_watch(root, stem, "1@00000000:03:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["attention_host_bounce_checkpoint"],
                "begin:mixed:layer17:pos512:tokens512:home0:partner2",
            )
            self.assertEqual(row["lost_devices"], "1@00000000:03:00.0")
            report = (root / "summary.md").read_text()
            self.assertIn("durable measured pos=512 begin/failed checkpoint", report)
            self.assertIn("forced-host-bounce submission boundary", report)
            self.assertIn(
                "does not claim that the pos=512 attention operation completed",
                report,
            )
            self.assertIn("No top-k transfer was observed", report)
            self.assertIn("not proof that none occurred", report)
            self.assertIn(
                "not necessary for the observed pair-0 device loss", report
            )
            self.assertIn("partner attention execution", report)
            self.assertIn("pair-1 direct traffic", report)

    def test_reports_completed_attention_host_bounce_before_device_loss(
            self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-host-bounce"
            (production / f"{stem}.result").write_text(
                "variant=attention-host-bounce\nstatus=failed\nexit_status=1\n"
                "last_phase=measured-prefill\nlast_event=chunk-start\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA prefill attention host-bounce checkpoint "
                "event=complete kind=mixed layer=2 pos=512 tokens=512 "
                "home=0 partner=2\n"
            )
            write_unhealthy_post(root, stem)
            write_lost_watch(root, stem, "1@00000000:03:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("event=complete checkpoint", report)
            self.assertIn(
                "measured layer at pos=512 completed end-to-end", report
            )
            self.assertNotIn(
                "does not claim that the pos=512 attention operation completed",
                report,
            )
            self.assertIn("No top-k transfer was observed", report)
            self.assertIn("not proof that none occurred", report)

    def test_combined_host_bounce_failure_requires_both_measured_routes(
            self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-host-bounce"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-host-bounce\nstatus=failed\nexit_status=1\n"
                "last_phase=measured-prefill\nlast_event=chunk-start\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA q8 partner transfer audit event=begin "
                "home_tier=0 partner_tier=2 calls=128 bytes=4096 "
                "activation_bytes=32 result_bytes=32 tokens=512 "
                "transport=host-bounce serialized=no\n"
                "ds4: CUDA q8 partner transfer audit event=complete "
                "home_tier=0 partner_tier=2 calls=128 bytes=4096\n"
                "ds4: CUDA prefill attention cache mirror transport=host-bounce "
                "home_tier=0 partner_tier=2 class=raw event=complete bytes=4096\n"
                "ds4: CUDA prefill attention cache mirror transport=host-bounce "
                "home_tier=0 partner_tier=2 class=attn-comp event=complete bytes=4096\n"
                "ds4: CUDA prefill attention cache mirror transport=host-bounce "
                "home_tier=0 partner_tier=2 class=index event=complete bytes=4096\n"
                "ds4: CUDA prefill attention host-bounce checkpoint "
                "event=complete kind=mixed layer=2 pos=512 tokens=512 "
                "home=0 partner=2\n"
                "ds4: CUDA prefill attention row audit dispatch=split "
                "kind=mixed layer=2 pos=512 tokens=512 home=0 partner=2 "
                "query_copy_stream=source gather_copy_stream=source "
                "query_copy_transport=host-bounce "
                "gather_copy_transport=host-bounce "
                "topk_copy_transport=none\n"
                "ds4: CUDA default-stream bounce h2d failed: "
                "unspecified launch failure\n"
                "ds4: CUDA prefill attention host-bounce failure "
                "class=result-gather stage=copy kind=mixed layer=16 pos=512 "
                "tokens=512 source_tier=2 source_device=1 "
                "destination_tier=0 destination_device=0 bytes=33554432\n"
            )
            write_unhealthy_post(root, stem)
            write_lost_watch(root, stem, "1@00000000:03:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["variant"], "attention-q8-host-bounce")
            self.assertEqual(
                row["pair0_q8_complete_checkpoint_calls"], "128"
            )
            self.assertEqual(
                row["pair0_attention_cache_host_bounce_classes"],
                "attn-comp,index,raw",
            )
            self.assertEqual(
                row["host_bounce_failure_context"],
                "cuda_phase=h2d class=result-gather stage=copy kind=mixed "
                "layer=16 pos=512 "
                "tokens=512 source_tier=2 source_device=1 "
                "destination_tier=0 destination_device=0 bytes=33554432",
            )
            report = (root / "summary.md").read_text()
            self.assertIn(
                "direct pair-0 production attention-owned and Q8-partner payload "
                "transport is unnecessary",
                report,
            )
            self.assertIn("Pair-1 traffic remained direct peer", report)
            self.assertIn("was not removed", report)

    def test_combined_host_bounce_warmup_only_is_inconclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-host-bounce"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-host-bounce\nstatus=failed\nexit_status=1\n"
                "last_phase=measured-prefill\nlast_event=chunk-start\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA q8 partner transfer audit event=begin "
                "home_tier=0 partner_tier=2 calls=64 bytes=2048 "
                "activation_bytes=32 result_bytes=32 tokens=512 "
                "transport=host-bounce serialized=no\n"
                "ds4: CUDA q8 partner transfer audit event=complete "
                "home_tier=0 partner_tier=2 calls=64 bytes=2048\n"
                "ds4: CUDA prefill attention host-bounce checkpoint "
                "event=complete kind=mixed layer=0 pos=0 tokens=512 "
                "home=0 partner=2\n"
            )
            write_unhealthy_post(root, stem)
            write_lost_watch(root, stem, "1@00000000:03:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("warmup evidence only", report)
            self.assertIn(
                "Do not infer that the cut pair-0 attention/Q8 payload routes were "
                "unnecessary",
                report,
            )
            self.assertIn("Pair-1 traffic retained direct peer transport", report)
            self.assertIn("no pair-0 indexer dispatch was durably logged", report)
            self.assertNotIn("those direct indexer routes", report)

    def test_combined_host_bounce_underloaded_completion_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-host-bounce"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-host-bounce\n"
                "status=inconclusive-underloaded\nexit_status=128\n"
                "last_phase=decode\nlast_event=frontier-complete\n"
            )
            (production / f"{stem}.csv").write_text(
                "ctx_tokens,prefill_tps,gen_tps\n32768,310.00,15.00\n"
            )
            (production / f"{stem}.log").write_text("")
            write_healthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("inconclusive-underloaded", report)
            self.assertIn("below the required 500 prefill tok/s", report)
            self.assertIn("cannot show", report)
            self.assertIn("Pair-1 traffic remained direct", report)

    def test_combined_host_bounce_verified_full_load_pass_is_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-host-bounce"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-host-bounce\nstatus=passed\nexit_status=0\n"
                "last_phase=decode\nlast_event=frontier-complete\n"
            )
            (production / f"{stem}.csv").write_text(
                "ctx_tokens,prefill_tps,gen_tps\n32768,575.00,15.00\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA q8 partner transfer audit event=begin "
                "home_tier=0 partner_tier=2 calls=128 bytes=4096 "
                "activation_bytes=32 result_bytes=32 tokens=512 "
                "transport=host-bounce serialized=no\n"
                "ds4: CUDA q8 partner transfer audit event=complete "
                "home_tier=0 partner_tier=2 calls=128 bytes=4096\n"
                "ds4: CUDA prefill attention cache mirror transport=host-bounce "
                "home_tier=0 partner_tier=2 class=raw event=complete bytes=4096\n"
                "ds4: CUDA prefill attention cache mirror transport=host-bounce "
                "home_tier=0 partner_tier=2 class=attn-comp event=complete "
                "bytes=4096\n"
                "ds4: CUDA prefill attention cache mirror transport=host-bounce "
                "home_tier=0 partner_tier=2 class=index event=complete bytes=4096\n"
                "ds4: CUDA prefill attention host-bounce checkpoint "
                "event=complete kind=indexed layer=27 pos=512 tokens=512 "
                "home=0 partner=2\n"
                "ds4: CUDA prefill attention row audit dispatch=split "
                "kind=indexed layer=27 pos=512 tokens=512 home=0 partner=2 "
                "query_copy_stream=source gather_copy_stream=source "
                "query_copy_transport=host-bounce "
                "gather_copy_transport=host-bounce "
                "topk_copy_transport=partner-local\n"
                "ds4: CUDA decode indexer row audit event=begin layer=27 "
                "home_tier=0 partner_tier=2 n_comp=7936 "
                "transfer_bytes=50176\n"
            )
            write_healthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("durable measured completion evidence", report)
            self.assertIn("indexer transport retained its direct-peer configuration", report)
            self.assertIn("durably logged 1 begun dispatch", report)
            self.assertIn("not by itself a", report)

    def test_combined_pass_markers_below_load_floor_are_inconclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-host-bounce"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-host-bounce\nstatus=passed\nexit_status=0\n"
                "last_phase=decode\nlast_event=frontier-complete\n"
            )
            (production / f"{stem}.csv").write_text(
                "ctx_tokens,prefill_tps,gen_tps\n32768,499.00,15.00\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA q8 partner transfer audit event=begin "
                "home_tier=0 partner_tier=2 calls=128 bytes=4096 "
                "activation_bytes=32 result_bytes=32 tokens=512 "
                "transport=host-bounce serialized=no\n"
                "ds4: CUDA q8 partner transfer audit event=complete "
                "home_tier=0 partner_tier=2 calls=128 bytes=4096\n"
                "ds4: CUDA prefill attention cache mirror transport=host-bounce "
                "home_tier=0 partner_tier=2 class=raw event=complete bytes=4096\n"
                "ds4: CUDA prefill attention cache mirror transport=host-bounce "
                "home_tier=0 partner_tier=2 class=attn-comp event=complete "
                "bytes=4096\n"
                "ds4: CUDA prefill attention cache mirror transport=host-bounce "
                "home_tier=0 partner_tier=2 class=index event=complete bytes=4096\n"
                "ds4: CUDA prefill attention host-bounce checkpoint "
                "event=complete kind=indexed layer=27 pos=512 tokens=512 "
                "home=0 partner=2\n"
                "ds4: CUDA prefill attention row audit dispatch=split "
                "kind=indexed layer=27 pos=512 tokens=512 home=0 partner=2 "
                "query_copy_stream=source gather_copy_stream=source "
                "query_copy_transport=host-bounce "
                "gather_copy_transport=host-bounce "
                "topk_copy_transport=partner-local\n"
            )
            write_healthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("recorded as passed", report)
            self.assertIn("required >=500 prefill tok/s load", report)
            self.assertIn("Treat it as inconclusive", report)
            self.assertNotIn("This implicates one of the removed", report)

    def test_combined_host_bounce_pair1_only_loss_is_inconclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-host-bounce"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-host-bounce\nstatus=failed-device-loss\n"
                "exit_status=1\nlast_phase=measured-prefill\n"
                "last_event=chunk-start\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA q8 partner transfer audit event=begin "
                "home_tier=0 partner_tier=2 calls=128 bytes=4096 "
                "activation_bytes=32 result_bytes=32 tokens=512 "
                "transport=host-bounce serialized=no\n"
                "ds4: CUDA q8 partner transfer audit event=complete "
                "home_tier=0 partner_tier=2 calls=128 bytes=4096\n"
                "ds4: CUDA prefill attention cache mirror transport=host-bounce "
                "home_tier=0 partner_tier=2 class=raw event=complete bytes=4096\n"
                "ds4: CUDA prefill attention cache mirror transport=host-bounce "
                "home_tier=0 partner_tier=2 class=attn-comp event=complete "
                "bytes=4096\n"
                "ds4: CUDA prefill attention cache mirror transport=host-bounce "
                "home_tier=0 partner_tier=2 class=index event=complete bytes=4096\n"
                "ds4: CUDA prefill attention host-bounce checkpoint "
                "event=complete kind=mixed layer=2 pos=512 tokens=512 "
                "home=0 partner=2\n"
                "ds4: CUDA prefill attention row audit dispatch=split "
                "kind=mixed layer=2 pos=512 tokens=512 home=0 partner=2 "
                "query_copy_stream=source gather_copy_stream=source "
                "query_copy_transport=host-bounce "
                "gather_copy_transport=host-bounce topk_copy_transport=none\n"
            )
            write_unhealthy_post(root, stem, device=2)
            write_lost_watch(root, stem, "2@00000000:81:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("does not identify physical GPU 0 or 1", report)
            self.assertIn("real failure evidence", report)
            self.assertIn("cannot show", report)

    def test_q8_phase_audit_compute_sync_failure_is_partner_side(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-phase-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-phase-audit\nstatus=failed\n"
                "exit_status=1\nlast_phase=untimed-warmup\n"
            )
            (production / f"{stem}.log").write_text(
                q8_phase_marker(41, "begin", "activation-prepare")
                + q8_phase_marker(41, "activation-complete", "activation-copy")
                + q8_phase_marker(
                    41, "pre-compute-sync-begin", "pre-compute-sync"
                )
                + q8_phase_marker(
                    41, "pre-compute-complete", "pre-compute-sync"
                )
                + q8_phase_marker(41, "compute-submitted", "compute")
                + q8_phase_marker(
                    41, "compute-sync-failed", "compute-sync",
                    cuda_error="unspecified launch failure",
                )
                + q8_phase_marker(
                    41, "home-restored-after-failure", "recovery"
                )
            )
            write_unhealthy_post(root, stem)
            write_lost_watch(root, stem, "0@00000000:02:00.0,1@00000000:03:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertIn(
                "sequence=41 event=compute-sync-failed stage=compute-sync",
                row["q8_phase_audit_last_checkpoint"],
            )
            self.assertEqual(
                row["q8_phase_audit_last_checkpoint"],
                row["q8_phase_audit_first_failure"],
            )
            self.assertEqual(
                row["q8_phase_audit_classification"],
                "target-compute-or-concurrent-partner-work",
            )
            self.assertIn(
                "cuda_error=unspecified launch failure",
                row["q8_phase_audit_first_failure"],
            )
            report = (root / "summary.md").read_text()
            self.assertIn(
                "audited compute or partner work submitted concurrently", report
            )
            self.assertIn("not in the result D2H host-bounce copy", report)
            self.assertIn("no pair-0 indexer dispatch was durably logged", report)

    def test_q8_phase_audit_compute_complete_then_d2h_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-phase-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-phase-audit\nstatus=failed\n"
                "exit_status=1\nlast_phase=untimed-warmup\n"
            )
            (production / f"{stem}.log").write_text(
                q8_phase_marker(42, "begin", "activation-prepare")
                + q8_phase_marker(42, "activation-complete", "activation-copy")
                + q8_phase_marker(
                    42, "pre-compute-sync-begin", "pre-compute-sync"
                )
                + q8_phase_marker(
                    42, "pre-compute-complete", "pre-compute-sync"
                )
                + q8_phase_marker(42, "compute-submitted", "compute")
                + q8_phase_marker(42, "compute-complete", "compute-sync")
                + "ds4: CUDA default-stream bounce d2h failed: "
                  "unspecified launch failure\n"
                + q8_phase_marker(
                    42, "result-copy-failed", "result-gather",
                    cuda_error="unspecified launch failure",
                )
                + "ds4: CUDA q8 partner host-bounce failure "
                  "class=result-gather stage=copy source_tier=2 "
                  "source_device=1 destination_tier=0 destination_device=0 "
                  "bytes=8388608 tokens=512 in=8192 out=4096 "
                  "label=attn_output_b\n"
            )
            write_unhealthy_post(root, stem)
            write_lost_watch(root, stem, "0@00000000:02:00.0,1@00000000:03:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertIn(
                "sequence=42 event=result-copy-failed stage=result-gather",
                row["q8_phase_audit_last_checkpoint"],
            )
            self.assertEqual(
                row["q8_phase_audit_last_checkpoint"],
                row["q8_phase_audit_first_failure"],
            )
            self.assertIn(
                "cuda_phase=d2h", row["q8_phase_audit_first_failure"]
            )
            self.assertEqual(
                row["q8_phase_audit_classification"],
                "result-d2h-pcie-host-bounce-transfer",
            )
            report = (root / "summary.md").read_text()
            self.assertIn("completed the partner compute synchronization", report)
            self.assertIn("partner-to-host PCIe leg", report)
            self.assertIn("rather than to the audited partner compute", report)

    def test_q8_phase_audit_without_same_sequence_compute_complete_is_inconclusive(
            self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-phase-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-phase-audit\nstatus=failed\n"
                "exit_status=1\nlast_phase=untimed-warmup\n"
            )
            (production / f"{stem}.log").write_text(
                q8_phase_marker(8, "compute-complete", "compute-sync")
                + q8_phase_marker(
                    9, "result-copy-failed", "result-gather",
                    cuda_error="unspecified launch failure",
                )
            )
            write_unhealthy_post(root, stem)
            write_lost_watch(root, stem, "1@00000000:03:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["q8_phase_audit_classification"], "inconclusive")
            report = (root / "summary.md").read_text()
            self.assertIn(
                "without a marker sequence that separates partner compute", report
            )
            self.assertNotIn("partner-to-host PCIe leg", report)

    def test_q8_phase_audit_result_failure_requires_precompute_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-phase-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-phase-audit\nstatus=failed\n"
                "exit_status=1\nlast_phase=untimed-warmup\n"
            )
            (production / f"{stem}.log").write_text(
                q8_phase_marker(10, "compute-complete", "compute-sync")
                + "ds4: CUDA default-stream bounce d2h failed: "
                  "unspecified launch failure\n"
                + q8_phase_marker(
                    10, "result-copy-failed", "result-gather",
                    cuda_error="unspecified launch failure",
                )
            )
            write_unhealthy_post(root, stem)
            write_lost_watch(root, stem, "1@00000000:03:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["q8_phase_audit_classification"], "inconclusive")
            report = (root / "summary.md").read_text()
            self.assertNotIn("partner-to-host PCIe leg", report)

    def test_targeted_q8_audit_precompute_failure_is_earlier_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-targeted-phase-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-targeted-phase-audit\nstatus=failed\n"
                "exit_status=1\nlast_phase=untimed-warmup\n"
            )
            (production / f"{stem}.log").write_text(
                q8_phase_marker(43, "begin", "activation-prepare")
                + q8_phase_marker(43, "activation-complete", "activation-copy")
                + q8_phase_marker(
                    43, "pre-compute-sync-begin", "pre-compute-sync"
                )
                + q8_phase_marker(
                    43, "pre-compute-sync-failed", "pre-compute-sync",
                    cuda_error="unspecified launch failure",
                )
            )
            write_unhealthy_post(root, stem)
            write_lost_watch(root, stem, "1@00000000:03:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["q8_phase_audit_classification"],
                "pre-target-partner-error",
            )
            self.assertIn(
                "event=pre-compute-sync-failed stage=pre-compute-sync",
                row["q8_phase_audit_first_failure"],
            )
            report = (root / "summary.md").read_text()
            self.assertIn("surfaced before the target submission", report)
            self.assertIn("does not prove that queued work caused the fault", report)
            self.assertIn("target projection or result D2H copy", report)
            self.assertIn("tensor:blk.14.attn_output_b.weight", report)

    def test_targeted_q8_audit_compute_sync_failure_is_target_interval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-targeted-phase-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-targeted-phase-audit\nstatus=failed\n"
                "exit_status=1\nlast_phase=untimed-warmup\n"
            )
            (production / f"{stem}.log").write_text(
                q8_phase_marker(44, "begin", "activation-prepare")
                + q8_phase_marker(44, "activation-complete", "activation-copy")
                + q8_phase_marker(
                    44, "pre-compute-sync-begin", "pre-compute-sync"
                )
                + q8_phase_marker(
                    44, "pre-compute-complete", "pre-compute-sync"
                )
                + q8_phase_marker(44, "compute-submitted", "compute")
                + q8_phase_marker(
                    44, "compute-sync-failed", "compute-sync",
                    cuda_error="unspecified launch failure",
                )
            )
            write_unhealthy_post(root, stem)
            write_lost_watch(root, stem, "1@00000000:03:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["q8_phase_audit_classification"],
                "target-compute-or-concurrent-partner-work",
            )
            report = (root / "summary.md").read_text()
            self.assertIn("completed the pre-compute partner boundary", report)
            self.assertIn("projection or partner work submitted concurrently", report)
            self.assertIn("not in its result D2H copy", report)

    def test_targeted_q8_audit_d2h_failure_is_result_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-targeted-phase-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-targeted-phase-audit\nstatus=failed\n"
                "exit_status=1\nlast_phase=untimed-warmup\n"
            )
            (production / f"{stem}.log").write_text(
                q8_phase_marker(45, "begin", "activation-prepare")
                + q8_phase_marker(45, "activation-complete", "activation-copy")
                + q8_phase_marker(
                    45, "pre-compute-sync-begin", "pre-compute-sync"
                )
                + q8_phase_marker(
                    45, "pre-compute-complete", "pre-compute-sync"
                )
                + q8_phase_marker(45, "compute-submitted", "compute")
                + q8_phase_marker(45, "compute-complete", "compute-sync")
                + "ds4: CUDA default-stream bounce d2h failed: "
                  "unspecified launch failure\n"
                + q8_phase_marker(
                    45, "result-copy-failed", "result-gather",
                    cuda_error="unspecified launch failure",
                )
            )
            write_unhealthy_post(root, stem)
            write_lost_watch(root, stem, "1@00000000:03:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["q8_phase_audit_classification"],
                "result-d2h-pcie-host-bounce-transfer",
            )
            self.assertIn("cuda_phase=d2h", row["q8_phase_audit_first_failure"])
            report = (root / "summary.md").read_text()
            self.assertIn("completed both pre- and post-compute", report)
            self.assertIn("partner-to-host transfer call", report)
            self.assertIn("unresolved physical causes", report)

    def test_targeted_q8_audit_completed_target_then_later_loss(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-targeted-phase-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-targeted-phase-audit\nstatus=failed\n"
                "exit_status=1\nlast_phase=measured-prefill\n"
            )
            (production / f"{stem}.log").write_text(
                q8_phase_marker(46, "begin", "activation-prepare")
                + q8_phase_marker(46, "activation-complete", "activation-copy")
                + q8_phase_marker(
                    46, "pre-compute-sync-begin", "pre-compute-sync"
                )
                + q8_phase_marker(
                    46, "pre-compute-complete", "pre-compute-sync"
                )
                + q8_phase_marker(46, "compute-submitted", "compute")
                + q8_phase_marker(46, "compute-complete", "compute-sync")
                + q8_phase_marker(46, "result-complete", "result-gather")
            )
            write_unhealthy_post(root, stem)
            write_lost_watch(root, stem, "1@00000000:03:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("outside that recorded target sequence", report)
            self.assertIn("delayed physical effect remains possible", report)

    def test_targeted_q8_audit_passes_at_validated_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-targeted-phase-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-targeted-phase-audit\nstatus=passed\n"
                "exit_status=0\nlast_phase=decode\n"
            )
            (production / f"{stem}.csv").write_text(
                "ctx_tokens,prefill_tps,gen_tps\n32768,575.00,15.00\n"
            )
            (production / f"{stem}.log").write_text(
                q8_phase_marker(47, "begin", "activation-prepare")
                + q8_phase_marker(47, "activation-complete", "activation-copy")
                + q8_phase_marker(
                    47, "pre-compute-sync-begin", "pre-compute-sync"
                )
                + q8_phase_marker(
                    47, "pre-compute-complete", "pre-compute-sync"
                )
                + q8_phase_marker(47, "compute-submitted", "compute")
                + q8_phase_marker(47, "compute-complete", "compute-sync")
                + q8_phase_marker(47, "result-complete", "result-gather")
            )
            write_healthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("validated production-load floor", report)
            self.assertIn("does not by itself prove", report)
            self.assertIn("establish a production mitigation", report)

    def test_q8_l14_l15_validator_accepts_paired_complete_chains(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = pathlib.Path(temporary) / "window.log"
            l15 = "tensor:blk.15.attn_output_b.weight"
            log.write_text(
                q8_phase_chain(1)
                + q8_phase_chain(2)
                + q8_phase_chain(3, binding_label=l15,
                                 weight_offset=143723876608)
                + q8_phase_chain(4, binding_label=l15,
                                 weight_offset=143723876608)
            )
            subprocess.run([
                sys.executable, str(SUMMARIZER),
                "--validate-q8-l14-l15-log", str(log), "2",
            ], check=True)

    def test_q8_l14_l15_validator_rejects_cross_product_tuple(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = pathlib.Path(temporary) / "window.log"
            l15 = "tensor:blk.15.attn_output_b.weight"
            log.write_text(
                q8_phase_chain(1)
                + q8_phase_chain(2, binding_label=l15,
                                 weight_offset=143571266304)
            )
            result = subprocess.run([
                sys.executable, str(SUMMARIZER),
                "--validate-q8-l14-l15-log", str(log), "1",
            ], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("wrong-target-tuple", result.stderr)

    def test_q8_l14_l15_validator_rejects_layer15_first(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = pathlib.Path(temporary) / "window.log"
            l15 = "tensor:blk.15.attn_output_b.weight"
            log.write_text(
                q8_phase_chain(1, binding_label=l15,
                               weight_offset=143723876608)
                + q8_phase_chain(2)
            )
            result = subprocess.run([
                sys.executable, str(SUMMARIZER),
                "--validate-q8-l14-l15-log", str(log), "1",
            ], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("layer15-preceded-layer14", result.stderr)

    def test_q8_l14_l15_precompute_failure_is_between_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-l14-l15-phase-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-l14-l15-phase-audit\nstatus=failed\n"
                "exit_status=1\nlast_phase=measured-prefill\n"
            )
            l15 = "tensor:blk.15.attn_output_b.weight"
            (production / f"{stem}.log").write_text(
                q8_phase_chain(1)
                + q8_phase_marker(
                    2, "begin", "activation-prepare",
                    binding_label=l15, weight_offset=143723876608,
                )
                + q8_phase_marker(
                    2, "activation-complete", "activation-copy",
                    binding_label=l15, weight_offset=143723876608,
                )
                + q8_phase_marker(
                    2, "pre-compute-sync-begin", "pre-compute-sync",
                    binding_label=l15, weight_offset=143723876608,
                )
                + q8_phase_marker(
                    2, "pre-compute-sync-failed", "pre-compute-sync",
                    binding_label=l15, weight_offset=143723876608,
                    cuda_error="unspecified launch failure",
                )
            )
            write_unhealthy_post(root, stem)
            write_lost_watch(root, stem, "1@00000000:03:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["q8_window_classification"],
                "between-layer14-result-and-layer15-compute",
            )
            self.assertEqual(row["q8_window_l14_complete"], "1")
            self.assertEqual(row["q8_window_l15_complete"], "0")
            report = (root / "summary.md").read_text()
            self.assertIn("after completed layer-14 result handling", report)
            self.assertIn("before layer-15 compute was submitted", report)

    def test_q8_l14_l15_d2h_failure_is_layer15_result_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-l14-l15-phase-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-l14-l15-phase-audit\nstatus=failed\n"
                "exit_status=1\nlast_phase=measured-prefill\n"
            )
            l15 = "tensor:blk.15.attn_output_b.weight"
            prefix = q8_phase_chain(1)
            for event, stage in (
                    ("begin", "activation-prepare"),
                    ("activation-complete", "activation-copy"),
                    ("pre-compute-sync-begin", "pre-compute-sync"),
                    ("pre-compute-complete", "pre-compute-sync"),
                    ("compute-submitted", "compute"),
                    ("compute-complete", "compute-sync")):
                prefix += q8_phase_marker(
                    2, event, stage, binding_label=l15,
                    weight_offset=143723876608,
                )
            (production / f"{stem}.log").write_text(
                prefix
                + "ds4: CUDA default-stream bounce d2h failed: "
                  "unspecified launch failure\n"
                + q8_phase_marker(
                    2, "result-copy-failed", "result-gather",
                    binding_label=l15, weight_offset=143723876608,
                    cuda_error="unspecified launch failure",
                )
            )
            write_unhealthy_post(root, stem)
            write_lost_watch(root, stem, "1@00000000:03:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["q8_window_classification"],
                "layer15-result-d2h-pcie-host-bounce-transfer",
            )
            report = (root / "summary.md").read_text()
            self.assertIn("layer-15 result-copy helper failed", report)
            self.assertIn("not proof that PCIe transfer caused", report)

    def test_q8_l14_l15_both_complete_then_later_loss(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-l14-l15-phase-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-l14-l15-phase-audit\nstatus=failed\n"
                "exit_status=1\nlast_phase=measured-prefill\n"
            )
            l15 = "tensor:blk.15.attn_output_b.weight"
            (production / f"{stem}.log").write_text(
                q8_phase_chain(1)
                + q8_phase_chain(2, binding_label=l15,
                                 weight_offset=143723876608)
                + "ds4: gpu layer 16 attention batch encode failed\n"
            )
            write_unhealthy_post(root, stem)
            write_lost_watch(root, stem, "1@00000000:03:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["q8_window_classification"], "both-targets-complete")
            self.assertIn(
                "binding_label=tensor:blk.15.attn_output_b.weight",
                row["q8_window_last_complete"],
            )
            report = (root / "summary.md").read_text()
            self.assertIn("moved downstream of the instrumented window", report)
            self.assertIn("cumulative workload/overlap trigger", report)
            table_lines = [
                line for line in report.splitlines() if line.startswith("|")
            ]
            self.assertEqual(len({line.count("|") for line in table_lines}), 1)

    def test_q8_l14_l15_warmup_completion_does_not_classify_measured_loss_as_downstream(
            self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-l14-l15-phase-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-l14-l15-phase-audit\nstatus=failed\n"
                "exit_status=1\nlast_phase=measured-prefill\n"
            )
            l15 = "tensor:blk.15.attn_output_b.weight"
            (production / f"{stem}.log").write_text(
                "ds4-bench: starting untimed CUDA warm-up frontier 512\n"
                + q8_phase_chain(1)
                + q8_phase_chain(
                    2, binding_label=l15, weight_offset=143723876608,
                )
                + "ds4-bench: completed untimed CUDA warm-up frontier 512\n"
                + "ds4: prefill fault breadcrumb event=chunk-begin "
                  "range_start=0 range_tokens=32768 chunk_start=0 "
                  "chunk_tokens=2048\n"
                + "ds4: CUDA default-stream bounce d2h failed: "
                  "unspecified launch failure\n"
                + "ds4: CUDA q8 partner host-bounce failure "
                  "class=result-gather stage=copy source_tier=2 "
                  "source_device=1 destination_tier=0 destination_device=0 "
                  "bytes=8388608 tokens=512 in=8192 out=4096 "
                  "binding_label=tensor:blk.12.attn_output_b.weight "
                  "passed_label=attn_output_b weight_offset=143236281600\n"
                + "ds4: gpu layer 12 attention batch encode failed\n"
            )
            write_unhealthy_post(root, stem)
            write_lost_watch(root, stem, "0@00000000:02:00.0,1@00000000:03:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["q8_window_classification"],
                "measured-target-window-not-reached",
            )
            self.assertEqual(row["q8_window_l14_complete"], "1")
            self.assertEqual(row["q8_window_l15_complete"], "1")
            self.assertIn("run_phase=warmup", row["q8_window_last_complete"])
            self.assertEqual(
                row["first_gpu_layer_failure"],
                "layer=12 operation=attention-batch-encode",
            )
            report = (root / "summary.md").read_text()
            self.assertIn("completed only during the 512-token warmup", report)
            self.assertIn("earlier summary's claim", report)
            self.assertIn("layer=12 operation=attention-batch-encode", report)
            self.assertNotIn(
                "The first observed failure moved downstream of the "
                "instrumented window",
                report,
            )
            table_lines = [
                line for line in report.splitlines() if line.startswith("|")
            ]
            self.assertEqual(len({line.count("|") for line in table_lines}), 1)

    def test_q8_l12_precompute_failure_excludes_target_compute(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-l12-phase-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-l12-phase-audit\nstatus=failed\n"
                "exit_status=1\nlast_phase=measured-prefill\n"
            )
            l12 = "tensor:blk.12.attn_output_b.weight"
            (production / f"{stem}.log").write_text(
                q8_l12_measured_selection_prefix()
                + q8_phase_marker(
                    1, "begin", "activation-prepare",
                    binding_label=l12, weight_offset=143236281600,
                )
                + q8_phase_marker(
                    1, "activation-complete", "activation-copy",
                    binding_label=l12, weight_offset=143236281600,
                )
                + q8_phase_marker(
                    1, "pre-compute-sync-begin", "pre-compute-sync",
                    binding_label=l12, weight_offset=143236281600,
                )
                + q8_phase_marker(
                    1, "pre-compute-sync-failed", "pre-compute-sync",
                    binding_label=l12, weight_offset=143236281600,
                    cuda_error="unspecified launch failure",
                )
            )
            write_unhealthy_post(root, stem)
            write_lost_watch(root, stem, "1@00000000:03:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["q8_phase_audit_classification"],
                "pre-target-partner-error",
            )
            self.assertEqual(
                row["q8_phase_audit_occurrence_mapping"], "verified"
            )
            report = (root / "summary.md").read_text()
            self.assertIn("already-pending CUDA error", report)
            self.assertIn("excludes layer-12 Q8 compute", report)
            self.assertIn("occurrence 2", report)

    def test_q8_l12_result_d2h_failure_follows_clean_compute(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-l12-phase-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-l12-phase-audit\nstatus=failed\n"
                "exit_status=1\nlast_phase=measured-prefill\n"
            )
            l12 = "tensor:blk.12.attn_output_b.weight"
            log = q8_l12_measured_selection_prefix()
            for event, stage in (
                    ("begin", "activation-prepare"),
                    ("activation-complete", "activation-copy"),
                    ("pre-compute-sync-begin", "pre-compute-sync"),
                    ("pre-compute-complete", "pre-compute-sync"),
                    ("compute-submitted", "compute"),
                    ("compute-complete", "compute-sync")):
                log += q8_phase_marker(
                    1, event, stage,
                    binding_label=l12, weight_offset=143236281600,
                )
            log += (
                "ds4: CUDA default-stream bounce d2h failed: "
                "unspecified launch failure\n"
                + q8_phase_marker(
                    1, "result-copy-failed", "result-gather",
                    binding_label=l12, weight_offset=143236281600,
                    cuda_error="unspecified launch failure",
                )
            )
            (production / f"{stem}.log").write_text(log)
            write_unhealthy_post(root, stem)
            write_lost_watch(root, stem, "1@00000000:03:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["q8_phase_audit_classification"],
                "result-d2h-pcie-host-bounce-transfer",
            )
            self.assertEqual(
                row["q8_phase_audit_occurrence_mapping"], "verified"
            )
            report = (root / "summary.md").read_text()
            self.assertIn("compute completed behind a clean partner", report)
            self.assertIn("does not prove that the transfer", report)

    def test_q8_l12_loss_without_occurrence_records_is_not_attributed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-l12-phase-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-l12-phase-audit\nstatus=failed\n"
                "exit_status=1\nlast_phase=measured-prefill\n"
            )
            l12 = "tensor:blk.12.attn_output_b.weight"
            (production / f"{stem}.log").write_text(
                q8_phase_marker(
                    1, "begin", "activation-prepare",
                    binding_label=l12, weight_offset=143236281600,
                )
                + q8_phase_marker(
                    1, "activation-complete", "activation-copy",
                    binding_label=l12, weight_offset=143236281600,
                )
                + q8_phase_marker(
                    1, "pre-compute-sync-begin", "pre-compute-sync",
                    binding_label=l12, weight_offset=143236281600,
                )
                + q8_phase_marker(
                    1, "pre-compute-sync-failed", "pre-compute-sync",
                    binding_label=l12, weight_offset=143236281600,
                    cuda_error="unspecified launch failure",
                )
            )
            write_unhealthy_post(root, stem)
            write_lost_watch(root, stem, "1@00000000:03:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["q8_phase_audit_occurrence_mapping"], "not-observed"
            )
            report = (root / "summary.md").read_text()
            self.assertIn("does not prove both the warmup skip", report)
            self.assertIn("Do not assign the loss to that target", report)
            self.assertNotIn("excludes layer-12 Q8 compute", report)

    def test_q8_l12_duplicate_selection_records_invalidate_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-l12-phase-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-l12-phase-audit\nstatus=failed\n"
                "exit_status=1\nlast_phase=measured-prefill\n"
            )
            l12 = "tensor:blk.12.attn_output_b.weight"
            (production / f"{stem}.log").write_text(
                q8_l12_measured_selection_prefix()
                + "ds4: CUDA q8 partner phase audit selected occurrence=2 "
                  "sequence=1 binding_label=" + l12
                + " weight_offset=143236281600\n"
                + q8_phase_marker(
                    1, "begin", "activation-prepare",
                    binding_label=l12, weight_offset=143236281600,
                )
            )
            write_unhealthy_post(root, stem)
            write_lost_watch(root, stem, "1@00000000:03:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["q8_phase_audit_occurrence_mapping"],
                "invalid-record-count:3",
            )
            report = (root / "summary.md").read_text()
            self.assertIn("does not prove both the warmup skip", report)
            self.assertNotIn("first measured layer-12 target reached", report)

    def test_q8_l12_pass_without_occurrence_mapping_is_inconsistent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-l12-phase-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-l12-phase-audit\nstatus=passed\n"
                "exit_status=0\nlast_phase=decode\nlast_event=frontier-complete\n"
            )
            (production / f"{stem}.csv").write_text(
                "ctx_tokens,prefill_tps,gen_tps\n32768,575.00,15.00\n"
            )
            (production / f"{stem}.log").write_text("")
            write_healthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("reports completion", report)
            self.assertIn("validation-inconsistent", report)
            self.assertNotIn("completed at full production load", report)

    def test_q8_l14_l15_later_partial_layer14_is_not_downstream(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-l14-l15-phase-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-l14-l15-phase-audit\nstatus=failed\n"
                "exit_status=1\nlast_phase=measured-prefill\n"
            )
            l15 = "tensor:blk.15.attn_output_b.weight"
            (production / f"{stem}.log").write_text(
                q8_phase_chain(1)
                + q8_phase_chain(2, binding_label=l15,
                                 weight_offset=143723876608)
                + q8_phase_marker(3, "begin", "activation-prepare")
                + q8_phase_marker(3, "activation-complete", "activation-copy")
                + q8_phase_marker(
                    3, "pre-compute-sync-begin", "pre-compute-sync"
                )
            )
            write_unhealthy_post(root, stem)
            write_lost_watch(root, stem, "1@00000000:03:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["q8_window_classification"],
                "layer14-partial-after-prior-completion",
            )
            self.assertIn(
                "binding_label=tensor:blk.15.attn_output_b.weight",
                row["q8_window_last_complete"],
            )
            self.assertIn(
                "binding_label=tensor:blk.14.attn_output_b.weight",
                row["q8_phase_audit_last_checkpoint"],
            )
            report = (root / "summary.md").read_text()
            self.assertIn("inside a later layer-14 chain", report)
            self.assertIn("did not demonstrably move downstream", report)
            self.assertNotIn("moved downstream of the instrumented window", report)

    def test_q8_l14_l15_later_complete_unpaired_layer14_is_not_downstream(
            self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-l14-l15-phase-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-l14-l15-phase-audit\nstatus=failed\n"
                "exit_status=1\nlast_phase=measured-prefill\n"
            )
            l15 = "tensor:blk.15.attn_output_b.weight"
            (production / f"{stem}.log").write_text(
                q8_phase_chain(1)
                + q8_phase_chain(2, binding_label=l15,
                                 weight_offset=143723876608)
                + q8_phase_chain(3)
            )
            write_unhealthy_post(root, stem)
            write_lost_watch(root, stem, "1@00000000:03:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["q8_window_classification"],
                "layer14-complete-layer15-not-complete",
            )
            self.assertEqual(row["q8_window_l14_complete"], "2")
            self.assertEqual(row["q8_window_l15_complete"], "1")
            report = (root / "summary.md").read_text()
            self.assertIn("last durable completed target is layer 14", report)
            self.assertNotIn("moved downstream of the instrumented window", report)

    def test_q8_l14_l15_interruption_without_loss_is_not_causal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-l14-l15-phase-audit"
            (production / f"{stem}.started").write_text(
                "variant=attention-q8-l14-l15-phase-audit\nrepeat=1\n"
            )
            l15 = "tensor:blk.15.attn_output_b.weight"
            (production / f"{stem}.log").write_text(
                q8_phase_chain(1)
                + q8_phase_chain(2, binding_label=l15,
                                 weight_offset=143723876608)
            )
            (production / f"{stem}-progress.csv").write_text(
                "realtime_sec,realtime_nsec,phase,event,current,total\n"
                "1,0,measured-prefill,chunk-start,512,32768\n"
            )

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["status"], "interrupted-no-result")
            self.assertEqual(row["q8_window_classification"], "both-targets-complete")
            report = (root / "summary.md").read_text()
            self.assertIn("without a watcher record", report)
            self.assertIn("not causal GPU-loss evidence", report)
            self.assertNotIn("moved downstream of the instrumented window", report)

    def test_q8_l14_l15_repeat_cannot_borrow_loss_corroboration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            l15 = "tensor:blk.15.attn_output_b.weight"

            interrupted_stem = "r1-s1-attention-q8-l14-l15-phase-audit"
            (production / f"{interrupted_stem}.started").write_text(
                "variant=attention-q8-l14-l15-phase-audit\nrepeat=1\n"
            )
            (production / f"{interrupted_stem}.log").write_text(
                q8_phase_chain(1)
                + q8_phase_chain(2, binding_label=l15,
                                 weight_offset=143723876608)
            )

            lost_stem = "r2-s2-attention-q8-l14-l15-phase-audit"
            (production / f"{lost_stem}.started").write_text(
                "variant=attention-q8-l14-l15-phase-audit\nrepeat=2\n"
            )
            (production / f"{lost_stem}.log").write_text("")
            write_unhealthy_post(root, lost_stem)
            write_lost_watch(root, lost_stem, "1@00000000:03:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("ended without a classified target failure", report)
            self.assertNotIn("moved downstream of the instrumented window", report)

    def test_q8_l14_l15_partial_layer15_preserves_last_complete_layer14(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-l14-l15-phase-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-l14-l15-phase-audit\nstatus=failed\n"
                "exit_status=1\nlast_phase=measured-prefill\n"
            )
            l15 = "tensor:blk.15.attn_output_b.weight"
            (production / f"{stem}.log").write_text(
                q8_phase_chain(1)
                + q8_phase_marker(
                    2, "begin", "activation-prepare",
                    binding_label=l15, weight_offset=143723876608,
                )
            )
            write_unhealthy_post(root, stem)
            write_lost_watch(root, stem, "1@00000000:03:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["q8_window_classification"],
                "layer14-complete-layer15-not-complete",
            )
            self.assertIn(
                "binding_label=tensor:blk.14.attn_output_b.weight",
                row["q8_window_last_complete"],
            )
            self.assertIn(
                "binding_label=tensor:blk.15.attn_output_b.weight",
                row["q8_phase_audit_last_checkpoint"],
            )
            report = (root / "summary.md").read_text()
            self.assertIn("last durable completed target is layer 14", report)

    def test_q8_phase_audit_variant_sorts_after_combined_host_bounce(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            for slot, variant in enumerate((
                    "attention-q8-l12-phase-audit",
                    "attention-q8-l14-l15-phase-audit",
                    "attention-q8-targeted-phase-audit",
                    "attention-q8-phase-audit",
                    "attention-q8-async-completion",
                    "attention-q8-host-bounce"), 1):
                stem = f"r1-s{slot}-{variant}"
                (production / f"{stem}.result").write_text(
                    f"variant={variant}\nstatus=validation-failed\nexit_status=127\n"
                )
                (production / f"{stem}.log").write_text("")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                [row["variant"] for row in rows],
                [
                    "attention-q8-host-bounce",
                    "attention-q8-async-completion",
                    "attention-q8-phase-audit",
                    "attention-q8-targeted-phase-audit",
                    "attention-q8-l14-l15-phase-audit",
                    "attention-q8-l12-phase-audit",
                ],
            )

    def test_invalid_outcome_outranks_underloaded_across_repeats(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            for repeat, status in ((1, "inconclusive-underloaded"),
                                   (2, "validation-failed")):
                stem = f"r{repeat}-s{repeat}-attention-q8-host-bounce"
                (production / f"{stem}.result").write_text(
                    "variant=attention-q8-host-bounce\n"
                    f"status={status}\nexit_status=127\n"
                )
                (production / f"{stem}.log").write_text("")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("failed its production-path", report)
            self.assertNotIn("inconclusive-underloaded: survival", report)

    def test_parses_q8_activation_host_bounce_failure_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-production"
            (production / f"{stem}.result").write_text(
                "variant=production\nstatus=run-failed-unverified\n"
                "exit_status=1\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA default-stream bounce d2h failed: "
                "unspecified launch failure\n"
                "ds4: CUDA q8 partner host-bounce failure "
                "class=activation stage=copy source_tier=0 source_device=0 "
                "destination_tier=2 destination_device=1 bytes=1048576 "
                "tokens=512 in=512 out=32768 label=q8_0\n"
            )

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["host_bounce_failure_context"],
                "cuda_phase=d2h class=activation stage=copy source_tier=0 "
                "source_device=0 destination_tier=2 destination_device=1 "
                "bytes=1048576 tokens=512 in=512 out=32768 label=q8_0",
            )

    def test_parses_attention_cache_host_bounce_failure_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-production"
            (production / f"{stem}.result").write_text(
                "variant=production\nstatus=run-failed-unverified\n"
                "exit_status=1\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA default-stream bounce h2d failed: "
                "unspecified launch failure\n"
                "ds4: CUDA prefill attention host-bounce failure "
                "class=cache-raw stage=copy layer=16 row=512 "
                "source_tier=0 source_device=0 destination_tier=2 "
                "destination_device=1 bytes=1048576\n"
            )

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["host_bounce_failure_context"],
                "cuda_phase=h2d class=cache-raw stage=copy layer=16 row=512 "
                "source_tier=0 source_device=0 destination_tier=2 "
                "destination_device=1 bytes=1048576",
            )

    def test_does_not_attribute_pair1_loss_to_pair0_host_bounce(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-host-bounce"
            (production / f"{stem}.result").write_text(
                "variant=attention-host-bounce\nstatus=failed\nexit_status=1\n"
                "last_phase=measured-prefill\nlast_event=chunk-start\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA prefill attention host-bounce checkpoint "
                "event=begin kind=mixed layer=17 pos=512 tokens=512 "
                "home=0 partner=2\n"
            )
            write_unhealthy_post(root, stem, device=2)
            write_lost_watch(root, stem, "2@00000000:81:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("does not identify physical GPU 0 or 1", report)
            self.assertIn("pair 1 retained direct peer traffic", report)
            self.assertIn("real but inconclusive", report)
            self.assertNotIn("not a necessary trigger", report)

    def test_reports_all_destination_stream_arms_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            for slot, variant in enumerate((
                    "attention-query-dst", "attention-gather-dst",
                    "attention-both-dst"), 1):
                stem = f"r1-s{slot}-{variant}"
                (production / f"{stem}.result").write_text(
                    f"variant={variant}\nstatus=failed\nexit_status=1\n"
                    "last_phase=measured-prefill\nlast_event=chunk-start\n"
                )
                (production / f"{stem}.log").write_text("")
                write_unhealthy_post(root, stem)
            stem = "r1-s4-production"
            (production / f"{stem}.result").write_text(
                "variant=production\nstatus=failed\nexit_status=1\n"
                "last_phase=measured-prefill\nlast_event=chunk-start\n"
            )
            (production / f"{stem}.log").write_text("")
            write_unhealthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("all recorded failures", report)
            self.assertIn("was not sufficient to prevent the fault", report)
            self.assertIn("does not identify a physical copy engine", report)

    def test_reports_unverified_ownership_matrix_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-query-dst"
            (production / f"{stem}.started").write_text(
                "variant=attention-query-dst\n"
            )
            (production / f"{stem}.log").write_text("")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("has no verified outcome", report)
            self.assertIn("not a failed arm and is not a safe pass", report)
            self.assertIn("fresh full matrix", report)

    def test_validates_complete_q8_pre_gather_fence_with_periodic_checkpoint(
            self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = pathlib.Path(temporary) / "fence.log"
            log.write_text(
                q8_pre_gather_enabled() + q8_pre_gather_checkpointed_calls(70) +
                q8_pre_gather_summary(70)
            )
            subprocess.run([
                sys.executable, str(SUMMARIZER),
                "--validate-q8-pre-gather-fence-log", str(log),
            ], check=True)

    def test_classifies_unreturned_armed_call_for_all_device_loss_statuses(
            self) -> None:
        cases = (
            ("failed", "failed-device-loss", True),
            ("interrupted-prior-run", "interrupted-prior-run-device-loss", True),
            ("started-only", "interrupted-no-result-device-loss", False),
        )
        for source_status, expected_status, has_result in cases:
            with self.subTest(status=expected_status), \
                    tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                production = root / "production"
                production.mkdir()
                stem = "r1-s1-attention-q8-pre-gather-fence"
                if has_result:
                    (production / f"{stem}.result").write_text(
                        "variant=attention-q8-pre-gather-fence\n"
                        f"status={source_status}\nexit_status=124\n"
                    )
                else:
                    (production / f"{stem}.started").write_text(
                        "variant=attention-q8-pre-gather-fence\nrepeat=1\n"
                    )
                (production / f"{stem}.log").write_text(
                    q8_pre_gather_enabled() +
                    q8_pre_gather_checkpointed_calls(64) +
                    q8_pre_gather_armed(65)
                )
                write_lost_watch(root, stem, "1@00000000:03:00.0")

                subprocess.run(
                    [sys.executable, str(SUMMARIZER), str(root)], check=True
                )
                with (root / "summary.csv").open(newline="") as handle:
                    row = next(csv.DictReader(handle))
                self.assertEqual(row["status"], expected_status)
                self.assertEqual(
                    row["q8_pre_gather_fence_classification"],
                    "post-compute-confirmed-result-gather-return-not-observed",
                )
                self.assertEqual(row["q8_pre_gather_fence_armed_count"], "65")
                self.assertEqual(row["q8_pre_gather_fence_returned_count"], "64")
                self.assertIn(
                    "current_sequence=65",
                    row["q8_pre_gather_fence_last_armed"],
                )
                self.assertIn(
                    "current_sequence=64",
                    row["q8_pre_gather_fence_last_returned"],
                )
                report = (root / "summary.md").read_text()
                self.assertIn("no matching returned breadcrumb", report)
                self.assertIn("observation interval", report)
                self.assertIn("without proving", report)

    def test_classifies_first_armed_call_without_prior_return(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-pre-gather-fence"
            (production / f"{stem}.started").write_text(
                "variant=attention-q8-pre-gather-fence\nrepeat=1\n"
            )
            (production / f"{stem}.log").write_text(
                q8_pre_gather_enabled() + q8_pre_gather_checkpoint(1) +
                q8_pre_gather_armed(1)
            )
            write_lost_watch(root, stem, "1@00000000:03:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["q8_pre_gather_fence_classification"],
                "post-compute-confirmed-result-gather-return-not-observed",
            )
            self.assertEqual(row["q8_pre_gather_fence_returned_count"], "0")
            report = (root / "summary.md").read_text()
            self.assertIn(
                "Every earlier armed sequence, if any, has a matching successful "
                "return",
                report,
            )
            self.assertNotIn("preceding sequence returned", report)

    def test_classifies_trailing_checkpoint_before_armed_breadcrumb(self) -> None:
        cases = (
            (q8_pre_gather_checkpoint(1), "1", "0"),
            (
                q8_pre_gather_checkpointed_calls(63) +
                q8_pre_gather_checkpoint(64),
                "64",
                "63",
            ),
        )
        for log_body, checkpoint_sequence, armed_count in cases:
            with self.subTest(sequence=checkpoint_sequence), \
                    tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                production = root / "production"
                production.mkdir()
                stem = "r1-s1-attention-q8-pre-gather-fence"
                (production / f"{stem}.started").write_text(
                    "variant=attention-q8-pre-gather-fence\nrepeat=1\n"
                )
                (production / f"{stem}.log").write_text(
                    q8_pre_gather_enabled() + log_body
                )
                write_lost_watch(root, stem, "1@00000000:03:00.0")

                subprocess.run(
                    [sys.executable, str(SUMMARIZER), str(root)], check=True
                )
                with (root / "summary.csv").open(newline="") as handle:
                    row = next(csv.DictReader(handle))
                self.assertEqual(
                    row["q8_pre_gather_fence_classification"],
                    "post-compute-confirmed-before-result-gather-"
                    "armed-status-not-observed",
                )
                self.assertEqual(
                    row["q8_pre_gather_fence_armed_count"], armed_count
                )
                self.assertIn(
                    f"current_sequence={checkpoint_sequence}",
                    row["q8_pre_gather_fence_last_checkpoint"],
                )
                report = (root / "summary.md").read_text()
                self.assertIn("trailing sparse checkpoint proves", report)
                self.assertIn("whether that gather was attempted", report)

    def test_classifies_returned_final_gather_as_subsequent_locus_unresolved(
            self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-pre-gather-fence"
            (production / f"{stem}.started").write_text(
                "variant=attention-q8-pre-gather-fence\nrepeat=1\n"
            )
            (production / f"{stem}.log").write_text(
                q8_pre_gather_enabled() + q8_pre_gather_checkpointed_calls(65)
            )
            write_lost_watch(root, stem, "1@00000000:03:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["q8_pre_gather_fence_classification"],
                "last-confirmed-gather-returned-subsequent-locus-unresolved",
            )
            self.assertEqual(row["q8_pre_gather_fence_armed_count"], "65")
            self.assertEqual(row["q8_pre_gather_fence_returned_count"], "65")
            report = (root / "summary.md").read_text()
            self.assertIn("result gather returned successfully", report)
            self.assertIn("subsequent locus remains unresolved", report)

    def test_rejects_wrong_armed_returned_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-pre-gather-fence"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-pre-gather-fence\n"
                "status=failed\nexit_status=124\n"
            )
            (production / f"{stem}.log").write_text(
                q8_pre_gather_enabled() + q8_pre_gather_checkpoint(1) +
                q8_pre_gather_armed(1) + q8_pre_gather_armed(2) +
                q8_pre_gather_returned(1) + q8_pre_gather_returned(2)
            )
            write_lost_watch(root, stem, "1@00000000:03:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["q8_pre_gather_fence_classification"],
                "invalid-fence-record",
            )
            self.assertIn(
                "truncated or internally inconsistent",
                (root / "summary.md").read_text(),
            )

    def test_rejects_invalid_armed_or_returned_identity(self) -> None:
        complement = (~1) & ((1 << 64) - 1)
        cases = (
            (
                q8_pre_gather_armed(1).replace(
                    f"marker_complement={complement}", "marker_complement=0"
                ),
                q8_pre_gather_returned(1),
                "invalid-armed-record",
            ),
            (
                q8_pre_gather_armed(1).replace(
                    "partner_device=1", "partner_device=3"
                ),
                q8_pre_gather_returned(1),
                "invalid-armed-record",
            ),
            (
                q8_pre_gather_armed(1),
                q8_pre_gather_returned(1).replace(
                    "partner_device=1", "partner_device=3"
                ),
                "invalid-returned-record",
            ),
        )
        for armed, returned, expected_state in cases:
            with self.subTest(state=expected_state), \
                    tempfile.TemporaryDirectory() as temporary:
                log = pathlib.Path(temporary) / "fence.log"
                log.write_text(
                    q8_pre_gather_enabled() + q8_pre_gather_checkpoint(1) +
                    armed + returned + q8_pre_gather_summary(1)
                )
                completed = subprocess.run([
                    sys.executable, str(SUMMARIZER),
                    "--validate-q8-pre-gather-fence-log", str(log),
                ], capture_output=True, text=True)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(expected_state, completed.stderr)

    def test_rejects_checkpoint_emitted_before_prior_call_returned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = pathlib.Path(temporary) / "fence.log"
            log.write_text(
                q8_pre_gather_enabled() + q8_pre_gather_checkpoint(1) +
                q8_pre_gather_checkpoint(64) +
                q8_pre_gather_completed_calls(64) +
                q8_pre_gather_summary(64)
            )
            completed = subprocess.run([
                sys.executable, str(SUMMARIZER),
                "--validate-q8-pre-gather-fence-log", str(log),
            ], capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("checkpoint-order-mismatch", completed.stderr)

    def test_rejects_missing_first_checkpoint_in_interrupted_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-pre-gather-fence"
            (production / f"{stem}.started").write_text(
                "variant=attention-q8-pre-gather-fence\nrepeat=1\n"
            )
            (production / f"{stem}.log").write_text(
                q8_pre_gather_enabled() + q8_pre_gather_completed_calls(1) +
                q8_pre_gather_armed(2)
            )
            write_lost_watch(root, stem, "1@00000000:03:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["q8_pre_gather_fence_classification"],
                "invalid-fence-record",
            )

    def test_classifies_q8_pre_gather_stream_failure_before_d2h(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-pre-gather-fence"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-pre-gather-fence\n"
                "status=failed-device-loss\nexit_status=124\n"
            )
            (production / f"{stem}.log").write_text(
                q8_pre_gather_enabled() +
                q8_pre_gather_checkpointed_calls(64) +
                q8_pre_gather_failure(65, "sync-failed")
            )
            write_unhealthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["q8_pre_gather_fence_classification"],
                "pre-gather-stream-failure",
            )
            self.assertIn("result_d2h_attempted=no", row[
                "q8_pre_gather_fence_last_failure"
            ])
            report = (root / "summary.md").read_text()
            self.assertIn("failed before the result D2H was attempted", report)
            self.assertIn("not from attempting its result D2H", report)
            table_lines = [
                line for line in report.splitlines()
                if (line.startswith("| Variant |") or
                    line.startswith("| --- |") or
                    line.startswith("| attention-q8-pre-gather-fence |"))
            ]
            self.assertEqual(len({line.count("|") for line in table_lines}), 1)

    def test_classifies_state_invalid_sequence_gap_before_d2h(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-pre-gather-fence"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-pre-gather-fence\n"
                "status=failed-device-loss\nexit_status=124\n"
            )
            (production / f"{stem}.log").write_text(
                q8_pre_gather_enabled() +
                q8_pre_gather_checkpointed_calls(64) +
                q8_pre_gather_failure(
                    67, "state-invalid", attempted_override=65
                )
            )
            write_unhealthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["q8_pre_gather_fence_classification"],
                "pre-gather-state-failure",
            )
            self.assertIn("current_sequence=67", row[
                "q8_pre_gather_fence_last_failure"
            ])
            self.assertIn("attempted=65", row[
                "q8_pre_gather_fence_last_failure"
            ])
            report = (root / "summary.md").read_text()
            self.assertIn("software audit-state invariant failure", report)

    def test_rejects_state_invalid_without_sequence_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-pre-gather-fence"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-pre-gather-fence\n"
                "status=failed-device-loss\nexit_status=124\n"
            )
            (production / f"{stem}.log").write_text(
                q8_pre_gather_enabled() +
                q8_pre_gather_checkpointed_calls(64) +
                q8_pre_gather_failure(
                    65, "state-invalid", attempted_override=65
                )
            )
            write_unhealthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["q8_pre_gather_fence_classification"],
                "invalid-fence-record",
            )

    def test_rejects_missing_periodic_q8_pre_gather_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = pathlib.Path(temporary) / "fence.log"
            log.write_text(
                q8_pre_gather_enabled() + q8_pre_gather_checkpoint(64) +
                q8_pre_gather_completed_calls(70) +
                q8_pre_gather_summary(70)
            )
            completed = subprocess.run([
                sys.executable, str(SUMMARIZER),
                "--validate-q8-pre-gather-fence-log", str(log),
            ], capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("checkpoint-cadence-mismatch", completed.stderr)

    def test_classifies_q8_pre_gather_marker_channel_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-pre-gather-fence"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-pre-gather-fence\n"
                "status=failed-device-loss\nexit_status=124\n"
            )
            (production / f"{stem}.log").write_text(
                q8_pre_gather_enabled() +
                q8_pre_gather_checkpointed_calls(64) +
                q8_pre_gather_failure(65, "marker-invalid")
            )
            write_unhealthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["q8_pre_gather_fence_classification"],
                "marker-channel-failure",
            )
            report = (root / "summary.md").read_text()
            self.assertIn("event synchronized successfully", report)
            self.assertIn("marker-channel integrity failure", report)

    def test_classifies_result_d2h_failure_after_confirmed_q8_compute(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-pre-gather-fence"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-pre-gather-fence\n"
                "status=failed-device-loss\nexit_status=124\n"
            )
            (production / f"{stem}.log").write_text(
                q8_pre_gather_enabled() +
                q8_pre_gather_checkpointed_calls(64) +
                q8_pre_gather_armed(65) +
                q8_pre_gather_failure(65, "result-gather-failed")
            )
            write_unhealthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["q8_pre_gather_fence_classification"],
                "post-compute-confirmed-result-d2h-failed",
            )
            self.assertEqual(
                row[
                    "q8_pre_gather_fence_result_gather_failed_after_confirmed"
                ],
                "1",
            )
            report = (root / "summary.md").read_text()
            self.assertIn("exact marker matched before result D2H", report)
            self.assertIn("D2H API was invoked", report)
            self.assertIn("does not by itself prove", report)

    def test_classifies_host_bounce_allocation_failure_before_result_d2h(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-pre-gather-fence"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-pre-gather-fence\n"
                "status=failed-device-loss\nexit_status=124\n"
            )
            (production / f"{stem}.log").write_text(
                q8_pre_gather_enabled() +
                q8_pre_gather_checkpointed_calls(64) +
                q8_pre_gather_armed(65) +
                q8_pre_gather_failure(
                    65, "result-gather-failed", d2h_attempted_override="no"
                )
            )
            write_unhealthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["q8_pre_gather_fence_classification"],
                "post-compute-confirmed-result-gather-failed-before-d2h-attempt",
            )
            self.assertIn("result_d2h_attempted=no", row[
                "q8_pre_gather_fence_last_failure"
            ])
            report = (root / "summary.md").read_text()
            self.assertIn("before D2H was attempted", report)

    def test_classifies_pre_h2d_failure_after_result_d2h_completed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-pre-gather-fence"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-pre-gather-fence\n"
                "status=failed-device-loss\nexit_status=124\n"
            )
            (production / f"{stem}.log").write_text(
                q8_pre_gather_enabled() +
                q8_pre_gather_checkpointed_calls(64) +
                q8_pre_gather_armed(65) +
                q8_pre_gather_failure(
                    65, "result-gather-failed", d2h_completed_override="yes"
                )
            )
            write_unhealthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["q8_pre_gather_fence_classification"],
                "post-compute-confirmed-result-d2h-complete-result-h2d-not-attempted",
            )
            self.assertIn("result_d2h_attempted=yes", row[
                "q8_pre_gather_fence_last_failure"
            ])
            self.assertIn("result_d2h_completed=yes", row[
                "q8_pre_gather_fence_last_failure"
            ])
            report = (root / "summary.md").read_text()
            self.assertIn("result D2H completed successfully", report)
            self.assertIn("before destination H2D was attempted", report)
            self.assertIn("does not prove", report)

    def test_classifies_result_h2d_failure_after_result_d2h_completed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-pre-gather-fence"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-pre-gather-fence\n"
                "status=failed-device-loss\nexit_status=124\n"
            )
            (production / f"{stem}.log").write_text(
                q8_pre_gather_enabled() +
                q8_pre_gather_checkpointed_calls(64) +
                q8_pre_gather_armed(65) +
                q8_pre_gather_failure(
                    65, "result-gather-failed", d2h_completed_override="yes",
                    h2d_attempted_override="yes"
                )
            )
            write_unhealthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["q8_pre_gather_fence_classification"],
                "post-compute-confirmed-result-h2d-failed",
            )
            self.assertIn("result_h2d_attempted=yes", row[
                "q8_pre_gather_fence_last_failure"
            ])
            self.assertIn("result_h2d_completed=no", row[
                "q8_pre_gather_fence_last_failure"
            ])
            report = (root / "summary.md").read_text()
            self.assertIn("Destination H2D was then invoked", report)
            self.assertIn("first observed failing result-gather API", report)

    def test_classifies_later_failure_after_result_h2d_completed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-pre-gather-fence"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-pre-gather-fence\n"
                "status=failed-device-loss\nexit_status=124\n"
            )
            (production / f"{stem}.log").write_text(
                q8_pre_gather_enabled() +
                q8_pre_gather_checkpointed_calls(64) +
                q8_pre_gather_armed(65) +
                q8_pre_gather_failure(
                    65, "result-gather-failed", d2h_completed_override="yes",
                    h2d_attempted_override="yes", h2d_completed_override="yes"
                )
            )
            write_unhealthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["q8_pre_gather_fence_classification"],
                "post-compute-confirmed-result-h2d-complete-later-gather-failure",
            )
            self.assertIn("result_h2d_completed=yes", row[
                "q8_pre_gather_fence_last_failure"
            ])
            report = (root / "summary.md").read_text()
            self.assertIn("both result D2H and destination H2D completed", report)

    def test_rejects_inconsistent_q8_pre_gather_fence_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-pre-gather-fence"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-pre-gather-fence\n"
                "status=failed-device-loss\nexit_status=124\n"
            )
            invalid = q8_pre_gather_failure(65, "sync-failed").replace(
                "result_d2h_attempted=no", "result_d2h_attempted=yes"
            )
            (production / f"{stem}.log").write_text(
                q8_pre_gather_enabled() +
                q8_pre_gather_checkpointed_calls(64) + invalid
            )
            write_unhealthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["q8_pre_gather_fence_classification"],
                "invalid-fence-record",
            )
            report = (root / "summary.md").read_text()
            self.assertIn("truncated or internally inconsistent", report)

    def test_rejects_failure_sequence_that_does_not_match_call_history(
            self) -> None:
        cases = (
            (
                q8_pre_gather_checkpointed_calls(63) +
                q8_pre_gather_failure(65, "sync-failed")
            ),
            (
                q8_pre_gather_checkpointed_calls(65) +
                q8_pre_gather_failure(65, "result-gather-failed")
            ),
        )
        for body in cases:
            with self.subTest(body=body[-100:]), \
                    tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                production = root / "production"
                production.mkdir()
                stem = "r1-s1-attention-q8-pre-gather-fence"
                (production / f"{stem}.result").write_text(
                    "variant=attention-q8-pre-gather-fence\n"
                    "status=failed-device-loss\nexit_status=124\n"
                )
                (production / f"{stem}.log").write_text(
                    q8_pre_gather_enabled() + body
                )
                write_unhealthy_post(root, stem)

                subprocess.run(
                    [sys.executable, str(SUMMARIZER), str(root)], check=True
                )
                with (root / "summary.csv").open(newline="") as handle:
                    row = next(csv.DictReader(handle))
                self.assertEqual(
                    row["q8_pre_gather_fence_classification"],
                    "invalid-fence-record",
                )

    def test_rejects_completed_result_d2h_that_was_not_attempted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-pre-gather-fence"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-pre-gather-fence\n"
                "status=failed-device-loss\nexit_status=124\n"
            )
            (production / f"{stem}.log").write_text(
                q8_pre_gather_enabled() +
                q8_pre_gather_checkpointed_calls(64) +
                q8_pre_gather_armed(65) +
                q8_pre_gather_failure(
                    65, "result-gather-failed", d2h_attempted_override="no",
                    d2h_completed_override="yes"
                )
            )
            write_unhealthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["q8_pre_gather_fence_classification"],
                "invalid-fence-record",
            )

    def test_rejects_completed_result_h2d_that_was_not_attempted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-pre-gather-fence"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-pre-gather-fence\n"
                "status=failed-device-loss\nexit_status=124\n"
            )
            (production / f"{stem}.log").write_text(
                q8_pre_gather_enabled() +
                q8_pre_gather_checkpointed_calls(64) +
                q8_pre_gather_armed(65) +
                q8_pre_gather_failure(
                    65, "result-gather-failed", d2h_completed_override="yes",
                    h2d_attempted_override="no", h2d_completed_override="yes"
                )
            )
            write_unhealthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["q8_pre_gather_fence_classification"],
                "invalid-fence-record",
            )

    def test_validates_complete_q8_async_completion_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = pathlib.Path(temporary) / "async.log"
            log.write_text(
                q8_async_enabled() + q8_async_checkpoint(128) +
                q8_async_summary(128, 128, 128)
            )
            subprocess.run([
                sys.executable, str(SUMMARIZER),
                "--validate-q8-async-completion-log", str(log),
            ], check=True)

    def test_rejects_q8_async_completion_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = pathlib.Path(temporary) / "async.log"
            log.write_text(
                q8_async_enabled() + q8_async_checkpoint(127) +
                q8_async_summary(128, 128, 127)
            )
            completed = subprocess.run([
                sys.executable, str(SUMMARIZER),
                "--validate-q8-async-completion-log", str(log),
            ], capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("count-mismatch", completed.stderr)

    def test_reports_positive_q8_async_completion_before_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-async-completion"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-async-completion\n"
                "status=failed-device-loss\nexit_status=124\n"
            )
            (production / f"{stem}.log").write_text(
                q8_async_enabled() + q8_async_checkpoint(128) +
                q8_async_failure(
                    129, marker_matches="yes", event_status="complete",
                    interpretation="post-compute-confirmed",
                )
            )
            write_unhealthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["q8_async_completion_classification"],
                "current-call-post-compute-confirmed",
            )
            report = (root / "summary.md").read_text()
            self.assertIn("crossed its post-compute", report)
            self.assertIn("default-stream boundary", report)
            self.assertIn("does not prove an electrical or software root cause", report)
            self.assertIn("does not exclude the compute load as a trigger", report)
            self.assertIn("not assumed to be the terminal process call", report)

    def test_separates_marker_observation_when_event_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-async-completion"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-async-completion\n"
                "status=failed-device-loss\nexit_status=124\n"
            )
            (production / f"{stem}.log").write_text(
                q8_async_enabled() + q8_async_checkpoint(128) +
                q8_async_failure(
                    129, marker_matches="yes",
                    event_status="unspecified-launch-failure",
                    interpretation=(
                        "post-compute-marker-positive-event-unavailable"
                    ),
                )
            )
            write_unhealthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["q8_async_completion_classification"],
                "current-call-marker-observed-event-unavailable",
            )
            report = (root / "summary.md").read_text()
            self.assertIn("positive hardware breadcrumb", report)
            self.assertIn("not formal same-stream completion proof", report)

    def test_missing_q8_async_marker_is_explicitly_inconclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-async-completion"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-async-completion\n"
                "status=failed-device-loss\nexit_status=124\n"
            )
            (production / f"{stem}.log").write_text(
                q8_async_enabled() + q8_async_checkpoint(128) +
                q8_async_failure(
                    129, marker_matches="no", event_status="not-ready",
                    interpretation="inconclusive-no-positive-marker",
                )
            )
            write_unhealthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["q8_async_completion_classification"],
                "current-call-inconclusive",
            )
            report = (root / "summary.md").read_text()
            self.assertIn("absence is inconclusive", report)
            self.assertIn("not evidence that the Q8 compute itself failed", report)

    def test_event_complete_marker_invalid_is_positive_integrity_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-async-completion"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-async-completion\n"
                "status=failed-device-loss\nexit_status=124\n"
            )
            (production / f"{stem}.log").write_text(
                q8_async_enabled() + q8_async_checkpoint(128) +
                q8_async_failure(
                    129, marker_matches="no", event_status="complete",
                    interpretation=(
                        "post-compute-event-confirmed-marker-invalid"
                    ),
                )
            )
            write_unhealthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["q8_async_completion_classification"],
                "current-call-post-compute-event-confirmed-marker-invalid",
            )
            report = (root / "summary.md").read_text()
            self.assertIn("positive post-compute boundary evidence", report)
            self.assertIn("marker-channel integrity failure", report)

    def test_rejects_truncated_positive_q8_async_failure_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-async-completion"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-async-completion\n"
                "status=failed-device-loss\nexit_status=124\n"
            )
            (production / f"{stem}.log").write_text(
                q8_async_enabled() + q8_async_checkpoint(128) +
                "ds4: CUDA q8 partner async completion failure "
                "stage=result-gather current_sequence=129 marker_sequence=129 "
                "marker_complement=18446744073709551486 marker_matches=yes "
                "event_status=complete interpretation=post-compute-confirmed\n"
            )
            write_unhealthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["q8_async_completion_classification"],
                "invalid-failure-record",
            )
            report = (root / "summary.md").read_text()
            self.assertIn("truncated or internally inconsistent", report)
            self.assertNotIn("marker positively confirms", report)

    def test_rejects_bad_complement_pair_or_device_q8_async_records(self) -> None:
        for mutation in ("bad-complement", "wrong-pair", "wrong-device"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                production = root / "production"
                production.mkdir()
                stem = "r1-s1-attention-q8-async-completion"
                (production / f"{stem}.result").write_text(
                    "variant=attention-q8-async-completion\n"
                    "status=failed-device-loss\nexit_status=124\n"
                )
                failure = q8_async_failure(
                    129, marker_matches="yes", event_status="complete",
                    interpretation="post-compute-confirmed",
                )
                if mutation == "bad-complement":
                    expected = (~129) & ((1 << 64) - 1)
                    failure = failure.replace(
                        f"marker_complement={expected}", "marker_complement=0"
                    )
                else:
                    if mutation == "wrong-pair":
                        failure = failure.replace("home_tier=0", "home_tier=1")
                    else:
                        failure = failure.replace(
                            "partner_device=1", "partner_device=3"
                        )
                (production / f"{stem}.log").write_text(
                    q8_async_enabled() + q8_async_checkpoint(128) + failure
                )
                write_unhealthy_post(root, stem)

                subprocess.run(
                    [sys.executable, str(SUMMARIZER), str(root)], check=True
                )
                with (root / "summary.csv").open(newline="") as handle:
                    row = next(csv.DictReader(handle))
                self.assertEqual(
                    row["q8_async_completion_classification"],
                    "invalid-failure-record",
                )

    def test_mixed_invalid_repeat_outranks_positive_q8_async_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            positive = "r1-s1-attention-q8-async-completion"
            (production / f"{positive}.result").write_text(
                "variant=attention-q8-async-completion\n"
                "status=failed-device-loss\nexit_status=124\n"
            )
            (production / f"{positive}.log").write_text(
                q8_async_enabled() + q8_async_checkpoint(128) +
                q8_async_failure(
                    129, marker_matches="yes", event_status="complete",
                    interpretation="post-compute-confirmed",
                )
            )
            write_unhealthy_post(root, positive)

            invalid = "r2-s1-attention-q8-async-completion"
            (production / f"{invalid}.result").write_text(
                "variant=attention-q8-async-completion\n"
                "status=validation-failed\nexit_status=127\n"
            )
            (production / f"{invalid}.log").write_text(q8_async_enabled())
            write_healthy_post(root, invalid)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("Invalid evidence takes precedence", report)
            self.assertNotIn("marker positively confirms", report)

    def test_last_q8_async_failure_record_controls_classification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-async-completion"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-async-completion\n"
                "status=failed-device-loss\nexit_status=124\n"
            )
            (production / f"{stem}.log").write_text(
                q8_async_enabled() + q8_async_checkpoint(128) +
                q8_async_failure(
                    129, marker_matches="yes", event_status="complete",
                    interpretation="post-compute-confirmed",
                ) +
                q8_async_failure(
                    130, marker_matches="no", event_status="not-ready",
                    interpretation="inconclusive-no-positive-marker",
                )
            )
            write_unhealthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["q8_async_completion_classification"],
                "current-call-inconclusive",
            )
            self.assertIn(
                "current_sequence=130", row["q8_async_completion_last_failure"]
            )

    def test_invalid_outcome_outranks_positive_q8_async_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-async-completion"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-async-completion\n"
                "status=validation-failed\nexit_status=127\n"
            )
            (production / f"{stem}.log").write_text(
                q8_async_enabled() + q8_async_checkpoint(128) +
                q8_async_failure(
                    129, marker_matches="yes", event_status="complete",
                    interpretation="post-compute-confirmed",
                )
            )
            write_healthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("failed its exact marker/count/complement", report)
            self.assertNotIn("marker positively confirms", report)

    def test_requires_final_production_control(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-partner-bounce"
            (production / f"{stem}.result").write_text(
                "variant=partner-bounce\nstatus=passed\nexit_status=0\n"
                "last_phase=decode\nlast_event=frontier-complete\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA prefill attention row audit dispatch=split "
                "kind=mixed layer=17 pos=512 tokens=512 home=0 partner=2 "
                "q_bytes=1 result_bytes=1 query_copy_stream=destination "
                "gather_copy_stream=source\n"
            )
            write_healthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["pair0_attention_query_copy_schedule"], "destination"
            )
            self.assertEqual(
                row["pair0_attention_gather_copy_schedule"], "source"
            )
            report = (root / "summary.md").read_text()
            self.assertIn("production control has not run yet", report)
            self.assertIn("requires the final control", report)

    def test_shadow_phase_inference_advances_or_stops_at_decisive_arm(self) -> None:
        cases = (
            (
                (("attention-row-query-shadow", "failed-device-loss"),),
                "production-transport `query-copy` shadow arm lost a device",
            ),
            (
                (("attention-row-query-shadow", "passed"),),
                "Run `partner-compute` as a fresh one-shot arm",
            ),
            (
                (("attention-row-query-shadow", "validation-failed"),),
                "`query-copy` shadow arm is `invalid` rather than causal evidence",
            ),
            (
                (("attention-row-partner-shadow", "passed"),),
                "Run `result-gather` as a fresh one-shot arm",
            ),
            (
                (("attention-row-gather-shadow", "failed-device-loss"),),
                "Run `result-gather-dst` as a fresh one-shot arm",
            ),
            (
                (("attention-row-gather-dst-shadow", "passed"),),
                "peer-access direction are the isolated axis",
            ),
            (
                (("attention-row-gather-dst-shadow", "failed-device-loss"),),
                "Run `result-gather-chunk16` next",
            ),
            (
                (("attention-row-gather-chunk16-shadow", "passed"),),
                "single CUDA copy-operation size is the isolated axis",
            ),
            (
                (("attention-row-gather-chunk16-shadow", "failed-device-loss"),),
                "Operation size alone is not sufficient",
            ),
            (
                (
                    (
                        "attention-row-gather-chunk16-paced-shadow",
                        "failed-device-loss",
                    ),
                ),
                "Neither individual operation size nor an unbroken transfer burst",
            ),
            (
                (("attention-row-gather-chunk16-paced-shadow", "passed"),),
                "Continuous gather burst/overlap is the isolated axis",
            ),
            (
                (
                    (
                        "attention-row-gather-scratch-paced-shadow",
                        "failed-device-loss",
                    ),
                ),
                "production destination view, its address, and downstream "
                "consumption are therefore not required",
            ),
            (
                (("attention-row-gather-scratch-paced-shadow", "passed"),),
                "isolated axis is the production destination allocation/address",
            ),
            (
                (
                    (
                        "attention-row-gather-source-scratch-paced-shadow",
                        "failed-device-loss",
                    ),
                ),
                "Neither P2P endpoint allocation/address nor downstream "
                "consumption is required",
            ),
            (
                (
                    (
                        "attention-row-gather-source-scratch-paced-shadow",
                        "passed",
                    ),
                ),
                "direct P2P source being the production partner result allocation",
            ),
            (
                (
                    (
                        "attention-row-gather-preinitialized-source-paced-shadow",
                        "failed-device-loss",
                    ),
                ),
                "Production result contents, its allocation/address, a read from "
                "it, both P2P endpoint allocations, and downstream consumption "
                "are not required",
            ),
            (
                (
                    (
                        "attention-row-gather-preinitialized-source-paced-shadow",
                        "passed",
                    ),
                ),
                "isolated difference is reading/staging the freshly produced "
                "partner result",
            ),
            (
                (
                    (
                        "attention-row-gather-preinitialized-source-no-partner-paced-shadow",
                        "failed-device-loss",
                    ),
                ),
                "Partner attention compute, its output, and any read of that "
                "output are therefore not required",
            ),
            (
                (
                    (
                        "attention-row-gather-preinitialized-source-no-partner-paced-shadow",
                        "passed",
                    ),
                ),
                "partner attention execution is required in the measured trigger "
                "bundle",
            ),
            (
                (
                    ("attention-row-query-shadow", "passed"),
                    ("attention-row-partner-shadow", "failed-device-loss"),
                ),
                "production-transport `partner-compute` shadow arm lost a device",
            ),
        )
        for records, expected in cases:
            with self.subTest(records=records):
                with tempfile.TemporaryDirectory() as temporary:
                    root = pathlib.Path(temporary)
                    production = root / "production"
                    production.mkdir()
                    for slot, (variant, status) in enumerate(records, start=1):
                        stem = f"r1-s{slot}-{variant}"
                        exit_status = 0 if status == "passed" else 124
                        (production / f"{stem}.result").write_text(
                            f"variant={variant}\nstatus={status}\n"
                            f"exit_status={exit_status}\n"
                        )
                        (production / f"{stem}.log").write_text("")
                        if status == "passed":
                            write_healthy_post(root, stem)
                        else:
                            write_lost_watch(
                                root, stem, "1@00000000:03:00.0"
                            )

                    subprocess.run(
                        [sys.executable, str(SUMMARIZER), str(root)],
                        check=True,
                    )
                    report = (root / "summary.md").read_text()
                    self.assertIn(expected, report)

    def _summarize_activation(
            self, log_text: str, source_status: str = "failed"
    ) -> tuple[dict[str, str], str]:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-activation-fence"
            if source_status == "started-only":
                (production / f"{stem}.started").write_text(
                    "variant=attention-q8-activation-fence\nrepeat=1\n"
                )
            else:
                exit_status = 0 if source_status == "passed" else 124
                (production / f"{stem}.result").write_text(
                    "variant=attention-q8-activation-fence\n"
                    f"status={source_status}\nexit_status={exit_status}\n"
                )
            (production / f"{stem}.log").write_text(log_text)
            if source_status == "passed":
                write_healthy_post(root, stem)
            else:
                write_lost_watch(root, stem, "1@00000000:03:00.0")
            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            return row, (root / "summary.md").read_text()

    def test_validates_complete_q8_pre_activation_fence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = pathlib.Path(temporary) / "activation.log"
            log.write_text(
                q8_activation_enabled() + q8_activation_completed_calls(70) +
                q8_activation_summary(70)
            )
            subprocess.run([
                sys.executable, str(SUMMARIZER),
                "--validate-q8-pre-activation-fence-log", str(log),
            ], check=True)
        row, report = self._summarize_activation(
            q8_activation_enabled() + q8_activation_completed_calls(2) +
            q8_activation_summary(2),
            "passed",
        )
        self.assertEqual(
            row["q8_pre_activation_fence_classification"],
            "completed-all-activation-h2d-returned",
        )
        self.assertEqual(row["q8_pre_activation_fence_attempted"], "2")
        self.assertEqual(row["q8_pre_activation_fence_returned"], "2")
        self.assertIn("shared marker slot is later reused", report)
        self.assertIn("survival does not clear", report)

    def test_classifies_activation_failure_boundaries_without_causal_claims(
            self) -> None:
        cases = {
            "bounce-free-failed": (
                "activation-source-or-staging-setup-failed",
                "source staging/setup",
            ),
            "activation-d2h-failed": (
                "activation-source-d2h-failed", "surface earlier source-side work",
            ),
            "destination-switch-failed": (
                "activation-destination-switch-or-setup-failed",
                "destination switch/setup",
            ),
            "destination-sync-failed": (
                "activation-pre-h2d-device-sync-failed",
                "surfacing prior destination work or a poisoned context",
            ),
            "marker-launch-failed": (
                "activation-marker-channel-failure",
                "observation-channel boundary",
            ),
            "activation-h2d-failed": (
                "activation-fence-confirmed-h2d-failed",
                "does not prove that H2D initiated",
            ),
        }
        for event, (classification, wording) in cases.items():
            with self.subTest(event=event):
                body = q8_activation_completed_calls(1)
                if event == "activation-h2d-failed":
                    body += q8_activation_armed(2)
                body += q8_activation_failure(2, event)
                row, report = self._summarize_activation(
                    q8_activation_enabled() + body
                )
                self.assertEqual(
                    row["q8_pre_activation_fence_classification"], classification
                )
                self.assertIn(wording, report)
                self.assertIn("current_sequence=2", row[
                    "q8_pre_activation_fence_last_failure"
                ])

    def test_classifies_durable_unreturned_activation_for_device_loss_statuses(
            self) -> None:
        cases = (
            ("failed", "failed-device-loss"),
            ("interrupted-prior-run", "interrupted-prior-run-device-loss"),
            ("started-only", "interrupted-no-result-device-loss"),
        )
        log = (
            q8_activation_enabled() + q8_activation_armed(1) +
            q8_activation_checkpoint(1)
        )
        for source, expected_status in cases:
            with self.subTest(status=source):
                row, report = self._summarize_activation(log, source)
                self.assertEqual(row["status"], expected_status)
                self.assertEqual(
                    row["q8_pre_activation_fence_classification"],
                    "activation-fence-confirmed-h2d-return-not-observed-"
                    "trailing-checkpoint",
                )
                self.assertIn("same-sequence durable sparse checkpoint", report)
                self.assertIn("does not prove H2D failed or caused", report)

    def test_classifies_returned_activation_then_later_device_loss(self) -> None:
        row, report = self._summarize_activation(
            q8_activation_enabled() + q8_activation_completed_calls(2),
            "started-only",
        )
        self.assertEqual(
            row["q8_pre_activation_fence_classification"],
            "activation-h2d-returned-subsequent-locus-unresolved",
        )
        self.assertEqual(row["q8_pre_activation_fence_armed_count"], "2")
        self.assertEqual(row["q8_pre_activation_fence_returned_count"], "2")
        self.assertEqual(row["q8_pre_activation_fence_attempted"], "2")
        self.assertEqual(row["q8_pre_activation_fence_confirmed"], "2")
        self.assertEqual(row["q8_pre_activation_fence_returned"], "2")
        self.assertIn("synchronous activation H2D returned successfully", report)
        self.assertIn("subsequent locus unresolved", report)

    def test_uses_latest_breadcrumb_counts_after_sparse_activation_checkpoint(
            self) -> None:
        row, _ = self._summarize_activation(
            q8_activation_enabled() + q8_activation_completed_calls(2) +
            q8_activation_armed(3),
            "started-only",
        )
        self.assertEqual(
            row["q8_pre_activation_fence_classification"],
            "activation-fence-confirmed-h2d-return-not-observed",
        )
        self.assertEqual(row["q8_pre_activation_fence_attempted"], "3")
        self.assertEqual(row["q8_pre_activation_fence_confirmed"], "3")
        self.assertEqual(row["q8_pre_activation_fence_returned"], "2")

    def test_classifies_all_returned_summary_before_later_device_loss(
            self) -> None:
        for synchronized in ("yes", "no"):
            with self.subTest(partners_synchronized=synchronized):
                summary = q8_activation_summary(
                    2, partners_synchronized=synchronized,
                    # Without a successful cleanup sync, the inventory sample
                    # need not be an internally current marker pair.
                    shared_complement=(777 if synchronized == "no" else None),
                )
                row, report = self._summarize_activation(
                    q8_activation_enabled() + q8_activation_completed_calls(2) +
                    summary,
                    "started-only",
                )
                self.assertEqual(
                    row["q8_pre_activation_fence_classification"],
                    "activation-h2d-returned-subsequent-locus-unresolved",
                )
                self.assertEqual(row["q8_pre_activation_fence_attempted"], "2")
                self.assertEqual(row["q8_pre_activation_fence_confirmed"], "2")
                self.assertEqual(row["q8_pre_activation_fence_returned"], "2")
                self.assertIn("does not retract the earlier return evidence", report)
                self.assertIn("or identify the reset cause", report)

    def test_accepts_unsynchronized_cleanup_summary_after_unreturned_arm(
            self) -> None:
        body = (
            q8_activation_enabled() + q8_activation_armed(1) +
            q8_activation_checkpoint(1) +
            q8_activation_summary(
                1, returned=0, partners_synchronized="no",
                shared_complement=777,
            )
        )
        row, report = self._summarize_activation(body, "started-only")
        self.assertEqual(
            row["q8_pre_activation_fence_classification"],
            "activation-fence-confirmed-h2d-return-not-observed-"
            "trailing-checkpoint",
        )
        self.assertEqual(row["q8_pre_activation_fence_attempted"], "1")
        self.assertEqual(row["q8_pre_activation_fence_returned"], "0")
        self.assertIn("does not prove H2D failed or caused", report)

    def test_accepts_explicit_activation_sequence_domain_exhaustion(self) -> None:
        row, report = self._summarize_activation(
            q8_activation_enabled() + q8_activation_sequence_domain_exhausted()
        )
        self.assertEqual(
            row["q8_pre_activation_fence_classification"],
            "activation-audit-state-failed",
        )
        self.assertEqual(
            row["q8_pre_activation_fence_attempted"], str(Q8_ACTIVATION_TAG)
        )
        self.assertIn("software diagnostic evidence", report)

    def test_validates_failure_teardown_summary_consistency(self) -> None:
        first_call_failure = (
            q8_activation_enabled() +
            q8_activation_failure(1, "bounce-free-failed") +
            q8_activation_summary(
                1, confirmed=0, returned=0, failed=1,
                shared=0, shared_complement=0,
            )
        )
        row, _ = self._summarize_activation(first_call_failure)
        self.assertEqual(
            row["q8_pre_activation_fence_classification"],
            "activation-source-or-staging-setup-failed",
        )

        calls_and_failure = (
            q8_activation_enabled() + q8_activation_completed_calls(1) +
            q8_activation_failure(2, "activation-d2h-failed")
        )
        consistent = q8_activation_summary(
            2, confirmed=1, returned=1, failed=1,
            partners_synchronized="no", shared_complement=777,
        )
        row, _ = self._summarize_activation(calls_and_failure + consistent)
        self.assertEqual(
            row["q8_pre_activation_fence_classification"],
            "activation-source-d2h-failed",
        )
        self.assertEqual(row["q8_pre_activation_fence_attempted"], "2")
        self.assertEqual(row["q8_pre_activation_fence_confirmed"], "1")
        self.assertEqual(row["q8_pre_activation_fence_returned"], "1")
        self.assertEqual(row["q8_pre_activation_fence_failed"], "1")

        contradictory = q8_activation_summary(1)
        row, report = self._summarize_activation(
            calls_and_failure + contradictory
        )
        self.assertEqual(
            row["q8_pre_activation_fence_classification"],
            "invalid-activation-fence-record",
        )
        self.assertIn("invalid for boundary inference", report)

        reordered = (
            q8_activation_enabled() + q8_activation_completed_calls(1) +
            consistent + q8_activation_failure(2, "activation-d2h-failed")
        )
        row, _ = self._summarize_activation(reordered)
        self.assertEqual(
            row["q8_pre_activation_fence_classification"],
            "invalid-activation-fence-record",
        )

    def test_rejects_malformed_activation_sequences_tags_and_bindings(self) -> None:
        marker, _ = q8_activation_marker(1)
        complete = (
            q8_activation_enabled() + q8_activation_completed_calls(1) +
            q8_activation_summary(1)
        )
        cases = {
            "tag": complete.replace(
                f"marker_sequence={marker}", "marker_sequence=1", 1
            ),
            "physical-pair": complete.replace(
                "home_device=0", "home_device=3", 1
            ),
            "binding": complete.replace(
                "binding_label=tensor:blk.0.attn_output_b.weight",
                "binding_label=tensor:blk.1.attn_output_b.weight",
                1,
            ),
            "enable": complete.replace("tagged-exact", "exact", 1),
            "duplicate-enable": q8_activation_enabled() + complete,
            "enable-after-evidence": (
                q8_activation_completed_calls(1) + q8_activation_summary(1) +
                q8_activation_enabled()
            ),
            "summary-before-evidence": (
                q8_activation_enabled() + q8_activation_summary(1) +
                q8_activation_completed_calls(1)
            ),
            "shared-complement": complete.replace(
                "shared_slot_complement=18446744073709550838",
                "shared_slot_complement=777",
            ),
            "unsynchronized-complete": complete.replace(
                "partners_synchronized=yes", "partners_synchronized=no",
            ),
            "order": (
                q8_activation_enabled() + q8_activation_armed(1) +
                q8_activation_armed(2) + q8_activation_returned(1) +
                q8_activation_returned(2) + q8_activation_summary(2)
            ),
        }
        for name, body in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as temporary:
                log = pathlib.Path(temporary) / "activation.log"
                log.write_text(body)
                completed = subprocess.run([
                    sys.executable, str(SUMMARIZER),
                    "--validate-q8-pre-activation-fence-log", str(log),
                ], capture_output=True, text=True)
                self.assertNotEqual(completed.returncode, 0)

    def test_rejects_activation_checkpoint_metadata_mismatch(self) -> None:
        body = (
            q8_activation_enabled() + q8_activation_armed(1) +
            q8_activation_checkpoint(
                1, binding_label="tensor:blk.1.attn_output_b.weight"
            ) + q8_activation_returned(1) + q8_activation_summary(1)
        )
        with tempfile.TemporaryDirectory() as temporary:
            log = pathlib.Path(temporary) / "activation.log"
            log.write_text(body)
            completed = subprocess.run([
                sys.executable, str(SUMMARIZER),
                "--validate-q8-pre-activation-fence-log", str(log),
            ], capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("activation-checkpoint-binding-mismatch", completed.stderr)

    def test_validates_global_q8_compute_fence_dynamic_bindings(self) -> None:
        enabled = (
            "ds4: CUDA q8 partner compute fence audit enabled: logical_pairs=0 "
            "boundary=post-submit-device-sync "
            "scope=every-selected-partner-call identity=dynamic\n"
        )

        def record(kind: str, sequence: int, layer: int) -> str:
            status = "submitted" if kind == "armed" else "complete"
            return (
                f"ds4: CUDA q8 partner compute fence {kind} "
                f"sequence={sequence} status={status} home_tier=0 "
                "home_device=0 partner_tier=2 partner_device=1 tokens=512 "
                f"in=8192 out=4096 binding_label=tensor:blk.{layer}."
                "attn_output_b.weight passed_label=attn_output_b "
                f"weight_offset={143000000000 + layer}\n"
            )

        with tempfile.TemporaryDirectory() as temporary:
            log = pathlib.Path(temporary) / "compute.log"
            log.write_text(
                enabled + record("armed", 1, 7) + record("returned", 1, 7) +
                record("armed", 2, 13) + record("returned", 2, 13)
            )
            subprocess.run([
                sys.executable, str(SUMMARIZER),
                "--validate-q8-compute-fence-log", str(log),
            ], check=True)

    def test_rejects_unmatched_global_q8_compute_fence_arm(self) -> None:
        log_body = (
            "ds4: CUDA q8 partner compute fence audit enabled: logical_pairs=0 "
            "boundary=post-submit-device-sync "
            "scope=every-selected-partner-call identity=dynamic\n"
            "ds4: CUDA q8 partner compute fence armed sequence=1 "
            "status=submitted home_tier=0 home_device=0 partner_tier=2 "
            "partner_device=1 tokens=512 in=8192 out=4096 "
            "binding_label=tensor:blk.13.attn_output_b.weight "
            "passed_label=attn_output_b weight_offset=143388891904\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            log = pathlib.Path(temporary) / "compute.log"
            log.write_text(log_body)
            completed = subprocess.run([
                sys.executable, str(SUMMARIZER),
                "--validate-q8-compute-fence-log", str(log),
            ], capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("unmatched-armed-record", completed.stderr)

    def test_summarizes_unmatched_global_compute_arm_without_layer_claim(
            self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-global-compute-fence"
            (production / f"{stem}.started").write_text(
                "variant=attention-q8-global-compute-fence\nrepeat=1\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA q8 partner compute fence audit enabled: "
                "logical_pairs=0 boundary=post-submit-device-sync "
                "scope=every-selected-partner-call identity=dynamic\n"
                "ds4: CUDA q8 partner compute fence armed sequence=41 "
                "status=submitted home_tier=0 home_device=0 partner_tier=2 "
                "partner_device=1 tokens=512 in=8192 out=4096 "
                "binding_label=tensor:blk.13.attn_output_b.weight "
                "passed_label=attn_output_b weight_offset=143388891904\n"
            )
            write_lost_watch(root, stem, "1@00000000:03:00.0")
            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["q8_compute_fence_classification"],
                "compute-fence-return-not-observed",
            )
            report = (root / "summary.md").read_text()
            self.assertIn("dynamically observed final boundary", report)
            self.assertIn("without asserting", report)

    def test_classifies_partial_mapped_host_marker_after_compute_return(
            self) -> None:
        sequence = 65
        stale_complement = (~(sequence | (1 << 63))) & ((1 << 64) - 1)
        failure = q8_pre_gather_failure(sequence, "sync-failed")
        failure = failure.replace(
            f"marker_sequence={sequence - 1} "
            f"marker_complement={(~(sequence - 1)) & ((1 << 64) - 1)}",
            f"marker_sequence={sequence} marker_complement={stale_complement}",
        )
        compute_enable = (
            "ds4: CUDA q8 partner compute fence audit enabled: logical_pairs=0 "
            "boundary=post-submit-device-sync "
            "scope=every-selected-partner-call identity=dynamic\n"
        )
        compute_identity = (
            "sequence=1 home_tier=0 home_device=0 partner_tier=2 "
            "partner_device=1 tokens=512 in=8192 out=4096 "
            "binding_label=tensor:blk.11.attn_output_b.weight "
            "passed_label=attn_output_b weight_offset=143053907200\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-global-compute-fence"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-global-compute-fence\n"
                "status=failed-device-loss\nexit_status=124\n"
            )
            (production / f"{stem}.log").write_text(
                compute_enable +
                "ds4: CUDA q8 partner compute fence armed status=submitted " +
                compute_identity +
                "ds4: CUDA q8 partner compute fence returned status=complete " +
                compute_identity + q8_pre_gather_enabled() +
                q8_pre_gather_checkpointed_calls(sequence - 1) + failure
            )
            write_unhealthy_post(root, stem)
            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["q8_compute_fence_classification"],
                "all-observed-compute-fences-returned",
            )
            self.assertEqual(
                row["q8_pre_gather_fence_classification"],
                "mapped-host-marker-partial-write",
            )
            report = (root / "summary.md").read_text()
            self.assertIn("complement remained the stale complement", report)
            self.assertIn("result D2H was never attempted", report)

    def test_validates_marker_free_direct_gather_sequence(self) -> None:
        enable = (
            "ds4: CUDA q8 partner direct gather audit enabled: "
            "logical_pairs=0 boundary=compute-sync-to-synchronous-host-bounce "
            "mapped_host_marker=no event=no identity=dynamic\n"
        )

        def record(kind: str, sequence: int, status: str, flags: str) -> str:
            return (
                f"ds4: CUDA q8 partner direct gather {kind} "
                f"sequence={sequence} status={status} {flags} "
                "home_tier=0 home_device=0 partner_tier=2 partner_device=1 "
                "tokens=512 in=8192 out=4096 "
                "binding_label=tensor:blk.11.attn_output_b.weight "
                "passed_label=attn_output_b weight_offset=143053907200\n"
            )

        empty = (
            "result_d2h_attempted=no result_d2h_completed=no "
            "result_h2d_attempted=no result_h2d_completed=no"
        )
        complete = (
            "result_d2h_attempted=yes result_d2h_completed=yes "
            "result_h2d_attempted=yes result_h2d_completed=yes"
        )
        with tempfile.TemporaryDirectory() as temporary:
            log = pathlib.Path(temporary) / "direct.log"
            log.write_text(
                enable + record("armed", 1, "post-compute-sync", empty) +
                record("returned", 1, "copy-complete", complete) +
                record("armed", 2, "post-compute-sync", empty) +
                record("returned", 2, "copy-complete", complete)
            )
            subprocess.run([
                sys.executable, str(SUMMARIZER),
                "--validate-q8-direct-gather-log", str(log),
            ], check=True)

    def test_classifies_direct_gather_copy_boundaries(self) -> None:
        enable = (
            "ds4: CUDA q8 partner direct gather audit enabled: "
            "logical_pairs=0 boundary=compute-sync-to-synchronous-host-bounce "
            "mapped_host_marker=no event=no identity=dynamic\n"
        )
        identity = (
            "home_tier=0 home_device=0 partner_tier=2 partner_device=1 "
            "tokens=512 in=8192 out=4096 "
            "binding_label=tensor:blk.11.attn_output_b.weight "
            "passed_label=attn_output_b weight_offset=143053907200"
        )
        armed = (
            "ds4: CUDA q8 partner direct gather armed sequence=1 "
            "status=post-compute-sync result_d2h_attempted=no "
            "result_d2h_completed=no result_h2d_attempted=no "
            f"result_h2d_completed=no {identity}\n"
        )
        cases = {
            "result-d2h-attempt-failed": (
                "yes", "no", "no", "no"
            ),
            "result-h2d-attempt-failed": (
                "yes", "yes", "yes", "no"
            ),
        }
        for expected, flags in cases.items():
            with self.subTest(expected=expected):
                d2ha, d2hc, h2da, h2dc = flags
                failure = (
                    "ds4: CUDA q8 partner direct gather failure sequence=1 "
                    "status=copy-failed "
                    f"result_d2h_attempted={d2ha} "
                    f"result_d2h_completed={d2hc} "
                    f"result_h2d_attempted={h2da} "
                    f"result_h2d_completed={h2dc} {identity}\n"
                )
                with tempfile.TemporaryDirectory() as temporary:
                    root = pathlib.Path(temporary)
                    production = root / "production"
                    production.mkdir()
                    stem = "r1-s1-attention-q8-direct-gather-fence"
                    (production / f"{stem}.result").write_text(
                        "variant=attention-q8-direct-gather-fence\n"
                        "status=failed-device-loss\nexit_status=124\n"
                    )
                    (production / f"{stem}.log").write_text(
                        enable + armed + failure
                    )
                    write_unhealthy_post(root, stem)
                    subprocess.run([
                        sys.executable, str(SUMMARIZER), str(root)
                    ], check=True)
                    with (root / "summary.csv").open(newline="") as handle:
                        row = next(csv.DictReader(handle))
                    self.assertEqual(
                        row["q8_direct_gather_classification"], expected
                    )

    def test_rejects_direct_gather_identity_mismatch(self) -> None:
        enable = (
            "ds4: CUDA q8 partner direct gather audit enabled: "
            "logical_pairs=0 boundary=compute-sync-to-synchronous-host-bounce "
            "mapped_host_marker=no event=no identity=dynamic\n"
        )
        common = (
            "home_tier=0 home_device=0 partner_tier=2 partner_device=1 "
            "tokens=512 in=8192 out=4096 passed_label=attn_output_b "
            "weight_offset=143053907200"
        )
        log_body = (
            enable +
            "ds4: CUDA q8 partner direct gather armed sequence=1 "
            "status=post-compute-sync result_d2h_attempted=no "
            "result_d2h_completed=no result_h2d_attempted=no "
            "result_h2d_completed=no " + common +
            " binding_label=tensor:blk.11.attn_output_b.weight\n" +
            "ds4: CUDA q8 partner direct gather returned sequence=1 "
            "status=copy-complete result_d2h_attempted=yes "
            "result_d2h_completed=yes result_h2d_attempted=yes "
            "result_h2d_completed=yes " + common +
            " binding_label=tensor:blk.12.attn_output_b.weight\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            log = pathlib.Path(temporary) / "direct.log"
            log.write_text(log_body)
            completed = subprocess.run([
                sys.executable, str(SUMMARIZER),
                "--validate-q8-direct-gather-log", str(log),
            ], capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("armed-completion-identity-mismatch", completed.stderr)


if __name__ == "__main__":
    unittest.main()
