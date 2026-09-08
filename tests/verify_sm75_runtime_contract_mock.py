#!/usr/bin/env python3
"""Check native host-mock logs. No CUDA imports or GPU execution."""
import json
import stat
import sys
from pathlib import Path

if not __debug__:
    raise SystemExit("mock verification requires assertions; run Python without -O")


def verify(directory):
    root = Path(directory)
    analysis = json.loads((root / "mock-analysis.json").read_text())
    assert analysis["trace_complete"] and analysis["required_capture_present"]
    assert len(analysis["process_ids"]) == 1 and not analysis["violations"]
    assert not analysis["expected_transition"]["postburnin_transition_required_present"], "small mock is not the failure workload"
    loaded = [json.loads(x) for x in (root / "load-only.jsonl").read_text().splitlines()]
    assert [x["api"] for x in loaded] == ["trace_start", "trace_end"], "constructor invoked APIs"
    events = [json.loads(x) for x in (root / "mock.jsonl").read_text().splitlines()]
    assert events[0]["api"] == "trace_start"
    assert sum(e["api"] == "trace_end" for e in events) == 1
    finalizer_index = next(i for i,e in enumerate(events) if e["api"] == "trace_end")
    assert all(e["api"] == "__cudaUnregisterFatBinary" for e in events[finalizer_index+1:])
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    assert stat.S_IMODE((root / "mock.jsonl").stat().st_mode) == 0o600
    enters = {e["call_id"]: e for e in events if e["event"] == "enter"}
    exits = {e["call_id"]: e for e in events if e["event"] == "exit"}
    assert enters.keys() == exits.keys(), "missing native wrapper exits"
    assert all(e["status"] == 0 for e in exits.values())
    registrations = [e for e in enters.values() if e["api"] == "__cudaRegisterFunction"]
    unregistrations = [e for e in enters.values() if e["api"] == "__cudaUnregisterFatBinary"]
    assert len(registrations) == 2 and len(unregistrations) == 2
    registered_owners = {e["args"]["owner"] for e in registrations}
    assert len(registered_owners) == 2 and registered_owners == {e["args"]["owner"] for e in unregistrations}
    first_device_set = next(e["seq"] for e in enters.values() if e["api"] == "cudaSetDevice")
    assert registrations[0]["seq"] < first_device_set, "provider constructor was not exercised before main"
    late_unregistrations = sum(e["event"] == "enter" for e in events[finalizer_index+1:])
    gemms = [e for e in enters.values() if e["api"] == "cublasGemmEx"]
    assert len(gemms) == 2 and all(e["args"]["algorithm"] == 103 for e in gemms)
    contracts = [e["result"] for e in events if e["api"] == "gemm_contract"]
    assert len(contracts) == 3
    assert contracts[0]["scalars"]["alpha"]["bits_hex"] == "0000803f"
    assert contracts[0]["scalars"]["beta"]["bits_hex"] == "00000000"
    assert contracts[2]["handle"]["pointer_mode"] == 1
    assert "scalars" not in contracts[2], "DEVICE scalars must not be read"
    assert contracts[0]["operands"]["A"]["bytes"] == 64
    assert contracts[0]["operands"]["B"]["bytes"] == 32
    assert contracts[0]["operands"]["C"]["bytes"] == 32
    launch = [e for e in enters.values() if e["api"] == "cudaLaunchKernel"]
    assert len(launch) == 1 and launch[0]["args"]["args_pointer"] == "0x1"
    assert "kernel_args" not in launch[0]["args"], "arbitrary kernel args must not be read"
    conversion = [e["result"] for e in events if e["api"] == "conversion_contract"]
    assert len(conversion) == 2, "unregistered handles must retire recognized decoding"
    assert conversion[0]["argument_read_status"] == "complete" and conversion[0]["count"] == 8
    assert conversion[1]["argument_read_status"] == "unreadable"
    assert "dst" not in conversion[1], "partial reads must not become guessed arguments"
    forwarded = {e["parent_call_id"] for e in events if e["api"] == "forward_to_real"}
    assert all(e["call_id"] in forwarded for e in gemms)
    batches = [e for e in enters.values() if e["api"] == "cublasGemmStridedBatchedEx"]
    assert len(batches) == 1 and batches[0]["args"]["stride_a"] == (1 << 34) + 32
    assert batches[0]["args"]["stride_b"] == (1 << 35) + 16
    assert batches[0]["args"]["stride_c"] == (1 << 36) + 8
    assert batches[0]["args"]["batch_count"] == 1 and batches[0]["call_id"] in forwarded
    assert contracts[0]["context_query_status"] == 0 and contracts[0]["context"] == "0x9abc"
    failure = [json.loads(x) for x in (root / "mock-query-failure.jsonl").read_text().splitlines()]
    failed_gemm = [e for e in failure if e["event"] == "enter" and e["api"] == "cublasGemmEx"]
    assert len(failed_gemm) == 1
    assert not any(e["api"] == "forward_to_real" and e.get("parent_call_id") == failed_gemm[0]["call_id"] for e in failure)
    assert failure[-1]["api"] == "trace.cudaPointerGetAttributes" and failure[-1]["status"] != 0
    return {"status": "host-native-mock-pass", "gpu_calls": 0, "event_count": len(events),
            "load_only_events": len(loaded), "gemm_passthrough_calls": len(gemms), "batched_passthrough_calls": len(batches),
            "provider_constructor_exercised": True,
            "native_trace_analyzer_integration": "passed",
            "post_finalizer_unregisters_observed": late_unregistrations,
            "late_finalizer_case_exercised": bool(late_unregistrations)}


if __name__ == "__main__":
    print(json.dumps(verify(sys.argv[1]), indent=2, sort_keys=True))
