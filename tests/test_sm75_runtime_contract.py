#!/usr/bin/env python3
"""CPU-only tests: synthetic trace bytes, never load CUDA or run a GPU tool."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


MODULE = Path(__file__).resolve().parents[1] / "speed-bench/analyze-sm75-runtime-contract.py"
SPEC = importlib.util.spec_from_file_location("runtime_contract", MODULE)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
STAMP = 1788812244000000123  # Deliberately not exactly representable as a double.


class Trace:
    def __init__(self):
        self.rows = []
        self.next_call = 0
        self.meta("trace_start")

    def record(self, event, api, call_id=0, tid=11, **fields):
        seq = len(self.rows) + 1
        result = dict(schema=module.SCHEMA, seq=seq, call_id=call_id, event=event,
                      api=api, pid=100, tid=tid, monotonic_ns=10000000000000000 + seq,
                      realtime_ns=STAMP + seq)
        result.update(fields)
        self.rows.append(result)
        return result

    def meta(self, api):
        return self.record("meta", api)

    def enter(self, api, args=None, tid=11):
        self.next_call += 1
        self.record("enter", api, self.next_call, args=args or {}, tid=tid)
        return self.next_call

    def exit(self, api, call_id, status=0, result=None, tid=11):
        return self.record("exit", api, call_id, status=status, result=result or {}, tid=tid)

    def call(self, api, args=None, status=0, result=None, tid=11):
        call_id = self.enter(api, args, tid)
        self.exit(api, call_id, status, result, tid)
        return call_id

    def malloc(self, ptr, size, tid=11):
        return self.call("cudaMalloc", {"size": size, "device_query_status": 0, "device": 0,
                                        "context_query_status": 0, "context": "0x7000"},
                         result={"ptr": hex(ptr)}, tid=tid)

    def sync(self, status=0, tid=11, device=0):
        return self.call("cudaDeviceSynchronize", {"device_query_status": 0, "device": device,
                                                   "context_query_status": 0, "context": "0x7000"}, status=status, tid=tid)

    def gemm(self, args=None, tid=11, status=0, close=True, observe=True, observation_override=None, api="cublasGemmEx", forward=True):
        args = standard_args() if args is None else args
        call_id = self.enter(api, args, tid=tid)
        if observe:
            result = dict(device_query_status=0, device=0, context_query_status=0, context="0x7000",
                          handle={"pointer_mode_query_status": 0, "pointer_mode": 0,
                                  "math_mode_query_status": 0, "math_mode": 0,
                                  "stream_query_status": 0, "stream": "0x0"},
                          operands={key: {"attributes_status": 0, "type": 2, "device": 0,
                                          "device_pointer": args[key], "host_pointer": "0x0",
                                          "range_status": 0, "base": hex(POINTERS[key]), "bytes": SIZES[key]}
                                    for key in ("A", "B", "C")},
                          scalars={"alpha": {"host_read_bytes": 4, "bits_hex": "0000803f"},
                                   "beta": {"host_read_bytes": 4, "bits_hex": "00000000"}})
            if observation_override:
                result.update(observation_override)
            self.record("observation", "gemm_contract", parent_call_id=call_id, result=result, tid=tid)
        if forward:
            self.record("observation", "forward_to_real", parent_call_id=call_id, result={}, tid=tid)
        if close:
            self.exit(api, call_id, status=status, tid=tid)
        return call_id

    def standard_allocations(self):
        self.call("cudaSetDevice", {"device": 0})
        for key in ("A", "B", "C"):
            self.malloc(POINTERS[key], SIZES[key])
        self.call("cublasCreate_v2", result={"handle": "0x9000"})
        self.call("__cudaLaunchKernel", {"kernel": "0x8000", "stream": "0x0"})

    def conversion(self, count=8192*256, src=0x40000000, dst=0x20000000,
                   argument_read_status="complete", tid=11, stream="0x0"):
        call_id = self.enter("__cudaLaunchKernel", {"kernel": "0x8800", "stream": stream,
                                                     "device_query_status": 0, "device": 0,
                                                     "context_query_status": 0, "context": "0x7000"}, tid=tid)
        self.record("observation", "conversion_contract", parent_call_id=call_id, tid=tid,
                    result={"recognized": True, "kernel_name": "_Z17f32_to_f16_kernelP6__halfPKfm",
                            "argument_read_status": argument_read_status, "src": hex(src), "dst": hex(dst), "count": count})
        self.exit("__cudaLaunchKernel", call_id, tid=tid)
        return call_id

    def finish(self):
        self.meta("trace_end")
        return self

    def bytes(self):
        return b"".join(json.dumps(row).encode() + b"\n" for row in self.rows)


POINTERS = {"A": 0x10000000, "B": 0x20000000, "C": 0x30000000}
SIZES = {"A": 8192 * 4096 * 2, "B": 8192 * 512 * 2, "C": 4096 * 512 * 4}


def standard_args(n=256, algorithm=103):
    return dict(handle="0x9000", A=hex(POINTERS["A"]), B=hex(POINTERS["B"]), C=hex(POINTERS["C"]),
                transa=1, transb=0, m=4096, n=n, k=8192, atype=2, btype=2, ctype=0,
                lda=8192, ldb=8192, ldc=4096, compute_type=68, algorithm=algorithm)


def codes(report, field="gaps"):
    return {item["code"] for item in report[field]}


class AnalyzeTests(unittest.TestCase):
    def analyze(self, trace):
        data = trace.bytes() if isinstance(trace, Trace) else trace
        with tempfile.TemporaryDirectory(prefix="runtime-contract-cpu-") as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_bytes(data)
            original = path.read_bytes()
            result = module.analyze_file(path)
            self.assertEqual(path.read_bytes(), original)
            return result

    def normal(self):
        trace = Trace()
        trace.standard_allocations()
        trace.gemm(standard_args(512, -1))
        trace.sync()
        trace.gemm()
        trace.sync()
        return trace.finish()

    def test_complete_trace_has_bounded_claim_and_integer_timestamps(self):
        report = self.analyze(self.normal())
        self.assertTrue(report["trace_complete"])
        self.assertTrue(report["required_capture_present"])
        self.assertEqual(report["process_ids"], [100])
        self.assertEqual(report["api_counts"]["cublasGemmEx"], 2)
        self.assertEqual(report["violations"], [])
        self.assertEqual(report["status"], "observed-contract-with-gaps")
        self.assertEqual(report["metadata"][0]["realtime_ns"], STAMP + 1)
        self.assertEqual(json.loads(json.dumps(report))["metadata"][0]["realtime_ns"], STAMP + 1)
        transition = report["expected_transition"]["observed_transitions"][0]
        self.assertTrue(transition["default512_device_sync_completed_before_half_submit"])
        self.assertTrue(report["expected_transition"]["expected_transition_required_present"])
        self.assertEqual(report["expected_transition"]["post_1024_observed_half_calls_adjacent_transition_candidates"], [])

    def test_partial_fault_tail_preserves_open_call(self):
        trace = Trace()
        trace.standard_allocations()
        call_id = trace.gemm(close=False)
        report = self.analyze(trace.bytes() + b'{"schema":"ds4-runtime')
        self.assertFalse(report["trace_complete"])
        self.assertEqual(report["open_calls"][0]["call_id"], call_id)
        self.assertIn("partial-final-record", codes(report))
        self.assertNotIn("status", next(c for c in report["calls"] if c["call_id"] == call_id))

    def test_missing_footer_is_not_complete(self):
        trace = self.normal()
        trace.rows.pop()
        report = self.analyze(trace)
        self.assertFalse(report["trace_complete"])
        self.assertIn("missing-or-duplicate-trace-end", codes(report))

    def test_balanced_late_module_unregistration_accepted(self):
        trace = self.normal()
        trace.call("__cudaUnregisterFatBinary", {"fatbin_handle": "0x555"})
        report = self.analyze(trace)
        self.assertTrue(report["trace_complete"])
        self.assertEqual(len(report["accepted_late_cuda_module_finalizers"]), 1)

    def test_unrelated_call_after_finalizer_is_gap(self):
        trace = self.normal()
        trace.call("cudaSetDevice", {"device": 0})
        report = self.analyze(trace)
        self.assertFalse(report["trace_complete"])
        self.assertIn("unqualified-records-after-trace-finalizer", codes(report))

    def test_unfinished_late_unregistration_not_complete(self):
        trace = self.normal()
        trace.enter("__cudaUnregisterFatBinary", {"fatbin_handle": "0x555"})
        report = self.analyze(trace)
        self.assertFalse(report["trace_complete"])

    def test_missing_header_is_not_complete(self):
        trace = self.normal()
        trace.rows.pop(0)
        self.assertFalse(self.analyze(trace)["trace_complete"])

    def test_empty_trace(self):
        report = self.analyze(b"")
        self.assertFalse(report["trace_complete"])
        self.assertIn("empty-trace", codes(report))

    def test_missing_gemm_observation_not_success(self):
        trace = Trace()
        trace.standard_allocations()
        trace.gemm(observe=False)
        report = self.analyze(trace.finish())
        self.assertIn("missing-or-duplicate-gemm-contract-observation", codes(report))
        self.assertIn("operand-driver-address-range-unresolved", codes(report))

    def test_unknown_allocation_remains_unknown_with_driver_range(self):
        trace = Trace()
        trace.gemm()
        report = self.analyze(trace.finish())
        self.assertIn("operand-allocation-unobserved", codes(report))
        operand = report["calls"][0]["operand_contracts"]["A"]
        self.assertEqual(operand["range_status"], "unknown")
        self.assertEqual(operand["query_range_status"], "within-queried-range-at-observation")

    def test_oversized_operand_allocation(self):
        trace = Trace()
        trace.malloc(POINTERS["A"], 10)
        trace.gemm()
        report = self.analyze(trace.finish())
        self.assertIn("operand-exceeds-recorded-allocation", codes(report, "violations"))

    def test_driver_range_oob(self):
        trace = Trace()
        trace.standard_allocations()
        trace.gemm(observation_override={"operands": {"A": {"attributes_status": 0, "range_status": 0,
                                                             "base": hex(POINTERS["A"]), "bytes": 10}}})
        report = self.analyze(trace.finish())
        self.assertIn("operand-exceeds-queried-address-range", codes(report, "violations"))

    def test_subview_offset_fits(self):
        trace = Trace()
        trace.standard_allocations()
        args = standard_args()
        args["B"] = hex(POINTERS["B"] + SIZES["B"] // 2)
        args["C"] = hex(POINTERS["C"] + SIZES["C"] // 2)
        trace.gemm(args)
        report = self.analyze(trace.finish())
        self.assertEqual(report["violations"], [])
        c = next(c for c in report["calls"] if c["api"] == "cublasGemmEx")
        self.assertEqual(c["operand_contracts"]["C"]["offset_bytes"], 4194304)

    def test_free_and_address_reuse_generation(self):
        trace = Trace()
        trace.standard_allocations()
        trace.call("cudaFree", {"ptr": hex(POINTERS["A"])})
        trace.malloc(POINTERS["A"], SIZES["A"])
        trace.gemm()
        report = self.analyze(trace.finish())
        self.assertEqual(report["violations"], [])
        call = next(c for c in report["calls"] if c["api"] == "cublasGemmEx")
        self.assertEqual(call["operand_contracts"]["A"]["allocation_generation"], 2)

    def test_free_without_reallocation_detected(self):
        trace = Trace()
        trace.standard_allocations()
        trace.call("cudaFree", {"ptr": hex(POINTERS["A"])})
        trace.gemm()
        report = self.analyze(trace.finish())
        self.assertIn("operand-in-freed-recorded-allocation", codes(report, "violations"))

    def test_failed_free_is_unknown_not_proven_freed(self):
        trace = Trace()
        trace.standard_allocations()
        trace.call("cudaFree", {"ptr": hex(POINTERS["A"])}, status=719)
        trace.gemm()
        report = self.analyze(trace.finish())
        self.assertIn("operand-lifetime-after-failed-free-unresolved", codes(report))
        self.assertNotIn("operand-in-freed-recorded-allocation", codes(report, "violations"))
        self.assertIsNone(report["allocation_history"][0]["free_exit_seq"])

    def test_free_overlapping_submit_not_ordered(self):
        trace = Trace()
        trace.standard_allocations()
        freeing = trace.enter("cudaFree", {"ptr": hex(POINTERS["A"])}, tid=22)
        trace.gemm()
        trace.exit("cudaFree", freeing, tid=22)
        report = self.analyze(trace.finish())
        self.assertIn("operand-use-overlaps-host-free-interval", codes(report))

    def test_cross_thread_sync_does_not_clear_completion(self):
        trace = Trace()
        trace.standard_allocations()
        trace.gemm()
        trace.sync(tid=22)
        report = self.analyze(trace.finish())
        gemm = next(c for c in report["calls"] if c["api"] == "cublasGemmEx")
        self.assertEqual(gemm["completion_evidence"]["status"], "unproven")
        self.assertIn("cross-thread-or-cross-stream-dependencies-not-fully-modeled", codes(report))

    def test_different_device_sync_not_clear(self):
        trace = Trace()
        trace.standard_allocations()
        trace.gemm()
        trace.sync(device=1)
        report = self.analyze(trace.finish())
        self.assertIn("gemm-completion-not-established", codes(report))

    def test_failed_sync_records_api_error_not_success(self):
        trace = Trace()
        trace.standard_allocations()
        trace.gemm()
        trace.sync(status=719)
        report = self.analyze(trace.finish())
        self.assertEqual(report["api_errors"][0]["status"], 719)
        self.assertIn("gemm-completion-not-established", codes(report))

    def test_successful_enqueue_does_not_mean_completion(self):
        trace = Trace()
        trace.gemm()
        report = self.analyze(trace.finish())
        self.assertEqual(report["calls"][0]["return_interpretation"], "host-library-return-success-not-gpu-completion")

    def test_interposer_entry_without_forward_is_not_real_library_call(self):
        trace = Trace()
        trace.gemm(forward=False, close=False)
        report = self.analyze(trace)
        self.assertIn("gemm-real-library-forwarding-unresolved", codes(report))
        self.assertEqual(report["expected_transition"]["production103_half256_calls"], 0)

    def test_gemm_error_followed_by_successful_sync_is_not_gemm_completion(self):
        trace = Trace()
        trace.standard_allocations()
        trace.gemm(status=15)
        trace.sync()
        report = self.analyze(trace.finish())
        gemm = next(c for c in report["calls"] if c["api"] == "cublasGemmEx")
        self.assertEqual(gemm["completion_evidence"]["status"], "unproven")

    def test_same_device_different_context_does_not_clear_completion(self):
        trace = Trace()
        trace.standard_allocations()
        trace.gemm()
        trace.call("cudaDeviceSynchronize", {"device_query_status": 0, "device": 0,
                                             "context_query_status": 0, "context": "0x7777"})
        report = self.analyze(trace.finish())
        self.assertIn("gemm-completion-not-established", codes(report))

    def test_unknown_handle_queries(self):
        trace = Trace()
        trace.gemm(observation_override={"handle": {"pointer_mode_query_status": 1, "math_mode_query_status": 1,
                                                    "stream_query_status": 1}})
        report = self.analyze(trace.finish())
        self.assertIn("handle-pointer-mode-unresolved", codes(report))
        self.assertIn("handle-math-mode-unresolved", codes(report))
        self.assertIn("handle-stream-unresolved", codes(report))

    def test_destroyed_handle_detected(self):
        trace = Trace()
        trace.standard_allocations()
        trace.call("cublasDestroy_v2", {"handle": "0x9000"})
        trace.gemm()
        report = self.analyze(trace.finish())
        self.assertIn("gemm-submitted-with-destroyed-handle", codes(report, "violations"))

    def test_host_scalars_decode_raw_f32_not_pointer_guess(self):
        report = self.analyze(self.normal())
        gemm = next(c for c in report["calls"] if c["api"] == "cublasGemmEx")
        self.assertEqual(gemm["scalar_contract"]["alpha"]["f32_value"], 1.0)
        self.assertTrue(gemm["scalar_contract"]["beta"]["is_zero"])

    def test_nonfinite_scalar_keeps_bits_without_nonstandard_json(self):
        trace = Trace()
        trace.gemm(observation_override={"scalars": {"alpha": {"host_read_bytes": 4, "bits_hex": "0000807f"}}})
        report = self.analyze(trace.finish())
        alpha = report["calls"][0]["scalar_contract"]["alpha"]
        self.assertFalse(alpha["finite"])
        self.assertIsNone(alpha["f32_value"])
        json.dumps(report, allow_nan=False)

    def test_unreadable_scalar_is_unknown(self):
        trace = Trace()
        trace.gemm(observation_override={"scalars": {"alpha": {"host_read_bytes": -1}}})
        report = self.analyze(trace.finish())
        self.assertIn("host-scalar-value-unresolved", codes(report))

    def test_negative_stride_is_unsupported_not_proven_bad_application(self):
        trace = Trace()
        args = standard_args()
        args.update(batch_count=2, stride_a=-1, stride_b=0, stride_c=0)
        trace.gemm(args, api="cublasGemmStridedBatchedEx")
        report = self.analyze(trace.finish())
        self.assertIn("unsupported-negative-stride-A", codes(report))
        self.assertEqual(report["violations"], [])

    def test_device_scalar_mode_not_host_assumed(self):
        trace = Trace()
        trace.gemm(observation_override={"handle": {"pointer_mode_query_status": 0, "pointer_mode": 1}})
        report = self.analyze(trace.finish())
        self.assertIn("device-pointer-mode-scalar-allocation-contract-unresolved", codes(report))

    def test_shape_near_miss_not_transition(self):
        trace = Trace()
        args = standard_args(512, -1)
        args["m"] = 4095
        trace.gemm(args)
        trace.gemm()
        report = self.analyze(trace.finish())
        self.assertEqual(report["expected_transition"]["observed_transitions"], [])

    def test_implicit_algorithm_not_explicit_default(self):
        trace = Trace()
        args = standard_args(512, -1)
        args["algorithm_explicit"] = False
        trace.gemm(args, api="cublasSgemm_v2")
        trace.gemm()
        report = self.analyze(trace.finish())
        self.assertEqual(report["expected_transition"]["default512_calls"], 0)

    def test_wrong_dtype_not_expected_production_half(self):
        trace = Trace()
        args = standard_args()
        args["atype"] = 0
        trace.gemm(args)
        report = self.analyze(trace.finish())
        self.assertEqual(report["expected_transition"]["production103_half256_calls"], 0)

    def test_wrong_scalar_explicitly_not_expected_scalar_contract(self):
        trace = Trace()
        trace.gemm(standard_args(512, -1))
        trace.gemm(observation_override={"scalars": {"alpha": {"host_read_bytes": 4, "bits_hex": "0000803f"},
                                                     "beta": {"host_read_bytes": 4, "bits_hex": "0000803f"}}})
        report = self.analyze(trace.finish())
        observations = report["expected_transition"]["observed_transitions"][0]["half256_observations"]
        self.assertFalse(observations["expected_scalar_handle_contract_observed"])

    def test_sync_after_half_does_not_order_before_half(self):
        trace = Trace()
        trace.standard_allocations()
        trace.gemm(standard_args(512, -1))
        trace.gemm()
        trace.sync()
        report = self.analyze(trace.finish())
        transition = report["expected_transition"]["observed_transitions"][0]
        self.assertFalse(transition["default512_device_sync_completed_before_half_submit"])
        self.assertFalse(report["expected_transition"]["expected_transition_required_present"])

    def test_post1024_pattern_counted_without_assuming_phase(self):
        trace = Trace()
        trace.standard_allocations()
        for _ in range(1024):
            trace.gemm()
        trace.sync()
        trace.gemm(standard_args(512, -1))
        trace.sync()
        trace.gemm(close=False)
        report = self.analyze(trace)
        candidates = report["expected_transition"]["post_1024_observed_half_calls_adjacent_transition_candidates"]
        self.assertEqual(len(candidates), 1)
        self.assertTrue(report["expected_transition"]["postburnin_transition_required_present"])
        self.assertEqual(candidates[0]["observed_production103_half_calls_before_default512"], 1024)
        self.assertFalse(report["trace_complete"])

    def test_async_allocator_unknown(self):
        trace = Trace()
        trace.call("cudaMallocAsync", {"size": SIZES["A"]}, result={"ptr": hex(POINTERS["A"])})
        trace.gemm()
        report = self.analyze(trace.finish())
        self.assertIn("allocator-lifetime-model-unsupported", codes(report))
        self.assertIn("operand-allocation-unobserved", codes(report))

    def test_memcpy_not_completion_fence(self):
        trace = Trace()
        trace.gemm()
        trace.call("cudaMemcpy", {"bytes": 4, "kind": 2})
        report = self.analyze(trace.finish())
        self.assertIn("gemm-completion-not-established", codes(report))

    def test_event_edges_not_inferred(self):
        trace = Trace()
        trace.gemm()
        trace.call("cudaEventRecord", {"event": "0x555", "stream": "0x0"})
        trace.call("cudaEventSynchronize", {"event": "0x555"})
        report = self.analyze(trace.finish())
        self.assertIn("event-and-stream-edge-reconstruction-not-implemented", codes(report))
        self.assertIn("gemm-completion-not-established", codes(report))

    def test_pointer_array_batch_unsupported(self):
        trace = Trace()
        trace.gemm(api="cublasGemmBatchedEx")
        report = self.analyze(trace.finish())
        self.assertIn("pointer-array-batched-gemm-unsupported", codes(report))

    def test_input_output_alias_reported(self):
        trace = Trace()
        args = standard_args()
        args["C"] = args["B"]
        trace.gemm(args)
        report = self.analyze(trace.finish())
        self.assertIn("input-output-address-envelopes-overlap", codes(report, "violations"))

    def test_exact_conversion_spans_and_consumer_link(self):
        trace = Trace()
        trace.standard_allocations()
        trace.malloc(0x40000000, 8192*256*4)
        cid = trace.conversion()
        trace.gemm()
        report = self.analyze(trace.finish())
        conversion = report["conversion_contracts"][0]
        self.assertEqual(conversion["ranges"]["src"]["bytes"], 8388608)
        self.assertEqual(conversion["ranges"]["dst"]["bytes"], 4194304)
        self.assertEqual(report["violations"], [])
        gemm = next(c for c in report["calls"] if c["api"] == "cublasGemmEx")
        candidate = gemm["conversion_producer_candidates"][0]
        self.assertEqual(candidate["conversion_call_id"], cid)
        self.assertEqual(candidate["operand"], "B")
        self.assertTrue(candidate["same_recorded_allocation_generation"])
        self.assertTrue(candidate["same_recorded_context"])

    def test_conversion_out_of_bounds(self):
        trace = Trace()
        trace.standard_allocations()
        trace.malloc(0x40000000, 10)
        trace.conversion()
        report = self.analyze(trace.finish())
        self.assertIn("conversion-exceeds-recorded-allocation", codes(report, "violations"))

    def test_conversion_unreadable_arguments_remain_unknown(self):
        trace = Trace()
        trace.conversion(argument_read_status="unreadable")
        report = self.analyze(trace.finish())
        self.assertIn("conversion-arguments-unreadable", codes(report))
        self.assertEqual(report["conversion_contracts"][0]["ranges"], {})

    def test_conversion_count_multiplication_overflow(self):
        trace = Trace()
        trace.conversion(count=module.U64_MAX)
        report = self.analyze(trace.finish())
        self.assertIn("conversion-span-overflow", codes(report, "violations"))

    def test_conversion_overlap_not_race_clearance(self):
        trace = Trace()
        trace.conversion(src=0x40000000, dst=0x40000000)
        report = self.analyze(trace.finish())
        self.assertIn("conversion-source-destination-overlap-no-race-proof", codes(report))

    def test_conversion_cross_thread_is_unresolved(self):
        trace = Trace()
        trace.standard_allocations()
        trace.malloc(0x40000000, 8192*256*4)
        trace.conversion(tid=22)
        trace.gemm()
        report = self.analyze(trace.finish())
        self.assertIn("conversion-to-gemm-dependency-not-established", codes(report))

    def test_conversion_address_reused_not_same_generation(self):
        trace = Trace()
        trace.standard_allocations()
        trace.malloc(0x40000000, 8192*256*4)
        trace.conversion()
        trace.call("cudaFree", {"ptr": hex(POINTERS["B"])})
        trace.malloc(POINTERS["B"], SIZES["B"])
        trace.gemm()
        report = self.analyze(trace.finish())
        gemm = next(c for c in report["calls"] if c["api"] == "cublasGemmEx")
        self.assertFalse(gemm["conversion_producer_candidates"][0]["same_recorded_allocation_generation"])
        self.assertIn("conversion-to-gemm-dependency-not-established", codes(report))

    def test_conversion_wrong_stream_not_cleared(self):
        trace = Trace()
        trace.standard_allocations()
        trace.malloc(0x40000000, 8192*256*4)
        trace.conversion(stream="0x9999")
        trace.gemm()
        report = self.analyze(trace.finish())
        self.assertIn("conversion-to-gemm-dependency-not-established", codes(report))

    def test_multiple_processes_not_complete_coverage(self):
        trace = self.normal()
        trace.rows[-1]["pid"] = 101
        report = self.analyze(trace)
        self.assertFalse(report["required_capture_present"])
        self.assertIn("multiple-process-trace-no-cross-process-order-proof", codes(report))

    def test_api_exit_before_enter_rejected(self):
        trace = Trace()
        trace.exit("cudaFree", 1)
        trace.enter("cudaFree", {"ptr": "0x0"})
        with self.assertRaises(module.EvidenceError):
            self.analyze(trace)

    def test_duplicate_seq_rejected(self):
        trace = self.normal()
        trace.rows[2]["seq"] = 1
        with self.assertRaises(module.EvidenceError):
            self.analyze(trace)

    def test_duplicate_call_id_rejected(self):
        trace = self.normal()
        enters = [r for r in trace.rows if r["event"] == "enter"]
        enters[1]["call_id"] = enters[0]["call_id"]
        with self.assertRaises(module.EvidenceError):
            self.analyze(trace)

    def test_duplicate_exit_rejected(self):
        trace = Trace()
        cid = trace.call("cudaSetDevice", {"device": 0})
        trace.exit("cudaSetDevice", cid)
        with self.assertRaises(module.EvidenceError):
            self.analyze(trace)

    def test_sequence_gap_retained(self):
        trace = self.normal()
        for row in trace.rows[2:]:
            row["seq"] += 1
        report = self.analyze(trace)
        self.assertFalse(report["trace_complete"])
        self.assertIn("noncontiguous-sequence", codes(report))

    def test_malformed_complete_line_rejected(self):
        with self.assertRaises(module.EvidenceError):
            self.analyze(b'{"bad":\n')

    def test_duplicate_json_key_rejected(self):
        with self.assertRaises(module.EvidenceError):
            self.analyze(b'{"schema":"a","schema":"b"}\n')

    def test_nan_rejected(self):
        with self.assertRaises(module.EvidenceError):
            self.analyze(b'{"value":NaN}\n')

    def test_overflowing_json_float_rejected(self):
        trace = Trace()
        raw = trace.bytes().replace(b'"call_id": 0', b'"unused": 1e400, "call_id": 0')
        with self.assertRaises(module.EvidenceError):
            self.analyze(raw)

    def test_float_observation_success_status_rejected(self):
        trace = Trace()
        trace.gemm(observation_override={"device_query_status": 0.0})
        with self.assertRaises(module.EvidenceError):
            self.analyze(trace)

    def test_float_timestamp_rejected(self):
        trace = Trace()
        trace.rows[0]["realtime_ns"] = float(STAMP)
        with self.assertRaises(module.EvidenceError):
            self.analyze(trace)

    def test_bool_sequence_rejected(self):
        trace = Trace()
        trace.rows[0]["seq"] = True
        with self.assertRaises(module.EvidenceError):
            self.analyze(trace)

    def test_timestamp_overflow_rejected(self):
        trace = Trace()
        trace.rows[0]["realtime_ns"] = 1 << 64
        with self.assertRaises(module.EvidenceError):
            self.analyze(trace)

    def test_line_size_limit(self):
        with patch.object(module, "MAX_LINE_BYTES", 32), self.assertRaises(module.EvidenceError):
            self.analyze(Trace())

    def test_file_size_limit(self):
        with patch.object(module, "MAX_FILE_BYTES", 256), self.assertRaises(module.EvidenceError):
            self.analyze(self.normal())

    def test_record_count_limit(self):
        with patch.object(module, "MAX_RECORDS", 2), self.assertRaises(module.EvidenceError):
            self.analyze(self.normal())

    def test_relation_work_is_bounded(self):
        with patch.object(module, "MAX_RELATION_CHECKS", 1), self.assertRaises(module.EvidenceError):
            self.analyze(self.normal())

    def test_process_count_is_bounded(self):
        trace = Trace()
        for i in range(16):
            row = trace.meta("trace_start")
            row["pid"] = 200+i
        with self.assertRaises(module.EvidenceError):
            self.analyze(trace)

    def test_nested_complexity_limit(self):
        trace = Trace()
        trace.rows[0]["result"] = {"items": list(range(4096))}
        with self.assertRaises(module.EvidenceError):
            self.analyze(trace)

    def test_invalid_observation_nested_shape_rejected(self):
        trace = Trace()
        trace.gemm(observation_override={"operands": []})
        with self.assertRaises(module.EvidenceError):
            self.analyze(trace)

    def test_pointer_overflow_rejected(self):
        trace = Trace()
        args = standard_args()
        args["A"] = "0x10000000000000000"
        trace.gemm(args)
        with self.assertRaises(module.EvidenceError):
            self.analyze(trace)

    def test_address_plus_span_overflow_reported(self):
        trace = Trace()
        args = standard_args()
        args["A"] = hex(module.U64_MAX - 2)
        trace.gemm(args)
        report = self.analyze(trace.finish())
        self.assertIn("operand-address-overflow", codes(report, "violations"))

    def test_cli_invalid_input_exit_two(self):
        with tempfile.TemporaryDirectory(prefix="runtime-contract-cli-") as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_bytes(b"not json\n")
            result = subprocess.run([sys.executable, str(MODULE), str(path)], text=True, capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "invalid-evidence")


class MatrixTests(unittest.TestCase):
    def test_full_and_half(self):
        half = module.matrix_extents(standard_args())
        full = module.matrix_extents(standard_args(512, -1))
        self.assertEqual(half["A"]["bytes"], 67108864)
        self.assertEqual(half["B"]["bytes"], 4194304)
        self.assertEqual(half["C"]["bytes"], 4194304)
        self.assertEqual(full["B"]["bytes"], 8388608)
        self.assertEqual(full["C"]["bytes"], 8388608)

    def test_padded_leading_dimension_exact_envelope(self):
        args = dict(m=3, n=5, k=4, transa=0, transb=0, lda=7, ldb=8, ldc=6, atype=2, btype=2, ctype=0)
        result = module.matrix_extents(args)
        self.assertEqual(result["A"]["bytes"], ((4-1)*7+3)*2)
        self.assertEqual(result["B"]["bytes"], ((5-1)*8+4)*2)
        self.assertEqual(result["C"]["bytes"], ((5-1)*6+3)*4)

    def test_transpose_b(self):
        args = dict(m=3, n=5, k=4, transa=1, transb=1, lda=4, ldb=5, ldc=3, atype=2, btype=2, ctype=0)
        result = module.matrix_extents(args)
        self.assertEqual((result["B"]["rows"], result["B"]["columns"]), (5, 4))
        self.assertEqual(result["B"]["bytes"], 40)

    def test_strided_batch(self):
        args = standard_args()
        args.update(batch_count=3, stride_a=4096*8192, stride_b=256*8192, stride_c=256*4096)
        result = module.matrix_extents(args)
        self.assertEqual(result["A"]["bytes"], 3*67108864)
        self.assertEqual(result["B"]["bytes"], 3*4194304)
        self.assertEqual(result["C"]["bytes"], 3*4194304)

    def test_zero_stride_broadcast(self):
        args = standard_args()
        args.update(batch_count=3, stride_a=0, stride_b=0, stride_c=256*4096)
        result = module.matrix_extents(args)
        self.assertEqual(result["A"]["bytes"], 67108864)
        self.assertEqual(result["C"]["bytes"], 12582912)

    def test_negative_stride_rejected(self):
        args = standard_args()
        args.update(batch_count=2, stride_a=-1, stride_b=0, stride_c=0)
        with self.assertRaises(module.ContractError):
            module.matrix_extents(args)

    def test_leading_dimension_too_small(self):
        args = standard_args()
        args["lda"] = 4096
        with self.assertRaises(module.ContractError):
            module.matrix_extents(args)

    def test_zero_work(self):
        args = standard_args(0)
        self.assertEqual([v["bytes"] for v in module.matrix_extents(args).values()], [0, 0, 0])

    def test_unknown_dtype_not_assumed(self):
        args = standard_args()
        args["atype"] = 999
        with self.assertRaisesRegex(module.ContractError, "unsupported-dtype"):
            module.matrix_extents(args)

    def test_extent_overflow_rejected(self):
        args = standard_args()
        args.update(batch_count=module.I64_MAX, stride_a=module.I64_MAX, stride_b=0, stride_c=0)
        with self.assertRaisesRegex(module.ContractError, "overflow"):
            module.matrix_extents(args)

    def test_missing_batch_stride_not_assumed_contiguous(self):
        args = standard_args()
        args["batch_count"] = 3
        with self.assertRaisesRegex(module.ContractError, "missing-stride"):
            module.matrix_extents(args)


if __name__ == "__main__":
    unittest.main()
