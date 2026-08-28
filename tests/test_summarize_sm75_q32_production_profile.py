#!/usr/bin/env python3
"""Unit tests for the integrated SM75 Q32 production summarizer."""

from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "speed-bench" / "summarize-sm75-q32-production-profile.py"
SPEC = importlib.util.spec_from_file_location("q32_profile_summary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(
    name: str,
    *,
    kind: str = "kernel",
    layer: int = 0,
    partner: str = "",
    attention_rows: str = "",
) -> dict[str, str]:
    return {
        "name": name,
        "kind": kind,
        "layer_range": f"stage=0/layer={layer}/part=test",
        "stage_range": "",
        "partner_range": partner,
        "attention_rows_range": attention_rows,
        "handoff_range": "",
    }


class ClassifierTests(unittest.TestCase):
    def test_routed_formats_are_role_and_layer_aware(self) -> None:
        kernel = "moe_gate_up_mid_sm75_q32_tile8_kernel<512>"
        self.assertEqual(MODULE.family_of(row(kernel, layer=6)), "q3a4_gate_up")
        self.assertEqual(MODULE.family_of(row(kernel, layer=7)), "q4_32_gate_up")
        self.assertEqual(
            MODULE.family_of(row("moe_down_sm75_q4_32_tile_kernel<16>")),
            "q4_32_down",
        )

    def test_integrated_critical_path_families_are_separate(self) -> None:
        cases = {
            "indexer_scores_streaming64_native_kernel": "indexer_score",
            "indexed_topk_monolithic_kernel": "indexer_topk",
            "attention_indexed_mixed_heads8_online_kernel": "attention_indexed",
            "attention_decode_mixed_heads8_online_kernel": "attention_mixed",
            "matmul_q8_0_mma_sm75_exact_kernel": "dense_q8_native",
            "cutlass::Kernel2<turing_s1688gemm>": "local_fp16_gemm",
            "rms_norm_kernel": "norm_hyperconnection",
            "compressor_prefill_kernel": "compressor",
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(MODULE.family_of(row(name)), expected)

    def test_cross_device_work_is_not_folded_into_local_compute(self) -> None:
        partner_range = "label=attn_output_b/tokens=512"
        self.assertEqual(
            MODULE.family_of(row("turing_s1688gemm", partner=partner_range)),
            "partner_t256_cublas",
        )
        self.assertEqual(
            MODULE.family_of(
                row("Memcpy DtoD", kind="memcpy", partner=partner_range)
            ),
            "partner_t256_memcpy",
        )
        self.assertEqual(
            MODULE.family_of(
                row("Memcpy DtoD", kind="memcpy", attention_rows="stage=0")
            ),
            "attention_row_split_memcpy",
        )


if __name__ == "__main__":
    unittest.main()
