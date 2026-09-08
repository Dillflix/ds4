#!/usr/bin/env python3
"""CPU-only checks of the 039eef3 canonical GPU1 reproducer's index formulas.

This models reviewed source expressions; it does not execute CUDA, discover
runtime addresses, validate floating point, or prove compiled kernels/library
code race-free. No NVIDIA tools, subprocesses, or GPU libraries are invoked.

Default checks use complete token/group block permutations with affine
within-block bounds, rather than allocating full-sized tensor bitmaps.
--exhaustive additionally enumerates 31,457,280 pack/unpack elements using at
most a 16 MiB bitmap. This optional check is CPU-only too.
"""

import sys
import unittest


EXHAUSTIVE = "--exhaustive" in sys.argv
if EXHAUSTIVE:
    sys.argv.remove("--exhaustive")

N_TOK = 512
HALF_TOK = 256
IN_DIM = 1024
N_HEAD = 64
HEAD_DIM = 512
N_ROT = 64
N_GROUP = 8
GROUP_DIM = 4096
RANK = 1024
Q_DIM = N_HEAD * HEAD_DIM
LOW_DIM = N_GROUP * RANK
OUT_DIM = 4096
N_COMP = 128
WINDOW = 128
RATIO = 4


def align256(value):
    return (value + 255) & ~255


def group_to_token_index(gid, n_tokens, width):
    """Pack source / unpack destination, ds4_cuda.cu:13817 and :13833."""
    q, column = divmod(gid, width)
    group, token = divmod(q, n_tokens)
    return (token * N_GROUP + group) * width + column


def token_to_group_index(gid, n_tokens, width):
    q, column = divmod(gid, width)
    token, group = divmod(q, N_GROUP)
    return (group * n_tokens + token) * width + column


def parent_sizes():
    return {
        "input": N_TOK * IN_DIM * 4,
        "q_full": N_TOK * Q_DIM * 4,
        "q_split": N_TOK * Q_DIM * 4,
        "qh_full": N_TOK * Q_DIM * 2,
        "qh_split": N_TOK * Q_DIM * 2,
        "heads_full": N_TOK * Q_DIM * 4,
        "heads_split": N_TOK * Q_DIM * 4,
        "low_full": N_TOK * LOW_DIM * 4,
        "low_split": N_TOK * LOW_DIM * 4,
        "out_full": N_TOK * OUT_DIM * 4,
        "out_split": N_TOK * OUT_DIM * 4,
        "raw": N_TOK * HEAD_DIM * 4,
        "comp": N_COMP * HEAD_DIM * 4,
    }


def scratch_requests():
    # Canonical weights are resident; these requests contain no weight expansion.
    return [
        ("q-external-full", N_TOK * IN_DIM * 2),
        ("q-external-half0", HALF_TOK * IN_DIM * 2),
        ("q-external-half1", HALF_TOK * IN_DIM * 2),
        ("q-internal-half0", align256(HALF_TOK * Q_DIM * 2) + HALF_TOK * IN_DIM * 2),
        ("q-internal-half1", align256(HALF_TOK * Q_DIM * 2) + HALF_TOK * IN_DIM * 2),
        ("a-full", align256(N_GROUP * N_TOK * GROUP_DIM * 2) + N_GROUP * N_TOK * RANK * 4),
        ("b-full", N_TOK * LOW_DIM * 2),
        ("a-half0", align256(N_GROUP * HALF_TOK * GROUP_DIM * 2) + N_GROUP * HALF_TOK * RANK * 4),
        ("b-half0", HALF_TOK * LOW_DIM * 2),
        ("a-half1", align256(N_GROUP * HALF_TOK * GROUP_DIM * 2) + N_GROUP * HALF_TOK * RANK * 4),
        ("b-half1", HALF_TOK * LOW_DIM * 2),
        ("suffix-b-full", N_TOK * LOW_DIM * 2),
        ("suffix-b-half0", HALF_TOK * LOW_DIM * 2),
        ("suffix-b-half1", HALF_TOK * LOW_DIM * 2),
    ]


class AllocationLayoutTests(unittest.TestCase):
    def test_all_thirteen_persistent_tensors(self):
        sizes = parent_sizes()
        self.assertEqual(len(sizes), 13)
        self.assertEqual(sum(sizes.values()), 389_283_840)
        self.assertTrue(all(size > 0 and size % 256 == 0 for size in sizes.values()))
        self.assertEqual(sizes["raw"], 1_048_576)
        self.assertEqual(sizes["comp"], 262_144)

    def test_model_weight_ranges_and_full_allocation_accounting(self):
        weights = [(Q_DIM, IN_DIM), (LOW_DIM, GROUP_DIM), (OUT_DIM, LOW_DIM)]
        q8_sizes = [rows * (columns // 32) * 34 for rows, columns in weights]
        half_sizes = [rows * columns * 2 for rows, columns in weights]
        self.assertEqual(q8_sizes, [35_651_584] * 3)
        self.assertEqual(half_sizes, [67_108_864] * 3)
        starts = [0, q8_sizes[0] + N_HEAD * 4,
                  q8_sizes[0] + N_HEAD * 4 + q8_sizes[1]]
        ranges = [(0, q8_sizes[0]), (q8_sizes[0], starts[1])]
        ranges += [(starts[i], starts[i] + q8_sizes[i]) for i in (1, 2)]
        self.assertEqual([a[1] for a in ranges[:-1]], [a[0] for a in ranges[1:]])
        model_size = ranges[-1][1]
        self.assertEqual(model_size, 106_955_008)
        self.assertEqual(sum(half_sizes), 201_326_592)
        self.assertEqual(sum(parent_sizes().values()) + model_size +
                         sum(half_sizes) + max(size for _, size in scratch_requests()),
                         747_897_088)

    def test_all_eighteen_half_views_cover_correct_parent(self):
        bindings = {"input": "input", "q": "q_split", "qh": "qh_split",
                    "q_ref": "q_full", "heads": "heads_split",
                    "heads_ref": "heads_full", "low": "low_split",
                    "out": "out_split", "low_ref": "low_full"}
        sizes = parent_sizes()
        views = []
        for name, parent in bindings.items():
            half = sizes[parent] // 2
            pair = [(parent, i * half, half) for i in range(2)]
            for parent_name, offset, length in pair:
                self.assertLessEqual(offset, sizes[parent_name])
                self.assertLessEqual(length, sizes[parent_name] - offset)
                self.assertEqual(offset % 256, 0)
                views.append((name, parent_name, offset, length))
            self.assertEqual(pair[0][1] + pair[0][2], pair[1][1])
            self.assertEqual(pair[1][1] + pair[1][2], sizes[parent])
        self.assertEqual(len(views), 18)

    def test_host_snapshot_ranges_do_not_overlap_readback_region(self):
        low, out, slab = N_TOK * LOW_DIM, N_TOK * OUT_DIM, N_TOK * Q_DIM
        snapshots = [(low, 2 * low), (2 * low, 2 * low + out),
                     (2 * low + out, 2 * low + 2 * out)]
        self.assertEqual(snapshots[-1][1], 12_582_912)
        self.assertLessEqual(snapshots[-1][1], slab)
        self.assertGreaterEqual(snapshots[0][0], max(low, out))
        for first, second in zip(snapshots, snapshots[1:]):
            self.assertLessEqual(first[1], second[0])

    def test_canonical_dequant_source_last_byte_and_output_extent(self):
        for rows, columns in [(Q_DIM, IN_DIM), (LOW_DIM, GROUP_DIM), (OUT_DIM, LOW_DIM)]:
            blocks = columns // 32
            self.assertEqual(columns % 32, 0)
            last_gid = rows * columns - 1
            row, column = divmod(last_gid, columns)
            block, lane = divmod(column, 32)
            source_byte = (row * blocks + block) * 34 + 2 + lane
            self.assertEqual(source_byte, rows * blocks * 34 - 1)
            self.assertEqual((last_gid + 1) * 2, 67_108_864)


class KernelIndexTests(unittest.TestCase):
    def test_pack_unpack_complete_block_permutations_and_inverses(self):
        for n_tokens in (512, 256):
            for width in (GROUP_DIM, RANK):
                with self.subTest(rows=n_tokens, width=width):
                    blocks, elements = N_GROUP * n_tokens, N_GROUP * n_tokens * width
                    visited_blocks = set()
                    for block in range(blocks):
                        first = group_to_token_index(block * width, n_tokens, width)
                        last = group_to_token_index((block + 1) * width - 1, n_tokens, width)
                        self.assertEqual(first % width, 0)
                        self.assertEqual(last - first, width - 1)
                        self.assertLess(last, elements)
                        visited_blocks.add(first // width)
                        for column in (0, 1, width // 2, width - 1):
                            original = block * width + column
                            mapped = group_to_token_index(original, n_tokens, width)
                            self.assertEqual(mapped, first + column)
                            self.assertEqual(token_to_group_index(mapped, n_tokens, width), original)
                    self.assertEqual(visited_blocks, set(range(blocks)))

    def test_rms_rope_writers_cover_each_head_once(self):
        writes = [0] * HEAD_DIM
        for thread in range(256):
            for index in range(thread, HEAD_DIM - N_ROT, 256):
                writes[index] += 1
            for pair in range(thread, N_ROT // 2, 256):
                writes[HEAD_DIM - N_ROT + 2 * pair] += 1
                writes[HEAD_DIM - N_ROT + 2 * pair + 1] += 1
        self.assertEqual(writes, [1] * HEAD_DIM)
        for rows in (N_TOK, HALF_TOK):
            self.assertLess(rows * N_HEAD, 2**32)
            self.assertEqual(rows * N_HEAD * HEAD_DIM, rows * Q_DIM)

    def test_rms_reduction_every_read_initialized(self):
        # Track contributors, not floating-point values. Each stage has a
        # barrier; this verifies its literal index topology only.
        partials = [{thread, thread + 256} for thread in range(256)]
        stride = 128
        while stride:
            before = list(partials)
            for thread in range(stride):
                self.assertTrue(before[thread].isdisjoint(before[thread + stride]))
                partials[thread] = before[thread] | before[thread + stride]
            stride //= 2
        self.assertEqual(partials[0], set(range(HEAD_DIM)))

    def test_all_causal_attention_rows_and_local_global_mapping(self):
        logical_reads = 0
        for q_row0, n_q in [(0, N_TOK), (0, HALF_TOK), (HALF_TOK, HALF_TOK)]:
            for local in range(n_q):
                global_row = q_row0 + local
                self.assertLess(global_row, N_TOK)
                raw_count = min(WINDOW, global_row + 1)
                raw_start = global_row + 1 - raw_count
                comp_count = min(N_COMP, (global_row + 1) // RATIO)
                for score_row in range(raw_count + comp_count):
                    if score_row < raw_count:
                        index = raw_start + score_row
                        self.assertGreaterEqual(index, 0)
                        self.assertLessEqual(index, global_row)
                        self.assertLess(index, N_TOK)
                    else:
                        index = score_row - raw_count
                        self.assertGreaterEqual(index, 0)
                        self.assertLess(index, N_COMP)
                    if n_q == N_TOK:
                        logical_reads += 1
                last_query_index = (local * N_HEAD + N_HEAD - 1) * HEAD_DIM + HEAD_DIM - 1
                self.assertLess(last_query_index, n_q * Q_DIM)
        self.assertEqual(logical_reads, 90_048)

    def test_attention_tile_initialization_and_float4_head_coverage(self):
        for tile_rows in range(1, 5):
            producers = [0] * (tile_rows * 128)
            for thread in range(256):
                for offset in range(thread, tile_rows * 128, 256):
                    producers[offset] += 1
            self.assertEqual(producers, [1] * len(producers))
            consumers = {row * 128 + lane + plane for row in range(tile_rows)
                         for lane in range(32) for plane in (0, 32, 64, 96)}
            self.assertEqual(consumers, set(range(len(producers))))
        per_head = [lane + plane for lane in range(32) for plane in (0, 32, 64, 96)]
        self.assertEqual(sorted(per_head), list(range(HEAD_DIM // 4)))
        self.assertEqual(4 * 128 * 16, 8192)

    def test_inverse_rope_pairs_unique_and_in_range(self):
        for rows in (N_TOK, HALF_TOK):
            count = rows * N_HEAD * (N_ROT // 2)
            self.assertLess(count, 2**32)
            # Every token/head block owns a distinct contiguous tail of
            # 32 pairs. Check all block endpoints and all within-head pairs.
            for row in range(rows * N_HEAD):
                first = row * HEAD_DIM + HEAD_DIM - N_ROT
                last = first + N_ROT - 1
                self.assertLess(last, rows * Q_DIM)
                self.assertEqual(last, (row + 1) * HEAD_DIM - 1)
            columns = [HEAD_DIM - N_ROT + pair * 2 + side
                       for pair in range(N_ROT // 2) for side in (0, 1)]
            self.assertEqual(columns, list(range(HEAD_DIM - N_ROT, HEAD_DIM)))

    def test_gemm_last_element_extents(self):
        for rows in (N_TOK, HALF_TOK):
            # Column-major matrices as passed to cuBLAS; A transposed.
            q_weight_end = ((Q_DIM - 1) * IN_DIM + IN_DIM) * 2
            q_input_end = ((rows - 1) * IN_DIM + IN_DIM) * 2
            q_output_end = ((rows - 1) * Q_DIM + Q_DIM) * 2
            self.assertEqual(q_weight_end, 67_108_864)
            self.assertEqual(q_input_end, 1_048_576 if rows == 512 else 524_288)
            self.assertEqual(q_output_end, 33_554_432 if rows == 512 else 16_777_216)

            a_weight_stride = RANK * GROUP_DIM
            a_input_stride = rows * GROUP_DIM
            a_output_stride = rows * RANK
            a_weight_last = (N_GROUP - 1) * a_weight_stride + (RANK - 1) * GROUP_DIM + GROUP_DIM - 1
            a_input_last = (N_GROUP - 1) * a_input_stride + (rows - 1) * GROUP_DIM + GROUP_DIM - 1
            a_output_last = (N_GROUP - 1) * a_output_stride + (rows - 1) * RANK + RANK - 1
            self.assertEqual((a_weight_last + 1) * 2, 67_108_864)
            self.assertEqual((a_input_last + 1) * 2, 33_554_432 if rows == 512 else 16_777_216)
            self.assertEqual((a_output_last + 1) * 4, 16_777_216 if rows == 512 else 8_388_608)
            for stride, matrix_values in [(a_weight_stride, RANK * GROUP_DIM),
                                          (a_input_stride, rows * GROUP_DIM),
                                          (a_output_stride, rows * RANK)]:
                self.assertGreaterEqual(stride, matrix_values)

            b_weight_end = ((OUT_DIM - 1) * LOW_DIM + LOW_DIM) * 2
            b_input_end = ((rows - 1) * LOW_DIM + LOW_DIM) * 2
            b_output_end = ((rows - 1) * OUT_DIM + OUT_DIM) * 4
            self.assertEqual(b_weight_end, 67_108_864)
            self.assertEqual(b_input_end, 8_388_608 if rows == 512 else 4_194_304)
            self.assertEqual(b_output_end, 8_388_608 if rows == 512 else 4_194_304)


class ArenaLayoutTests(unittest.TestCase):
    def test_requested_generations_grow_only_before_b_suffix(self):
        capacity = generation = 0
        growth = []
        for name, request in scratch_requests():
            if request > capacity:
                capacity = request
                generation += 1
                growth.append((name, capacity))
            if name.startswith("suffix-"):
                self.assertEqual(generation, 3)
                self.assertEqual(capacity, 50_331_648)
        self.assertEqual(growth, [("q-external-full", 1_048_576),
                                  ("q-internal-half0", 17_301_504),
                                  ("a-full", 50_331_648)])

    def test_scratch_subregions_and_following_b_conversion(self):
        qhalf_end = HALF_TOK * Q_DIM * 2
        input_start = align256(qhalf_end)
        self.assertGreaterEqual(input_start, qhalf_end)
        self.assertEqual(input_start + HALF_TOK * IN_DIM * 2, 17_301_504)
        for rows in (N_TOK, HALF_TOK):
            heads_end = N_GROUP * rows * GROUP_DIM * 2
            packed_low_start = align256(heads_end)
            packed_low_end = packed_low_start + N_GROUP * rows * RANK * 4
            b_input_end = rows * LOW_DIM * 2
            self.assertGreaterEqual(packed_low_start, heads_end)
            self.assertLessEqual(b_input_end, packed_low_start)
            self.assertLessEqual(packed_low_end, 50_331_648)


class OptionalExhaustiveTests(unittest.TestCase):
    @unittest.skipUnless(EXHAUSTIVE, "pass --exhaustive for full CPU element enumeration")
    def test_every_pack_unpack_element(self):
        checked = 0
        for rows in (N_TOK, HALF_TOK):
            for width in (GROUP_DIM, RANK):
                count = rows * N_GROUP * width
                seen = bytearray(count)
                for gid in range(count):
                    mapped = group_to_token_index(gid, rows, width)
                    if not 0 <= mapped < count or seen[mapped]:
                        self.fail(f"out-of-range/duplicate rows={rows} width={width} gid={gid}")
                    seen[mapped] = 1
                self.assertEqual(seen.count(1), count)
                checked += count
        self.assertEqual(checked, 31_457_280)


if __name__ == "__main__":
    unittest.main()
