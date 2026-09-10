# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a private E4M3 grouped-attention scheduling candidate.

The original per-head arithmetic, FP32 numerator/max/sum workspace and
native input validation remain intact. Candidates change CTA grouping,
loop scheduling or address-equivalent loads. The builder installs no route.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"Expected one source anchor: {old}")
    return source.replace(old, new)


def prefetch_values(source: str) -> str:
    """Fill a disjoint V panel with idle QK warps before the existing barrier."""
    start = source.index("__launch_bounds__(kGroupedVerifyThreads, 1) void ")
    end = source.index(
        "void flash_attention_grouped_verify_e5m2_combine_kernel(", start
    )
    partial = source[start:end]
    value_argument = partial.index("        shared_kv, v_cache, page_ids,")
    load_start = partial.rfind("    load_xqa_tc_kv_panel<", 0, value_argument)
    barrier = "    __syncthreads();"
    load_end = partial.index(barrier, value_argument) + len(barrier)
    load = partial[load_start:load_end]
    load = load[: load.rfind(barrier)]
    load = load.replace("shared_kv", "shared_values")
    load = load.replace("kGroupedVerifyThreads", "kValueLoadThreads")
    load = load.replace("idx = tid +", "idx = value_load_tid +")
    load = replace_once(
        load,
        "v_block_stride, v_token_stride, v_head_stride, 0);",
        "v_block_stride, v_token_stride, v_head_stride, 0, value_load_tid);",
    )
    partial = partial[:load_start] + partial[load_end:]
    marker = "  constexpr int kResidualStride = kGroupedVerifyProbStride;"
    partial = replace_once(
        partial,
        marker,
        marker + "\n  __half* shared_values = shared_prob_residual + "
        "kGroupedVerifyRows * kResidualStride;",
    )
    qk = (
        "    grouped_verify_qk<COMPENSATE_P>(shared_q, shared_kv, shared_scores,\n"
        "                                    qk_scale, active_m_tiles);"
    )
    partial = replace_once(
        partial,
        qk,
        qk + "\n    if (warp_id >= kGroupedVerifyQKWarps) {\n"
        "      constexpr int kValueLoadThreads = kGroupedVerifyThreads - "
        "kGroupedVerifyQKWarps * kWarpSize;\n"
        "      const int value_load_tid = tid - kGroupedVerifyQKWarps * kWarpSize;\n"
        + load
        + "    }\n",
    )
    partial = replace_once(
        partial,
        "shared_kv + k_offset * kGroupedVerifyKVStride + d_tile * 16,",
        "shared_values + k_offset * kGroupedVerifyKVStride + d_tile * 16,",
    )
    source = source[:start] + partial + source[end:]
    source = replace_once(
        source,
        "kGroupedVerifyRows * kGroupedVerifyProbStride * sizeof(__half);",
        "kGroupedVerifyRows * kGroupedVerifyProbStride * sizeof(__half) +\n"
        "      kGroupedVerifyBlockN * kGroupedVerifyKVStride * sizeof(__half);",
    )
    source = replace_once(
        source,
        "static_assert(kCompensatedSmemBytes <= 64 * 1024,",
        "static_assert(kCompensatedSmemBytes <= 96 * 1024,",
    )
    source = replace_once(
        source,
        '"compensated P must fit the SM70 shared-memory budget");',
        '"compensated P and prefetched V must fit the SM70 budget");\n'
        "  TORCH_CHECK(properties->sharedMemPerBlockOptin >= kCompensatedSmemBytes,\n"
        '              "V prefetch exceeds device opt-in shared memory");',
    )
    return source


def prefetch_keys(source: str, key_load_warps: int) -> str:
    """Overlap the next K panel with softmax using dedicated load warps."""
    start = source.index("__launch_bounds__(kGroupedVerifyThreads, 1) void ")
    end = source.index(
        "void flash_attention_grouped_verify_e5m2_combine_kernel(", start
    )
    partial = source[start:end]
    loop_start = partial.rindex("  for (int tile_start = split_start;")
    loop = partial[loop_start:]
    barrier = "    __syncthreads();"
    load_start = loop.index("    load_xqa_tc_kv_panel<")
    load_end = loop.index(barrier, load_start) + len(barrier)
    load = loop[load_start:load_end]
    # The first panel is loaded by the whole CTA. Subsequent panels are ready
    # at the previous tile's softmax barrier, before PV consumes disjoint V.
    loop = (
        loop[:load_start]
        + "    if (tile_start == split_start) {\n"
        + load
        + "\n    }\n"
        + loop[load_end:]
    )
    load = load[: load.rfind(barrier)]
    load = load.replace("kGroupedVerifyThreads", "kKeyLoadThreads")
    load = load.replace("idx = tid +", "idx = key_load_tid +")
    load = load.replace("valid_k_rows", "next_k_rows")
    load = load.replace("tile_page_offset", "next_page_offset")
    load = replace_once(
        load,
        "k_block_stride, k_token_stride, k_head_stride, 0);",
        "k_block_stride, k_token_stride, k_head_stride, 0, key_load_tid);",
    )
    softmax_start = loop.index(
        "#pragma unroll\n      for (int row = warp_id; row < kGroupedVerifyRows;"
    )
    softmax_end = loop.index("      __syncthreads();", softmax_start)
    softmax = loop[softmax_start:softmax_end]
    softmax = replace_once(
        softmax, "row += kGroupedVerifyWarps)", "row += kSoftmaxWarps)"
    )
    replacement = (
        f"      constexpr int kSoftmaxWarps = {16 - key_load_warps};\n"
        "      if (warp_id < kSoftmaxWarps) {\n"
        + softmax
        + "      } else if (tile_start + kGroupedVerifyBlockN < split_end) {\n"
        "        constexpr int kKeyLoadThreads = "
        "kGroupedVerifyThreads - kSoftmaxWarps * kWarpSize;\n"
        "        const int key_load_tid = tid - kSoftmaxWarps * kWarpSize;\n"
        "        const int next_k_rows = min(kGroupedVerifyBlockN, "
        "split_end - tile_start - kGroupedVerifyBlockN);\n"
        "        const int next_page_offset = tile_page_offset + "
        "kGroupedVerifyBlockN;\n" + load + "      }\n"
    )
    loop = loop[:softmax_start] + replacement + loop[softmax_end:]
    partial = partial[:loop_start] + loop
    return source[:start] + partial + source[end:]


def reuse_pv_values(source: str) -> str:
    """Interchange independent M tiles to reuse raw/scaled V fragments.

    Each accumulator still receives main0, residual0, main16, residual16 in that
    order, followed by one N32 online-state update. The six-head/16-warp layout
    assigns the same D tile to a warp for all three M tiles.
    """
    start = source.index("__launch_bounds__(kGroupedVerifyThreads, 1) void ")
    end = source.index(
        "void flash_attention_grouped_verify_e5m2_combine_kernel(", start
    )
    partial = source[start:end]
    begin = partial.index(
        "#pragma unroll\n    for (int fragment_idx = 0; "
        "fragment_idx < kGroupedVerifyOutputTilesPerWarp;"
    )
    finish = partial.index("    __syncthreads();\n  }", begin)
    replacement = r"""    static_assert(COMPENSATE_P && kGroupedVerifyWarps == 16,
                  "PV reuse is isolated to six-head compensated E4M3");
    volta::fragment<volta::accumulator, 16, 16, 16, float>
        tile_fragments[kGroupedVerifyOutputTilesPerWarp];
#pragma unroll
    for (int i = 0; i < kGroupedVerifyOutputTilesPerWarp; ++i)
      volta::fill_fragment(tile_fragments[i], 0.0f);
    const int d_tile = warp_id;
#pragma unroll
    for (int k_offset = 0; k_offset < kGroupedVerifyBlockN; k_offset += 16) {
      volta::fragment<volta::matrix_b, 16, 16, 16, half, volta::row_major>
          value_fragment;
      volta::load_matrix_sync(
          value_fragment,
          shared_values + k_offset * kGroupedVerifyKVStride + d_tile * 16,
          kGroupedVerifyKVStride);
      auto residual_value_fragment = value_fragment;
#pragma unroll
      for (int i = 0; i < value_fragment.num_elements / 2; ++i) {
        union {
          uint32_t bits;
          __half2 pair;
        } packed_value;
        packed_value.bits = value_fragment.x[i];
        packed_value.pair =
            __hmul2(packed_value.pair, __float2half2_rn(1.0f / 2048.0f));
        residual_value_fragment.x[i] = packed_value.bits;
      }
#pragma unroll
      for (int fragment_idx = 0;
           fragment_idx < kGroupedVerifyOutputTilesPerWarp; ++fragment_idx) {
        const int m_tile = fragment_idx;
        if ((active_m_tiles & (1 << m_tile)) == 0) continue;
        volta::fragment<volta::matrix_a, 16, 16, 16, half, volta::row_major>
            probability_fragment;
        volta::load_matrix_sync(
            probability_fragment,
            shared_probs + m_tile * 16 * kGroupedVerifyProbStride + k_offset,
            kGroupedVerifyProbStride);
        volta::mma_sync(tile_fragments[fragment_idx], probability_fragment,
                        value_fragment, tile_fragments[fragment_idx]);
        volta::load_matrix_sync(
            probability_fragment,
            shared_prob_residual + m_tile * 16 * kResidualStride + k_offset,
            kResidualStride);
        volta::mma_sync(tile_fragments[fragment_idx], probability_fragment,
                        residual_value_fragment, tile_fragments[fragment_idx]);
      }
    }
#pragma unroll
    for (int fragment_idx = 0;
         fragment_idx < kGroupedVerifyOutputTilesPerWarp; ++fragment_idx) {
      const int m_tile = fragment_idx;
      if ((active_m_tiles & (1 << m_tile)) == 0) continue;
      grouped_verify_add_output_tile(output_fragments[fragment_idx],
                                     tile_fragments[fragment_idx],
                                     smem.row_scale, m_tile * 16);
    }
"""
    partial = partial[:begin] + replacement + partial[finish:]
    return source[:start] + partial + source[end:]


def pair_qk_products(source: str) -> str:
    """Produce two independent K16 products before their ordered corrections.

    The dot products retain their own zero-initialized accumulators. Correction
    consumes the first product and then the second, in the original K16 order.
    This isolates instruction scheduling from a change in reduction arithmetic.
    """
    start = source.index("__device__ __forceinline__ void grouped_verify_qk(")
    end = source.index(
        "__device__ __forceinline__ void grouped_verify_scale_output_fragment(",
        start,
    )
    qk = source[start:end]
    qk = replace_once(
        qk,
        "k_offset < kGroupedVerifyHeadDim; k_offset += 16)",
        "k_offset < kGroupedVerifyHeadDim; k_offset += (COMPENSATE ? 32 : 16))",
    )
    product = (
        "      volta::mma_sync(tile_fragment, q_fragment, k_fragment, tile_fragment);"
    )
    qk = replace_once(
        qk,
        product,
        product
        + r"""
      volta::fragment<volta::accumulator, 16, 16, 16, float> next_tile_fragment;
      volta::fill_fragment(next_tile_fragment, 0.0f);
      volta::load_matrix_sync(
          q_fragment,
          shared_q + m_tile * 16 * kGroupedVerifyQStride + k_offset + 16,
          kGroupedVerifyQStride);
      volta::load_matrix_sync(
          k_fragment,
          shared_k + n_tile * 16 * kGroupedVerifyKVStride + k_offset + 16,
          kGroupedVerifyKVStride);
      volta::mma_sync(next_tile_fragment, q_fragment, k_fragment,
                      next_tile_fragment);
""",
    )
    correction = r"""#pragma unroll
      for (int i = 0; i < score_fragment.num_elements; ++i) {
        const float y = __fsub_rn(tile_fragment.x[i], correction[i]);
        const float sum = __fadd_rn(score_fragment.x[i], y);
        correction[i] = __fsub_rn(__fsub_rn(sum, score_fragment.x[i]), y);
        score_fragment.x[i] = sum;
      }"""
    qk = replace_once(
        qk,
        correction,
        correction
        + "\n"
        + correction.replace("tile_fragment.x", "next_tile_fragment.x"),
    )
    return source[:start] + qk + source[end:]


def retain_fp32_output(source: str) -> str:
    """Expose the final FP32 accumulator for an independent numerical audit."""
    symbol = source.index("void flash_attention_grouped_verify_e5m2_combine_kernel(")
    start = source.rfind("template <", 0, symbol)
    end = source.index("\ntemplate <", symbol)
    combine = source[start:end]
    for old, new in (
        ("__half* __restrict__ out,", "float* __restrict__ out,"),
        ("__float2half_rn(0.0f)", "0.0f"),
        ("__float2half_rn(accumulator)", "accumulator"),
    ):
        combine = replace_once(combine, old, new)
    source = source[:start] + combine + source[end:]
    start = source.index("at::Tensor flash_attention_grouped_e4m3_fp32_paged(")
    host = source[start:]
    for old, new in (
        ("out.scalar_type() == at::kHalf", "out.scalar_type() == at::kFloat"),
        ("output must be contiguous FP16", "audit output must be contiguous FP32"),
        (
            "reinterpret_cast<__half*>(out.data_ptr())",
            "reinterpret_cast<float*>(out.data_ptr())",
        ),
    ):
        host = replace_once(host, old, new)
    return source[:start] + host


def accumulate_qk_fp64(source: str) -> str:
    """Arithmetic experiment: sum the unchanged K16 products with FP64 adds.

    This is not an exact scheduling optimization. Even if a sampled output
    agrees, independent pre-cast reference and model audits remain mandatory.
    """
    start = source.index("template <bool COMPENSATE = false>")
    end = source.index("void grouped_verify_scale_output_fragment(", start)
    qk = source[start:end]
    qk = replace_once(qk, "float correction[8] = {};", "double wide_sum[8] = {};")
    qk = replace_once(
        qk,
        """      // E4M3 x FP16 products fit comfortably in FP32, but a D256 Tensor
      // Core accumulation can still lose low bits. Sum short K16 products
      // with compensated FP32 additions; explicit RN operations preserve
      // the correction under the standard fast-math build.""",
        """      // Arithmetic probe: retain each original FP32 K16 Tensor Core
      // product, but sum those products in FP64 before the final FP32 cast.
      // This needs an independent reference audit; it is not bit-exact by
      // construction and is not admitted by an operator timing result.""",
    )
    qk = replace_once(
        qk,
        """        const float y = __fsub_rn(tile_fragment.x[i], correction[i]);
        const float sum = __fadd_rn(score_fragment.x[i], y);
        correction[i] = __fsub_rn(__fsub_rn(sum, score_fragment.x[i]), y);
        score_fragment.x[i] = sum;""",
        """        wide_sum[i] = __dadd_rn(
            wide_sum[i], static_cast<double>(tile_fragment.x[i]));""",
    )
    qk = replace_once(
        qk,
        "    score_fragment.x[i] *= qk_scale;",
        """    if constexpr (COMPENSATE) {
      score_fragment.x[i] = __double2float_rn(wide_sum[i]) * qk_scale;
    } else {
      score_fragment.x[i] *= qk_scale;
    }""",
    )
    return source[:start] + qk + source[end:]


def register_softmax_state(partial: str) -> str:
    """Keep each warp's three online rows private until the final publication.

    Every lane consumes the same broadcast tile maximum/sum and runs the
    original N32 update. Only row_scale is shared with PV between tiles.
    """
    marker = "  // Recompute QK for the conservative path"
    partial = replace_once(
        partial,
        marker,
        "  static_assert(!TWO_PASS && kGroupedVerifyWarps == 16 &&\n"
        '      kGroupedVerifyRows == 48, "Three fixed online rows per warp");\n'
        "  float online_max[3] = {kXQANegInf, kXQANegInf, kXQANegInf};\n"
        "  float online_sum[3] = {};\n" + marker,
    )
    begin = partial.index(marker)
    end = partial.index("  // The compute buffers are dead.", begin)
    loop = partial[begin:end]
    loop = replace_once(
        loop,
        "const float old_max = smem.row_max[row];",
        "const float old_max = online_max[row / kGroupedVerifyWarps];",
    )
    loop = replace_once(
        loop,
        "        // Finish every lane's shared-state reads before lane 0 "
        "overwrites the\n"
        """        // online maximum. Shuffle synchronization does not order memory.
        __syncwarp();
        if (lane_id == 0) {
          if (tile_sum > 0.0f) {
            smem.row_sum[row] = smem.row_sum[row] * exp_diff + tile_sum;
            smem.row_max[row] = new_max;
          }
          smem.row_scale[row] = exp_diff;
        }""",
        "        // Each lane owns an identical copy; no shared maximum "
        "is overwritten.\n"
        """        const int local_row = row / kGroupedVerifyWarps;
        if (tile_sum > 0.0f) {
          online_sum[local_row] = online_sum[local_row] * exp_diff + tile_sum;
          online_max[local_row] = new_max;
        }
        if (lane_id == 0) smem.row_scale[row] = exp_diff;""",
    )
    return (
        partial[:begin]
        + loop
        + """  // Publish final row statistics before the existing output barrier.
  if (lane_id == 0) {
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      smem.row_max[warp_id + i * kGroupedVerifyWarps] = online_max[i];
      smem.row_sum[warp_id + i * kGroupedVerifyWarps] = online_sum[i];
    }
  }

"""
        + partial[end:]
    )


def specialize_full_q8(
    source: str, visible_tiles: bool, register_state: bool = False
) -> str:
    """Remove variable-Q branches only when all eight query rows exist.

    Per-row GPU lengths remain authoritative. The optional all-visible branch
    requires the complete N32 tile to precede the minimum of all eight lengths;
    padding, rejected rows and the causal tail use the original visibility code.
    """
    old_name = "flash_attention_grouped_verify_e5m2_partial_kernel"
    new_name = "flash_attention_grouped_verify_e4m3_full_q8_kernel"
    start = source.index(
        "template <int MAX_QUERY_TOKENS, bool TWO_PASS, int PAGE_BLOCK_SIZE"
    )
    end = source.index("template <int MAX_QUERY_TOKENS, bool SINGLE_QUERY", start)
    partial = source[start:end].replace(old_name, new_name)
    partial = replace_once(
        partial, "const int query_len,", "const int runtime_query_len,"
    )
    partial = replace_once(
        partial,
        "  using Traits = GroupedVerifyTraits<MAX_QUERY_TOKENS>;",
        "  static_assert(MAX_QUERY_TOKENS == 8 && COMPENSATE_P &&\n"
        '      ROW_SEQLENS && !SPARSE_PAGE4, "Full q8 compensated contract");\n'
        "  if (runtime_query_len != 8) return;\n"
        "  constexpr int query_len = 8;\n"
        "  using Traits = GroupedVerifyTraits<MAX_QUERY_TOKENS>;",
    )
    if register_state:
        partial = register_softmax_state(partial)
    if visible_tiles:
        loop_start = partial.index("  // Recompute QK for the conservative path")
        partial = (
            partial[:loop_start] + "  int minimum_visible_length = row_lengths[0];\n"
            "#pragma unroll\n"
            "  for (int i = 1; i < 8; ++i)\n"
            "    minimum_visible_length =\n"
            "        min(minimum_visible_length, row_lengths[i]);\n"
            + partial[loop_start:]
        )
        begin = partial.index(
            "#pragma unroll\n      for (int row = warp_id;", loop_start
        )
        finish = partial.index("      __syncthreads();", begin)
        original = partial[begin:finish]
        visible = original
        a = visible.index("        const bool visible =")
        b = visible.index("        const float score =", a)
        visible = visible[:a] + "        constexpr bool visible = true;\n" + visible[b:]
        partial = (
            partial[:begin] + "      if (tile_start + kGroupedVerifyBlockN <=\n"
            "          minimum_visible_length) {\n"
            + visible
            + "      } else {\n"
            + original
            + "      }\n"
            + partial[finish:]
        )
    source = source[:end] + partial + source[end:]
    host_start = source.index("  auto kernel =", source.index("at::Tensor "))
    host_end = source.index("  constexpr int kCompensatedSmemBytes", host_start)
    selection = source[host_start:host_end].replace("auto kernel =", "kernel =", 1)
    selection = selection.replace(old_name, new_name)
    return (
        source[:host_end]
        + "  if (q.size(0) == 8) {\n"
        + selection
        + "  }\n"
        + source[host_end:]
    )


def qk_head_rows(source: str) -> str:
    """Audit M8/N32 per head; WMMA shape equality is not assumed.

    Q is staged as six groups of eight rows. Scores are stored back to the
    parent's token/head order, so softmax, PV and merge retain their layout.
    """
    start = source.index("template <bool COMPENSATE = false>")
    end = source.index(
        "__device__ __forceinline__ void grouped_verify_scale_output_fragment(",
        start,
    )
    qk = source[start:end].replace("grouped_verify_qk(", "grouped_verify_qk_head(")
    qk = qk.replace("16, 16, 16", "8, 32, 16")
    a = qk.index("  const int m_tile =")
    b = qk.index("  volta::fragment<", a)
    qk = qk[:a] + "  const int head = warp_id;\n" + qk[b:]
    qk = replace_once(
        qk,
        "shared_q + m_tile * 16 * kGroupedVerifyQStride + k_offset",
        "shared_q + head * 8 * kGroupedVerifyQStride + k_offset",
    )
    qk = replace_once(
        qk,
        "shared_k + n_tile * 16 * kGroupedVerifyKVStride + k_offset",
        "shared_k + k_offset",
    )
    qk = replace_once(
        qk,
        """      shared_scores + m_tile * 16 * kGroupedVerifyScoreStride + n_tile * 16,
      score_fragment, kGroupedVerifyScoreStride, volta::mem_row_major);""",
        """      shared_scores + head * kGroupedVerifyScoreStride,
      score_fragment, 6 * kGroupedVerifyScoreStride, volta::mem_row_major);""",
    )
    source = source[:end] + qk + source[end:]
    start = source.index("void flash_attention_grouped_verify_e4m3_full_q8_kernel(")
    end = source.index("template <int MAX_QUERY_TOKENS, bool SINGLE_QUERY", start)
    partial = source[start:end]
    at = partial.index("  if (tid < kGroupedVerifyRows) {")
    setup = partial[:at]
    old = "shared_q_vec[row * kSharedQVecsPerRow + vec_col]"
    assert setup.count(old) == 2
    setup = setup.replace(
        old,
        "shared_q_vec[(local_head * 8 + token_idx) *\n"
        "                   kSharedQVecsPerRow + vec_col]",
    )
    partial = setup + partial[at:]
    partial = replace_once(
        partial, "grouped_verify_qk<COMPENSATE_P>(", "grouped_verify_qk_head<true>("
    )
    return source[:start] + partial + source[end:]


def pipeline_qk_operands(source: str) -> str:
    """Rotate Q/K fragment loads before the preceding K16 correction.

    Current operands are dead after MMA. Their registers can hold the next
    operands while the original FP32 correction consumes the current product.
    """
    start = source.index("template <bool COMPENSATE = false>")
    end = source.index(
        "__device__ __forceinline__ void grouped_verify_scale_output_fragment(",
        start,
    )
    qk = source[start:end]
    begin = qk.index("    volta::load_matrix_sync(")
    finish = qk.index("    if constexpr (COMPENSATE)", begin)
    loads = qk[begin:finish]
    qk = qk[:begin] + qk[finish:]
    before_loop = qk.index("#pragma unroll")
    qk = qk[:before_loop] + loads.replace("k_offset", "0") + qk[before_loop:]
    marker = (
        "      volta::mma_sync(tile_fragment, q_fragment, k_fragment, tile_fragment);"
    )
    qk = replace_once(
        qk,
        marker,
        marker
        + "\n      if (k_offset + 16 < kGroupedVerifyHeadDim) {\n"
        + loads.replace("k_offset", "(k_offset + 16)")
        + "      }",
    )
    marker = (
        "      volta::mma_sync(score_fragment, q_fragment, k_fragment, score_fragment);"
    )
    qk = replace_once(
        qk,
        marker,
        marker
        + "\n      if (k_offset + 16 < kGroupedVerifyHeadDim) {\n"
        + loads.replace("k_offset", "(k_offset + 16)")
        + "      }",
    )
    return source[:start] + qk + source[end:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--head-groups", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--qk-unroll", type=int, choices=(1, 2, 4, 8, 16))
    parser.add_argument("--vector-load", action="store_true")
    parser.add_argument("--page-specialize", action="store_true")
    parser.add_argument("--prefetch-v", action="store_true")
    parser.add_argument("--prefetch-k", action="store_true")
    parser.add_argument("--prefetch-k-warps", type=int, choices=(4, 8), default=8)
    parser.add_argument("--reuse-pv-values", action="store_true")
    parser.add_argument("--qk-paired-products", action="store_true")
    parser.add_argument("--qk-fp64-sum", action="store_true")
    parser.add_argument("--specialize-full-q8", action="store_true")
    parser.add_argument("--all-visible-tiles", action="store_true")
    parser.add_argument("--register-softmax-state", action="store_true")
    parser.add_argument("--qk-head-rows", action="store_true")
    parser.add_argument("--qk-operand-pipeline", action="store_true")
    parser.add_argument("--diagnostic-output-fp32", action="store_true")
    parser.add_argument("--splits", type=int, choices=(80, 160, 320), default=80)
    parser.add_argument("--grouped-only", action="store_true")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if args.prefetch_v and args.head_groups != 1:
        parser.error("--prefetch-v currently requires --head-groups 1")
    if args.prefetch_k and not (args.prefetch_v and args.grouped_only):
        parser.error("--prefetch-k requires --prefetch-v and --grouped-only")
    if args.reuse_pv_values and not (
        args.head_groups == 1
        and args.prefetch_v
        and args.grouped_only
        and not args.prefetch_k
    ):
        parser.error("--reuse-pv-values needs six heads, V prefetch and grouped-only")
    if args.specialize_full_q8 and not (
        args.grouped_only and args.head_groups == 1 and args.reuse_pv_values
    ):
        parser.error("Full-q8 specialization requires the six-head PV-reuse path")
    if args.all_visible_tiles and not args.specialize_full_q8:
        parser.error("Visible-tile specialization requires --specialize-full-q8")
    if args.register_softmax_state and not args.specialize_full_q8:
        parser.error("Register softmax state requires --specialize-full-q8")
    if args.qk_head_rows and not (
        args.specialize_full_q8
        and not args.register_softmax_state
        and not args.qk_paired_products
        and not args.qk_fp64_sum
    ):
        parser.error("Audit the M8/N32 QK shape independently on fixed q8")
    if args.qk_operand_pipeline and not (
        args.head_groups == 1
        and args.grouped_only
        and not args.qk_paired_products
        and not args.qk_fp64_sum
        and not args.qk_head_rows
        and not args.register_softmax_state
    ):
        parser.error("Audit QK operand rotation independently on six-head groups")
    if args.qk_fp64_sum and not (
        args.grouped_only and args.head_groups == 1 and not args.qk_paired_products
    ):
        parser.error("FP64 K16 accumulation needs an isolated six-head grouped build")
    if args.diagnostic_output_fp32 and not args.grouped_only:
        parser.error("Pre-cast FP32 output is an isolated grouped-operator audit")
    root = Path(__file__).resolve().parents[2] / "flash-attention-v100"
    original = root / "kernel/flash_decode_paged.cu"
    source = original.read_text()
    if args.grouped_only:
        prefix_end = source.index(
            "at::Tensor flash_attention_grouped_sparse_page4_plan("
        )
        entry_start = source.index(
            "at::Tensor flash_attention_grouped_e4m3_fp32_paged("
        )
        entry_end = source.index(
            "int64_t flash_attention_grouped_e4m3_fp32_precision_version()"
        )
        source = source[:prefix_end] + source[entry_start:entry_end]
    if args.qk_unroll is not None:
        start = source.index("__device__ __forceinline__ void grouped_verify_qk(")
        end = source.index(
            "__device__ __forceinline__ void grouped_verify_scale_", start
        )
        qk = source[start:end]
        qk = replace_once(
            qk,
            "#pragma unroll\n  for (int k_offset = 0; "
            "k_offset < kGroupedVerifyHeadDim; k_offset += 16)",
            f"#pragma unroll {args.qk_unroll}\n  for (int k_offset = 0; "
            "k_offset < kGroupedVerifyHeadDim; k_offset += 16)",
        )
        source = source[:start] + qk + source[end:]
    if args.vector_load:
        source = replace_once(
            source,
            "bool ROW_SEQLENS = false, bool COMPENSATE_P = false>",
            "bool ROW_SEQLENS = false, bool COMPENSATE_P = false, "
            "bool PAIR_E4M3 = false>",
        )
        start = source.index("__launch_bounds__(kGroupedVerifyThreads, 1) void ")
        end = source.index(
            "void flash_attention_grouped_verify_e5m2_combine_kernel(", start
        )
        partial = source[start:end]
        old = "!SPARSE_PAGE4 && !ROW_SEQLENS"
        if partial.count(old) != 3:
            raise ValueError("Expected three grouped KV panel loads")
        partial = partial.replace(old, "!SPARSE_PAGE4 && (!ROW_SEQLENS || PAIR_E4M3)")
        source = source[:start] + partial + source[end:]
        source = replace_once(
            source,
            "  auto kernel = flash_attention_grouped_verify_e5m2_partial_kernel<\n"
            "      8, false, 0, false, false, false, "
            "flash_v100::KV_CACHE_DTYPE_FP8_E4M3,\n"
            "      false, float, true, true>;",
            "  bool paired = true;\n"
            "  for (const auto* tensor : {&k, &v}) {\n"
            "    for (int dim = 0; dim < 3; ++dim)\n"
            "      paired = paired && tensor->stride(dim) % 16 == 0;\n"
            "  }\n"
            "  auto kernel = paired\n"
            "      ? flash_attention_grouped_verify_e5m2_partial_kernel<\n"
            "          8, false, 0, false, false, false, "
            "flash_v100::KV_CACHE_DTYPE_FP8_E4M3,\n"
            "          false, float, true, true, true>\n"
            "      : flash_attention_grouped_verify_e5m2_partial_kernel<\n"
            "          8, false, 0, false, false, false, "
            "flash_v100::KV_CACHE_DTYPE_FP8_E4M3,\n"
            "          false, float, true, true, false>;",
        )
    if args.page_specialize:
        statements = []
        for page in (1648, 3296):
            prefix = (
                "flash_attention_grouped_verify_e5m2_partial_kernel<"
                f"8, false, {page}, false, false, false, "
                "flash_v100::KV_CACHE_DTYPE_FP8_E4M3, false, float, true, true"
            )
            expression = (
                f"paired ? {prefix}, true> : {prefix}, false>"
                if args.vector_load
                else prefix + ">"
            )
            statements.append(f"  if (k.size(1) == {page}) kernel = {expression};\n")
        marker = "  constexpr int kCompensatedSmemBytes ="
        source = replace_once(source, marker, "".join(statements) + marker)
    if args.prefetch_v:
        source = prefetch_values(source)
    if args.prefetch_k:
        source = prefetch_keys(source, args.prefetch_k_warps)
    if args.reuse_pv_values:
        source = reuse_pv_values(source)
    if args.qk_paired_products:
        if not args.grouped_only or args.head_groups != 1 or args.qk_unroll != 1:
            parser.error(
                "QK paired products require grouped-only, six heads and unroll1"
            )
        source = pair_qk_products(source)
    if args.qk_fp64_sum:
        source = accumulate_qk_fp64(source)
    barrier = """        __syncwarp();
        if (lane_id == 0) {
          if (tile_sum > 0.0f) {"""
    if source.count(barrier) != 1:
        raise ValueError("The grouped online-softmax warp-state fix is required")
    if args.head_groups in (2, 3):
        source = replace_once(
            source,
            "constexpr int kGroupedVerifyRows = 48;",
            "constexpr int kGroupedVerifyRows = "
            f"{32 if args.head_groups == 2 else 16};",
        )
        source = replace_once(
            source,
            "constexpr int kGroupedVerifyThreads = 512;",
            "constexpr int kGroupedVerifyThreads = 256;",
        )
        source = replace_once(
            source,
            "kernel<<<dim3(1, 80), kGroupedVerifyThreads, "
            "kCompensatedSmemBytes, stream>>>",
            f"kernel<<<dim3({args.head_groups}, 80), kGroupedVerifyThreads, "
            "kCompensatedSmemBytes, stream>>>",
        )
    if args.head_groups == 2:
        source = replace_once(
            source,
            "static constexpr int kHeadsPerCta = "
            "kGroupedVerifyRows / MAX_QUERY_TOKENS;",
            "static constexpr int kHeadsPerCta = "
            "MAX_QUERY_TOKENS == kGroupedVerifyQ8MaxQ "
            "? 3 : kGroupedVerifyRows / MAX_QUERY_TOKENS;",
        )
        source = replace_once(
            source,
            "MAX_QUERY_TOKENS * kHeadsPerCta == kGroupedVerifyRows,",
            "MAX_QUERY_TOKENS * kHeadsPerCta <= kGroupedVerifyRows,",
        )
    if args.splits != 80:
        source = replace_once(
            source,
            "constexpr int kGroupedVerifyQ8Splits = 80;",
            f"constexpr int kGroupedVerifyQ8Splits = {args.splits};",
        )
        for old, new in (
            ("{80, 8, 6, 256}", f"{{{args.splits}, 8, 6, 256}}"),
            ("{80, 8, 6, 2}", f"{{{args.splits}, 8, 6, 2}}"),
            ("[80,8,6,256]", f"[{args.splits},8,6,256]"),
            ("[80,8,6,2]", f"[{args.splits},8,6,2]"),
            (
                f"kernel<<<dim3({args.head_groups}, 80),",
                f"kernel<<<dim3({args.head_groups}, {args.splits}),",
            ),
        ):
            source = replace_once(source, old, new)
    if args.specialize_full_q8:
        source = specialize_full_q8(
            source, args.all_visible_tiles, args.register_softmax_state
        )
    if args.qk_head_rows:
        source = qk_head_rows(source)
    if args.qk_operand_pipeline:
        source = pipeline_qk_operands(source)
    if args.diagnostic_output_fp32:
        source = retain_fp32_output(source)
    source = replace_once(
        source,
        "flash_attention_grouped_e4m3_fp32_paged(",
        "private_grouped_e4m3_fp32_paged(",
    )
    source += (
        "\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n"
        '  m.def("run", &private_grouped_e4m3_fp32_paged);\n}\n'
    )
    directory = args.output_dir.resolve()
    sources = directory / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    for name in ("include", "kernel"):
        target = sources / name
        target.mkdir(exist_ok=True)
        for pattern in ("*.h", "*.cuh"):
            for header in (root / name).glob(pattern):
                shutil.copy2(header, target)
    shutil.copy2(root / "LICENSE", sources)
    path = sources / "kernel/grouped-attention.cu"
    path.write_text(source)
    source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    module_name = "sm70_grouped_attention_" + source_sha256[:12]
    # Retain Flash-V100's existing math flags; this is a scheduling candidate.
    flags = [
        "-O3",
        "-std=c++17",
        "-gencode=arch=compute_70,code=sm_70",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "-lineinfo",
        "-Xptxas=-v",
    ]
    manifest = {
        "input_source_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "source_sha256": source_sha256,
        "module_name": module_name,
        "source_files": {
            str(p.relative_to(sources)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(sources.rglob("*"))
            if p.is_file()
        },
        "head_groups": args.head_groups,
        "qk_unroll": args.qk_unroll,
        "vector_load": args.vector_load,
        "page_specialize": args.page_specialize,
        "prefetch_v": args.prefetch_v,
        "prefetch_k": args.prefetch_k,
        "prefetch_k_warps": args.prefetch_k_warps if args.prefetch_k else None,
        "specialize_full_q8": args.specialize_full_q8,
        "register_softmax_state": args.register_softmax_state,
        "qk_head_rows": args.qk_head_rows,
        "qk_operand_pipeline": args.qk_operand_pipeline,
        "all_visible_tiles": args.all_visible_tiles,
        "qk_fp64_sum": args.qk_fp64_sum,
        "arithmetic_change": args.qk_fp64_sum or args.splits != 80,
        "diagnostic_output_fp32": args.diagnostic_output_fp32,
        "reuse_pv_values": args.reuse_pv_values,
        "qk_paired_products": args.qk_paired_products,
        "splits": args.splits,
        "grouped_only": args.grouped_only,
        "extra_cuda_cflags": flags,
        "scope": "Private operator candidate; full-model admission required",
    }
    if args.build:
        from torch.utils.cpp_extension import load

        build = directory / "build"
        build.mkdir(exist_ok=True)
        library = Path(
            load(
                name=module_name,
                sources=[str(path)],
                build_directory=str(build),
                extra_cuda_cflags=flags,
                extra_include_paths=[str(sources / "kernel"), str(sources / "include")],
                verbose=True,
            ).__file__
        )
        manifest["library"] = str(library)
        manifest["library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
