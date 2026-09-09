# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Private SM70 QK/PV warp pipeline; no serving route is installed.

Eight producer warps and sixteen consumer warps share two score/value panels.
Named ready/free barriers protect each panel. K16 compensation, N32 updates,
80 logical partitions and the final merge are copied from the audited source.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from benchmarks.kernels.build_sm70_grouped_attention_candidate import replace_once

PREFIX = r"""
constexpr int kPipelineProducerThreads = 256;
constexpr int kPipelineConsumerThreads = 512;
constexpr int kPipelineThreads = 768;
struct alignas(256) GroupedPipelineSmem {
  union {
    struct {
      alignas(16) __half q[48 * 264];
      alignas(16) __half k[32 * 264];
      alignas(16) __half values[2][32 * 264];
      alignas(16) float scores[2][48 * 32];
      alignas(16) __half probs[48 * 40];
      alignas(16) __half residual[48 * 40];
    } compute;
    alignas(16) float output[48 * 256];
  } storage;
  alignas(16) float row_max[48];
  alignas(16) float row_sum[48];
  alignas(16) float row_scale[48];
};
static_assert(sizeof(GroupedPipelineSmem) <= 96 * 1024, "pipeline storage budget");

// Deliberately use unaligned barrier instructions: producer and consumer
// warps follow different control flow. A panel cannot be reused until every
// consumer has arrived at its free barrier. No warp spins on shared flags.
template <int COUNT>
__device__ __forceinline__ void pipeline_sync(int id) {
  asm volatile("barrier.sync %0, %1;" :: "r"(id), "n"(COUNT) : "memory");
}
template <int COUNT>
__device__ __forceinline__ void pipeline_arrive(int id) {
  asm volatile("barrier.arrive %0, %1;" :: "r"(id), "n"(COUNT) : "memory");
}

template <int PAGE_BLOCK_SIZE, bool PAIR_E4M3>
__global__ __launch_bounds__(kPipelineThreads, 1)
void grouped_pipeline_partial_kernel(
    const __half* q, const void* k_cache, const void* v_cache,
    const int* page_ids, const int* row_lengths,
    float* partial_out, float* partial_lse, int query_len,
    int page_block_size, int64_t k_block_stride, int64_t k_token_stride,
    int64_t k_head_stride, int64_t v_block_stride, int64_t v_token_stride,
    int64_t v_head_stride, float qk_scale, float v_scale) {
  using Traits = GroupedVerifyTraits<8>;
  using PARTIAL_T = float;
  constexpr int MAX_QUERY_TOKENS = 8;
  constexpr bool SPARSE_PAGE4 = false;
  constexpr bool ROW_SEQLENS = true;
  constexpr bool COMPENSATE_P = true;
  constexpr int group_idx = 0;
  constexpr int head_start = 0;
  constexpr int active_m_tiles = 7;
  constexpr int kResidualStride = 40;
  const int block_tid = threadIdx.x;
  const int split_id = blockIdx.x;
  int total_kv = 0;
  for (int i = 0; i < query_len; ++i)
    total_kv = max(total_kv, row_lengths[i]);
  if (total_kv <= 0) return;
  const int active_splits = grouped_verify_active_splits<8, false>(total_kv);
  if (split_id >= active_splits) return;
  const int total_tiles = (total_kv + 31) / 32;
  const int base_tiles = total_tiles / active_splits;
  const int extra_tiles = total_tiles - base_tiles * active_splits;
  const int split_tile_start = split_id * base_tiles + min(split_id, extra_tiles);
  const int split_tiles = base_tiles + (split_id < extra_tiles ? 1 : 0);
  const int split_start = split_tile_start * 32;
  const int prefix_kv_len = max(0, total_kv - query_len);

  extern __shared__ char pipeline_smem_raw[];
  auto& smem = *reinterpret_cast<GroupedPipelineSmem*>(pipeline_smem_raw);
  __half* shared_q = smem.storage.compute.q;
  __half* shared_k = smem.storage.compute.k;
  auto* q_vec = reinterpret_cast<const uint4*>(q);
  for (int i = block_tid; i < 48 * 32; i += kPipelineThreads) {
    reinterpret_cast<uint4*>(shared_q)[(i / 32) * 33 + i % 32] = __ldg(q_vec + i);
  }
  if (block_tid < 48) {
    smem.row_max[block_tid] = kXQANegInf;
    smem.row_sum[block_tid] = 0.0f;
    smem.row_scale[block_tid] = 1.0f;
  }
  __syncthreads();

  if (block_tid < kPipelineProducerThreads) {
    for (int tile = 0; tile < split_tiles; ++tile) {
      const int panel = tile & 1;
      if (tile >= 2) pipeline_sync<kPipelineThreads>(3 + panel);
      const int tile_start = split_start + tile * 32;
      const int valid_k_rows = min(32, total_kv - tile_start);
      load_xqa_tc_kv_panel<PAGE_BLOCK_SIZE, false, kPipelineProducerThreads,
          flash_v100::KV_CACHE_DTYPE_FP8_E4M3, PAIR_E4M3>(
          shared_k, k_cache, page_ids, valid_k_rows, 32, 33,
          tile_start, 0, page_block_size, 0, k_block_stride, k_token_stride,
          k_head_stride, 0, block_tid);
      for (int i = block_tid + valid_k_rows * 33; i < 32 * 33;
           i += kPipelineProducerThreads)
        reinterpret_cast<uint4*>(shared_k)[i] = make_uint4(0, 0, 0, 0);
      pipeline_sync<kPipelineProducerThreads>(5);
      grouped_verify_qk<true>(shared_q, shared_k,
          smem.storage.compute.scores[panel], qk_scale, 7);
      __half* values = smem.storage.compute.values[panel];
      load_xqa_tc_kv_panel<PAGE_BLOCK_SIZE, false, kPipelineProducerThreads,
          flash_v100::KV_CACHE_DTYPE_FP8_E4M3, PAIR_E4M3>(
          values, v_cache, page_ids, valid_k_rows, 32, 33,
          tile_start, 0, page_block_size, 0, v_block_stride, v_token_stride,
          v_head_stride, 0, block_tid);
      for (int i = block_tid + valid_k_rows * 33; i < 32 * 33;
           i += kPipelineProducerThreads)
        reinterpret_cast<uint4*>(values)[i] = make_uint4(0, 0, 0, 0);
      // Every producer must finish reading K before any starts the next load.
      pipeline_sync<kPipelineProducerThreads>(5);
      pipeline_arrive<kPipelineThreads>(1 + panel);
    }
    // Consumers overlay compute storage only after production has finished.
    pipeline_sync<kPipelineThreads>(7);
    pipeline_sync<kPipelineThreads>(8);
    return;
  }

  const int tid = block_tid - kPipelineProducerThreads;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;
  __half* shared_probs = smem.storage.compute.probs;
  __half* shared_prob_residual = smem.storage.compute.residual;
  volta::fragment<volta::accumulator, 16, 16, 16, float> output_fragments[3];
#pragma unroll
  for (int i = 0; i < 3; ++i) volta::fill_fragment(output_fragments[i], 0.0f);
  for (int tile = 0; tile < split_tiles; ++tile) {
    const int panel = tile & 1;
    pipeline_sync<kPipelineThreads>(1 + panel);
    const int tile_start = split_start + tile * 32;
    const int valid_k_rows = min(32, total_kv - tile_start);
    float* shared_scores = smem.storage.compute.scores[panel];
    __half* shared_values = smem.storage.compute.values[panel];
"""


def pipeline_source(
    source: str, serialize: bool = False, debug_first_tile: bool = False
) -> str:
    start = source.index("__launch_bounds__(kGroupedVerifyThreads, 1) void ")
    end = source.index(
        "template <int MAX_QUERY_TOKENS, bool SINGLE_QUERY, typename PARTIAL_T",
        start,
    )
    partial = source[start:end]
    # The first row loop belongs to TWO_PASS and writes max/sum only. The
    # pipeline needs the online branch that also publishes P, its residual,
    # and the rescale consumed by PV. Copying the statistics-only pass leaves
    # those panels uninitialized even when producer/consumer overlap is off.
    anchor = "    } else {\n#pragma unroll\n      for (int row = warp_id;"
    a = partial.index(anchor) + len("    } else {\n")
    b = partial.index("      __syncthreads();", a)
    softmax = partial[a:b].replace("smem.sparse_token_masks", "nullptr")
    for publication in (
        "shared_probs[row * kGroupedVerifyProbStride + lane_id] =",
        "shared_prob_residual[row * kResidualStride + lane_id] =",
        "smem.row_scale[row] = exp_diff;",
        "__syncwarp();",
    ):
        assert softmax.count(publication) == 1, publication
    a = partial.index("    static_assert(COMPENSATE_P && kGroupedVerifyWarps == 16,")
    b = partial.index("    __syncthreads();\n  }", a)
    pv = partial[a:b]
    if debug_first_tile:
        pv = (
            r"""    if (split_id == 0 && tile == 0 && tid == 0)
      printf("PIPE_BEFORE tid=%d p=%f residual=%f v=%f scale=%f output=%f\n",
             int(threadIdx.x), __half2float(shared_probs[0]),
             __half2float(shared_prob_residual[0]), __half2float(shared_values[0]),
             smem.row_scale[0], output_fragments[0].x[0]);
"""
            + pv
            + r"""    if (split_id == 0 && tile == 0 && tid == 0)
      printf("PIPE_AFTER tile=%f output=%f vscale=%f\n",
             tile_fragments[0].x[0], output_fragments[0].x[0], v_scale);
"""
        )
    a = partial.index("  // The compute buffers are dead.", b)
    tail = partial[a:]
    tail = tail.replace(
        "  __syncthreads();", "  pipeline_sync<kPipelineThreads>(7);", 1
    )
    tail = tail.replace(
        "  __syncthreads();", "  pipeline_sync<kPipelineThreads>(8);", 1
    )
    assert "__syncthreads" not in tail
    kernel = (
        PREFIX
        + softmax
        + "    pipeline_sync<kPipelineConsumerThreads>(6);\n"
        + pv
        + "    if (tile + 2 < split_tiles) "
        "pipeline_arrive<kPipelineThreads>(3 + panel);\n  }\n" + tail
    )
    if serialize:
        kernel = replace_once(
            kernel,
            "      pipeline_arrive<kPipelineThreads>(1 + panel);",
            "      pipeline_arrive<kPipelineThreads>(1 + panel);\n"
            "      pipeline_sync<kPipelineThreads>(9);",
        )
        kernel = replace_once(
            kernel,
            "pipeline_arrive<kPipelineThreads>(3 + panel);\n  }",
            "pipeline_arrive<kPipelineThreads>(3 + panel);\n"
            "    pipeline_sync<kPipelineThreads>(9);\n  }",
        )
    source = source[:end] + kernel + source[end:]
    a = source.index(
        "  auto kernel = paired", source.index("private_grouped_e4m3_fp32_paged(")
    )
    b = source.index("  flash_attention_grouped_verify_e5m2_combine_kernel<", a)
    fallback = source[a:b]
    launch = r"""  if (q.size(0) == 8 && block_table.size(1) * k.size(1) > 32768) {
    auto pipeline = paired ? grouped_pipeline_partial_kernel<0, true>
                           : grouped_pipeline_partial_kernel<0, false>;
    if (k.size(1) == 1648)
      pipeline = paired ? grouped_pipeline_partial_kernel<1648, true>
                        : grouped_pipeline_partial_kernel<1648, false>;
    if (k.size(1) == 3296)
      pipeline = paired ? grouped_pipeline_partial_kernel<3296, true>
                        : grouped_pipeline_partial_kernel<3296, false>;
    TORCH_CHECK(properties->sharedMemPerBlockOptin >= sizeof(GroupedPipelineSmem),
                "warp pipeline exceeds the opt-in shared memory budget");
    C10_CUDA_CHECK(cudaFuncSetAttribute(pipeline,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(GroupedPipelineSmem)));
    C10_CUDA_CHECK(cudaFuncSetAttribute(pipeline,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    pipeline<<<80, kPipelineThreads, sizeof(GroupedPipelineSmem), stream>>>(
        reinterpret_cast<const __half*>(aligned_q.data_ptr()), k.data_ptr(),
        v.data_ptr(), block_table.data_ptr<int>(), row_lengths.data_ptr<int>(),
        partial.data_ptr<float>(), lse.data_ptr<float>(), q.size(0), k.size(1),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2), scale * k_scale, v_scale);
  } else {
"""
    return source[:a] + launch + fallback + "  }\n" + source[b:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--serialize", action="store_true")
    parser.add_argument("--debug-first-tile", action="store_true")
    args = parser.parse_args()
    base = json.loads(args.base_manifest.read_text())
    if not (
        base["head_groups"] == 1
        and base.get("reuse_pv_values")
        and base["splits"] == 80
    ):
        parser.error("The pipeline requires an 80-split six-head PV-reuse source")
    source_dir = args.base_manifest.parent / "sources"
    source_file = source_dir / "kernel/grouped-attention.cu"
    assert hashlib.sha256(source_file.read_bytes()).hexdigest() == base["source_sha256"]
    source = pipeline_source(
        source_file.read_text(), args.serialize, args.debug_first_tile
    )
    directory = args.output_dir.resolve()
    if directory.exists():
        parser.error("Use a new output directory for every prototype")
    sources = directory / "sources"
    shutil.copytree(source_dir, sources)
    path = sources / "kernel/grouped-attention.cu"
    path.write_text(source)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    module_name = "sm70_grouped_attention_" + digest[:12]
    manifest = dict(base)
    manifest.pop("library", None)
    manifest.pop("library_sha256", None)
    manifest.update(
        input_source_sha256=base["source_sha256"],
        source_sha256=digest,
        module_name=module_name,
        warp_pipeline=True,
        serialized_diagnostic=args.serialize,
        debug_first_tile=args.debug_first_tile,
        source_files={
            str(p.relative_to(sources)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(sources.rglob("*"))
            if p.is_file()
        },
        scope="Private warp-pipeline operator; no serving admission",
        synchronization_reference="https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar",
    )
    if args.build:
        from torch.utils.cpp_extension import load

        build = directory / "build"
        build.mkdir()
        library = Path(
            load(
                name=module_name,
                sources=[str(path)],
                build_directory=str(build),
                extra_cuda_cflags=base["extra_cuda_cflags"],
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
