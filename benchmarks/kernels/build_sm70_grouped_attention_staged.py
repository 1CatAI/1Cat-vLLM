# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Private two-stage q8 experiment; does not install or admit a serving route.

QK tiles are independent. A separate producer computes the same compensated
K16 scores; two PV column partitions retain all 80 logical context partitions
and each partition's N32 online-update order. The prototype allocates its score
scratch during warmup/capture. Serving integration and scratch admission are
deliberately separate from this feasibility experiment.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from benchmarks.kernels.build_sm70_grouped_attention_candidate import replace_once

PRODUCER = r"""
template <int PAGE_SIZE, bool PAIRED>
__global__ __launch_bounds__(256, 2) void grouped_staged_qk_kernel(
    const __half* q, const void* k_cache, const int* page_ids,
    const int* row_lengths, float* scores, int page_block_size,
    int64_t k_block_stride, int64_t k_token_stride, int64_t k_head_stride,
    float qk_scale) {
  int total_kv = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) total_kv = max(total_kv, row_lengths[i]);
  const int tile_start = blockIdx.x * 32;
  if (tile_start >= total_kv) return;
  const int valid_k_rows = min(32, total_kv - tile_start);
  const int tid = threadIdx.x;
  __shared__ __align__(16) __half shared_q[48 * 264];
  __shared__ __align__(16) __half shared_k[32 * 264];
  __shared__ __align__(16) float shared_scores[48 * 32];
  const uint4* q_vec = reinterpret_cast<const uint4*>(q);
  uint4* shared_q_vec = reinterpret_cast<uint4*>(shared_q);
  for (int i = tid; i < 48 * 32; i += 256) {
    shared_q_vec[(i / 32) * 33 + i % 32] = __ldg(q_vec + i);
  }
  load_xqa_tc_kv_panel<PAGE_SIZE, false, 256,
      flash_v100::KV_CACHE_DTYPE_FP8_E4M3, PAIRED>(
      shared_k, k_cache, page_ids, valid_k_rows, 32, 33,
      tile_start, 0, page_block_size, 0, k_block_stride, k_token_stride,
      k_head_stride, 0);
  for (int i = tid + valid_k_rows * 33; i < 32 * 33; i += 256) {
    reinterpret_cast<uint4*>(shared_k)[i] = make_uint4(0, 0, 0, 0);
  }
  __syncthreads();
  grouped_verify_qk<true>(shared_q, shared_k, shared_scores, qk_scale, 0x7);
  __syncthreads();
  for (int i = tid; i < 48 * 32; i += 256) {
    scores[static_cast<int64_t>(blockIdx.x) * 48 * 32 + i] = shared_scores[i];
  }
}

constexpr int kStagedPVThreads = 256;
constexpr int kStagedPVWarps = 8;
constexpr int kStagedPVHeadDim = 128;
constexpr int kStagedPVStride = 136;
constexpr int kStagedPVOutputTilesPerWarp = 3;
struct StagedPVSmem {
  union {
    struct {
      alignas(16) __half kv[32 * 136];
      alignas(16) float scores[48 * 32];
      alignas(16) __half probs[48 * 40];
      alignas(16) __half residual[48 * 40];
    } compute;
    alignas(16) float output[48 * 128];
  } storage;
  alignas(16) float row_max[48];
  alignas(16) float row_sum[48];
  alignas(16) float row_scale[48];
  alignas(16) int page_ids[16];
  alignas(16) uint32_t sparse_token_masks[8];
};
static_assert(sizeof(StagedPVSmem) <= 48 * 1024, "PV storage budget");
"""


def staged_source(source: str) -> str:
    start = source.index(
        "template <int MAX_QUERY_TOKENS, bool TWO_PASS, int PAGE_BLOCK_SIZE"
    )
    end = source.index(
        "template <int MAX_QUERY_TOKENS, bool SINGLE_QUERY, typename PARTIAL_T", start
    )
    partial = source[start:end]
    partial = partial.replace(
        "flash_attention_grouped_verify_e5m2_partial_kernel", "grouped_staged_pv_kernel"
    )
    partial = replace_once(
        partial,
        "const int* row_lengths = nullptr) {",
        "const int* row_lengths = nullptr, const float* staged_scores = nullptr) {",
    )
    partial = partial.replace("kGroupedVerifyThreads", "kStagedPVThreads")
    partial = partial.replace("kGroupedVerifyWarps", "kStagedPVWarps")
    partial = partial.replace(
        "kGroupedVerifyOutputTilesPerWarp", "kStagedPVOutputTilesPerWarp"
    )
    partial = partial.replace("kGroupedVerifyHeadDim", "kStagedPVHeadDim")
    partial = partial.replace("kGroupedVerifyKVStride", "kStagedPVStride")
    partial = partial.replace(
        "__launch_bounds__(kStagedPVThreads, 1)",
        "__launch_bounds__(kStagedPVThreads, 2)",
    )
    partial = replace_once(
        partial,
        "const int group_idx = blockIdx.z;",
        "const int group_idx = 0;\n  const int column_partition = blockIdx.z;",
    )
    partial = partial.replace("GroupedVerifySmem", "StagedPVSmem")
    partial = replace_once(
        partial,
        "__half* shared_q = smem.storage.compute.q;",
        "__half* shared_q = nullptr;",
    )
    a = partial.index("  __half* shared_prob_residual = ")
    b = partial.index("  const int* page_ids =", a)
    partial = (
        partial[:a]
        + """  __half* shared_prob_residual = smem.storage.compute.residual;
  constexpr int kResidualStride = kGroupedVerifyProbStride;
  __half* shared_values = shared_kv;
  v_cache = static_cast<const uint8_t*>(v_cache) + column_partition * 128;
"""
        + partial[b:]
    )
    # Q is consumed only by the independent producer. Do not reserve or fill a
    # query panel in the PV block. The TWO_PASS template is never instantiated.
    a = partial.index("  constexpr int kVecsPerRow =")
    b = partial.index("  if (tid < kGroupedVerifyRows)", a)
    partial = partial[:a] + partial[b:]
    a = partial.index("  // The conservative baseline computes")
    b = partial.index("  // Recompute QK for the conservative path", a)
    partial = partial[:a] + partial[b:]
    a = partial.index("    load_xqa_tc_kv_panel<")
    b = partial.index("    int active_m_tiles =", a)
    partial = (
        partial[:a] + "    for (int i = tid; i < 48 * 32; i += kStagedPVThreads) {\n"
        "      shared_scores[i] = staged_scores["
        "static_cast<int64_t>(tile_start / 32) * 48 * 32 + i];\n"
        "    }\n" + partial[b:]
    )
    a = partial.index("    grouped_verify_qk<COMPENSATE_P>")
    b = partial.index("    load_xqa_tc_kv_panel<", a)
    partial = (
        partial[:a]
        + """    constexpr int kValueLoadThreads = kStagedPVThreads;
    const int value_load_tid = tid;
"""
        + partial[b:]
    )
    partial = replace_once(
        partial, "    }\n    }\n\n    __syncthreads();", "    }\n\n    __syncthreads();"
    )
    # Both D halves retain the same N32 updates. Only their disjoint numerator
    # columns are stored; one partition alone publishes the common max/sum.
    a = partial.index("      int64_t output_idx;")
    b = partial.index("      if constexpr (std::is_same_v<PARTIAL_T, float>)", a)
    partial = (
        partial[:a]
        + """      const int64_t output_idx =
          (((static_cast<int64_t>(split_id) * MAX_QUERY_TOKENS + token_idx) *
            kGroupedVerifyHeads + head_idx) * 256 + column_partition * 128 + d);
"""
        + partial[b:]
    )
    a = partial.rindex("  if (tid < kGroupedVerifyRows)")
    partial = partial[:a] + partial[a:].replace(
        "if (tid < kGroupedVerifyRows)",
        "if (column_partition == 0 && tid < kGroupedVerifyRows)",
        1,
    )
    source = source[:end] + PRODUCER + partial + source[end:]
    # Keep the complete original host validation and fallback for untested
    # small/other shapes. Prototype scratch is capture-owned, not a cache.
    a = source.index(
        "  auto kernel = paired", source.index("private_grouped_e4m3_fp32_paged(")
    )
    b = source.index("  flash_attention_grouped_verify_e5m2_combine_kernel<", a)
    original = source[a:b]
    declarations = []
    for page in (0, 1648, 3296):
        prefix = "auto" if page == 0 else f"if (k.size(1) == {page})"
        for op, name in (
            ("grouped_staged_qk_kernel", "producer"),
            ("grouped_staged_pv_kernel", "consumer"),
        ):
            suffix = (
                ""
                if name == "producer"
                else ", false, false, false, flash_v100::KV_CACHE_DTYPE_FP8_E4M3, "
                "false, float, true, true"
            )
            args = (
                str(page) if name == "producer" else "8, false, " + str(page) + suffix
            )
            assign = f"auto {name} =" if page == 0 else f"{prefix} {name} ="
            declarations.append(
                f"    {assign} paired ? {op}<{args}, true> : {op}<{args}, false>;"
            )
    replacement = (
        """  if (q.size(0) == 8 && block_table.size(1) * k.size(1) > 32768) {
    const int tiles = (block_table.size(1) * k.size(1) + 31) / 32;
    auto scores = at::empty({tiles, 48, 32}, q.options().dtype(at::kFloat));
"""
        + "\n".join(declarations)
        + """
    C10_CUDA_CHECK(cudaFuncSetAttribute(consumer,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(StagedPVSmem)));
    producer<<<tiles, 256, 0, stream>>>(
        reinterpret_cast<const __half*>(aligned_q.data_ptr()), k.data_ptr(),
        block_table.data_ptr<int>(), row_lengths.data_ptr<int>(),
        scores.data_ptr<float>(), k.size(1), k.stride(0), k.stride(1),
        k.stride(2), scale * k_scale);
    consumer<<<dim3(1, 80, 2), kStagedPVThreads, sizeof(StagedPVSmem), stream>>>(
        reinterpret_cast<const __half*>(aligned_q.data_ptr()), k.data_ptr(),
        v.data_ptr(), block_table.data_ptr<int>(), row_lengths.data_ptr<int>(),
        partial.data_ptr<float>(), lse.data_ptr<float>(), q.size(0),
        block_table.size(1), k.size(1), k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2), scale * k_scale, v_scale,
        nullptr, 1, row_lengths.data_ptr<int>(), scores.data_ptr<float>());
  } else {
"""
        + original
        + "  }\n"
    )
    return source[:a] + replacement + source[b:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--shared-carveout", type=int, choices=(100,))
    args = parser.parse_args()
    base = json.loads(args.base_manifest.read_text())
    assert base["head_groups"] == 1 and base.get("splits", 80) == 80
    assert base["prefetch_v"] and base["qk_unroll"] == 2
    source_dir = args.base_manifest.parent / "sources"
    source_file = source_dir / "kernel/grouped-attention.cu"
    assert hashlib.sha256(source_file.read_bytes()).hexdigest() == base["source_sha256"]
    directory = args.output_dir.resolve()
    sources = directory / "sources"
    shutil.copytree(source_dir, sources)
    path = sources / "kernel/grouped-attention.cu"
    source = staged_source(source_file.read_text())
    if args.shared_carveout is not None:
        launch = "    producer<<<tiles, 256, 0, stream>>>("
        source = replace_once(
            source,
            launch,
            "    C10_CUDA_CHECK(cudaFuncSetAttribute(producer,\n"
            "        cudaFuncAttributePreferredSharedMemoryCarveout, 100));\n"
            "    C10_CUDA_CHECK(cudaFuncSetAttribute(consumer,\n"
            "        cudaFuncAttributePreferredSharedMemoryCarveout, 100));\n" + launch,
        )
        source = replace_once(
            source,
            "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {",
            r"""
pybind11::dict staged_resource_report() {
  auto describe = [](auto kernel, int threads, int dynamic_bytes) {
    cudaFuncAttributes attrs;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attrs, kernel));
    int blocks = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, kernel, threads, dynamic_bytes));
    pybind11::dict result;
    result["registers"] = attrs.numRegs;
    result["static_shared_bytes"] = attrs.sharedSizeBytes;
    result["dynamic_shared_bytes"] = dynamic_bytes;
    result["preferred_shared_carveout"] = attrs.preferredShmemCarveout;
    result["resource_limited_blocks_per_sm"] = blocks;
    return result;
  };
  pybind11::dict result;
  result["qk"] = describe(grouped_staged_qk_kernel<3296, true>, 256, 0);
  result["pv"] = describe(grouped_staged_pv_kernel<8, false, 3296,
      false, false, false, flash_v100::KV_CACHE_DTYPE_FP8_E4M3,
      false, float, true, true, true>, 256, sizeof(StagedPVSmem));
  return result;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("resource_report", &staged_resource_report);
""",
        )
    path.write_text(source)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    module_name = "sm70_staged_attention_" + digest[:12]
    manifest = {
        **base,
        "input_source_sha256": base["source_sha256"],
        "source_sha256": digest,
        "module_name": module_name,
        "staged_qk": True,
        "pv_column_partitions": 2,
        "capture_owned_score_scratch": True,
        "shared_carveout": args.shared_carveout,
        "source_files": {
            str(p.relative_to(sources)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(sources.rglob("*"))
            if p.is_file()
        },
        "scope": "Private feasibility candidate, not admitted for serving",
    }
    manifest.pop("library", None)
    manifest.pop("library_sha256", None)
    if args.build:
        from torch.utils.cpp_extension import load

        build = directory / "build"
        build.mkdir()
        module = load(
            name=module_name,
            sources=[str(path)],
            build_directory=str(build),
            extra_cuda_cflags=base["extra_cuda_cflags"],
            extra_include_paths=[str(sources / "kernel"), str(sources / "include")],
            verbose=True,
        )
        library = Path(module.__file__)
        manifest["library"] = str(library)
        manifest["library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
