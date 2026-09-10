# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch two independent QK tiles while retaining both ordered N32 updates.

Twelve QK warps and four raw-V prefetch warps share a 512-thread CTA. The
candidate retains 80 logical partitions and the original K16 compensation,
softmax and PV operations. Only aligned q8 inputs enter this private prototype.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import regex as re

from benchmarks.kernels.build_sm70_grouped_attention_candidate import replace_once

STORAGE = r"""
struct alignas(256) GroupedN64Smem {
  union {
    struct {
      alignas(16) __half q[48 * 264];
      alignas(16) __half kv[64 * 264];
      alignas(16) float scores[48 * 64];
      alignas(16) __half probs[48 * 40];
      alignas(16) __half residual[48 * 40];
    } compute;
    alignas(16) float output[48 * 256];
  } storage;
  alignas(16) float row_max[48];
  alignas(16) float row_sum[48];
  alignas(16) float row_scale[48];
  alignas(16) int page_ids[kGroupedVerifyPageIdsCapacity];
  alignas(16) uint32_t sparse_token_masks[8];
};
static_assert(sizeof(GroupedN64Smem) <= 96 * 1024, "N64 SM70 storage budget");
"""

PREFETCH = r"""
    uint4 prefetched_v[8];
    if (warp_id < 12) {
      grouped_verify_qk_n64<true>(shared_q, shared_kv, shared_scores,
                                 qk_scale, active_m_tiles);
    } else {
      const int copy_tid = tid - 384;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int index = copy_tid + i * 128;
        const int row = index / 16;
        const int vector = index % 16;
        uint4 raw = make_uint4(0, 0, 0, 0);
        if (row < valid_batch_rows) {
          const int token = batch_start + row;
          const int page = PAGE_BLOCK_SIZE > 0
              ? token / PAGE_BLOCK_SIZE : token / page_block_size;
          const int offset = token - page *
              (PAGE_BLOCK_SIZE > 0 ? PAGE_BLOCK_SIZE : page_block_size);
          const int64_t base = static_cast<int64_t>(page_ids[page]) *
              v_block_stride + static_cast<int64_t>(offset) * v_token_stride;
          raw = __ldg(reinterpret_cast<const uint4*>(v_cache) + base / 16 + vector);
        }
        prefetched_v[i] = raw;
      }
    }
    __syncthreads();
    // K is dead after QK. The prefetch warps may now publish decoded V in
    // the same panel; all consumers wait until publication is complete.
    if (warp_id >= 12) {
      const int copy_tid = tid - 384;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int index = copy_tid + i * 128;
        const int shared_offset = (index / 16) * 33 + (index % 16) * 2;
        const uint4 raw = prefetched_v[i];
        const uint64_t lo = static_cast<uint64_t>(raw.x) |
            (static_cast<uint64_t>(raw.y) << 32);
        const uint64_t hi = static_cast<uint64_t>(raw.z) |
            (static_cast<uint64_t>(raw.w) << 32);
        reinterpret_cast<uint4*>(shared_values)[shared_offset] =
            fp8_e4m3fn_vector_to_half8_fast(lo);
        reinterpret_cast<uint4*>(shared_values)[shared_offset + 1] =
            fp8_e4m3fn_vector_to_half8_fast(hi);
      }
    }
    __syncthreads();
"""


def batched_source(source: str, v_loading: str = "prefetch") -> str:
    begin = source.index("template <bool COMPENSATE = false>")
    end = source.index("__device__ __forceinline__ void grouped_verify_scale", begin)
    qk = source[begin:end]
    qk = qk.replace("grouped_verify_qk(", "grouped_verify_qk_n64(")
    qk = qk.replace("kGroupedVerifyQKWarps", "12")
    qk = qk.replace("(kGroupedVerifyBlockN / 16)", "4")
    qk = qk.replace("kGroupedVerifyScoreStride", "64")
    source = source[:end] + STORAGE + qk + source[end:]

    begin = source.index(
        "template <int MAX_QUERY_TOKENS, bool TWO_PASS, int PAGE_BLOCK_SIZE"
    )
    end = source.index("template <int MAX_QUERY_TOKENS, bool SINGLE_QUERY", begin)
    partial = source[begin:end]
    old = "flash_attention_grouped_verify_e5m2_partial_kernel"
    new = "flash_attention_grouped_verify_n64_partial_kernel"
    partial = partial.replace(old, new).replace("GroupedVerifySmem", "GroupedN64Smem")
    marker = "  using Traits = GroupedVerifyTraits<MAX_QUERY_TOKENS>;"
    partial = replace_once(
        partial,
        marker,
        "  static_assert(MAX_QUERY_TOKENS == 8 && COMPENSATE_P && PAIR_E4M3 &&\n"
        "      ROW_SEQLENS && !TWO_PASS && !SPARSE_PAGE4 &&\n"
        "      !CONTIGUOUS_HKV1_LAYOUT && !STAGE_PARTITION_PAGE_IDS,\n"
        '      "Private N64 q8 contract");\n' + marker,
    )
    a = partial.index("  __half* shared_prob_residual =")
    b = partial.index("  const int* page_ids =", a)
    partial = (
        partial[:a]
        + "  __half* shared_prob_residual = smem.storage.compute.residual;\n"
        "  constexpr int kResidualStride = kGroupedVerifyProbStride;\n"
        "  __half* shared_values = shared_kv;\n" + partial[b:]
    )
    a = partial.index("  // The conservative baseline computes")
    b = partial.index("  // Recompute QK for the conservative path", a)
    partial = partial[:a] + partial[b:]
    a = partial.index("  // Recompute QK for the conservative path")
    b = partial.index("    grouped_verify_qk<COMPENSATE_P>", a)
    prefix = partial[a:b]
    prefix = re.sub(r"\btile_start\b", "batch_start", prefix)
    prefix = re.sub(r"\bvalid_k_rows\b", "valid_batch_rows", prefix)
    prefix = prefix.replace("kGroupedVerifyBlockN", "64")
    c = partial.index("    if constexpr (TWO_PASS)", b)
    d = partial.index("    __syncthreads();\n  }", c)
    d += len("    __syncthreads();")
    ordered = partial[c:d]
    ordered = ordered.replace(
        "shared_scores[row * kGroupedVerifyScoreStride + lane_id]",
        "shared_scores[row * 64 + subtile_offset + lane_id]",
    )
    ordered = replace_once(
        ordered,
        "shared_values + k_offset * kGroupedVerifyKVStride + d_tile * 16,",
        "shared_values + (subtile_offset + k_offset) *\n"
        "              kGroupedVerifyKVStride + d_tile * 16,",
    )
    value_loading = PREFETCH
    publish_first = ""
    if v_loading == "after-qk":
        value_loading = r"""
    grouped_verify_qk_n64<true>(shared_q, shared_kv, shared_scores,
                               qk_scale, active_m_tiles);
    __syncthreads();
    load_xqa_tc_kv_panel<PAGE_BLOCK_SIZE, false, 512,
        flash_v100::KV_CACHE_DTYPE_FP8_E4M3, true>(
        shared_values, v_cache, page_ids, valid_batch_rows, 32, 33,
        batch_start, 0, page_block_size, 0, v_block_stride, v_token_stride,
        v_head_stride, 0);
    for (int i = tid + valid_batch_rows * 33; i < 64 * 33; i += 512)
      reinterpret_cast<uint4*>(shared_values)[i] = make_uint4(0, 0, 0, 0);
    __syncthreads();
"""
    elif v_loading == "softmax":
        # Hold only the first half in registers. Publishing it and reading the
        # other half are independent of the first N32 softmax's score/state.
        prefetch = PREFETCH.replace("prefetched_v[8]", "prefetched_v[4]")
        prefetch = prefetch.replace("i < 8", "i < 4")
        at = prefetch.index("    // K is dead after QK.")
        value_loading = prefetch[:at]
        publish = prefetch[at:]
        ending = "    }\n    __syncthreads();\n"
        assert publish.endswith(ending)
        publish = (
            publish[: -len(ending)]
            + r"""
      if (valid_batch_rows > 32) {
        load_xqa_tc_kv_panel<PAGE_BLOCK_SIZE, false, 128,
            flash_v100::KV_CACHE_DTYPE_FP8_E4M3, true>(
            shared_values + 32 * 264, v_cache, page_ids,
            valid_batch_rows - 32, 32, 33, batch_start + 32, 0,
            page_block_size, 0, v_block_stride, v_token_stride,
            v_head_stride, 0, copy_tid);
        for (int i = copy_tid + (valid_batch_rows - 32) * 33;
             i < 32 * 33; i += 128)
          reinterpret_cast<uint4*>(shared_values + 32 * 264)[i] =
              make_uint4(0, 0, 0, 0);
      }
    }
"""
        )
        publish_first = "      if (subtile_offset == 0) {\n" + publish + "      }\n"
        ordered = replace_once(
            ordered,
            """#pragma unroll
      for (int row = warp_id; row < kGroupedVerifyRows;
           row += kGroupedVerifyWarps) {""",
            """      const int softmax_warps = subtile_offset == 0 ? 12 : 16;
#pragma unroll
      for (int row = warp_id < softmax_warps ? warp_id : kGroupedVerifyRows;
           row < kGroupedVerifyRows; row += softmax_warps) {""",
        )
    else:
        assert v_loading == "prefetch"
    partial = (
        partial[:a]
        + prefix
        + value_loading
        + "    for (int subtile_offset = 0; subtile_offset < valid_batch_rows;\n"
        "         subtile_offset += 32) {\n"
        "      const int tile_start = batch_start + subtile_offset;\n"
        "      const int valid_k_rows = min(32, valid_batch_rows - subtile_offset);\n"
        + publish_first
        + ordered
        + "\n    }"
        + partial[d:]
    )
    source = source[:end] + partial + source[end:]
    a = source.index("  auto kernel =", source.index("private_grouped_e4m3_fp32_paged"))
    b = source.index("  constexpr int kCompensatedSmemBytes", a)
    selected = source[a:b]
    # Only paired aligned q8 is admitted; all other layouts use the parent.
    selected = selected.replace("auto kernel = paired", "kernel = true", 1)
    selected = selected.replace("paired ?", "true ?").replace(old, new)
    # Both sides of a constant conditional must instantiate a valid template.
    selected = selected.replace("true, true, false>", "true, true, true>")
    at = source.index("  C10_CUDA_CHECK(\n      cudaFuncSetAttribute(kernel", b)
    source = (
        source[:at] + "  int shared_bytes = kCompensatedSmemBytes;\n"
        "  if (q.size(0) == 8 && paired) {\n"
        + selected
        + "    shared_bytes = sizeof(GroupedN64Smem);\n"
        "  }\n"
        "  TORCH_CHECK(properties->sharedMemPerBlockOptin >= shared_bytes,\n"
        '              "N64 shared memory exceeds device limit");\n' + source[at:]
    )
    tail = source[at:]
    tail = tail.replace(
        "                           kCompensatedSmemBytes));",
        "                           shared_bytes));",
    )
    tail = tail.replace("kCompensatedSmemBytes, stream>>>", "shared_bytes, stream>>>")
    return source[:at] + tail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--build", action="store_true")
    parser.add_argument(
        "--v-loading", choices=("prefetch", "after-qk", "softmax"), default="prefetch"
    )
    args = parser.parse_args()
    base = json.loads(args.base_manifest.read_text())
    assert base["head_groups"] == 1 and base["splits"] == 80
    assert base["prefetch_v"] and base["reuse_pv_values"]
    original = args.base_manifest.parent / "sources"
    for name, digest in base["source_files"].items():
        assert hashlib.sha256((original / name).read_bytes()).hexdigest() == digest
    directory = args.output_dir.resolve()
    sources = directory / "sources"
    shutil.copytree(original, sources)
    path = sources / "kernel/grouped-attention.cu"
    path.write_text(batched_source(path.read_text(), args.v_loading))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    module_name = "sm70_grouped_n64_" + digest[:12]
    manifest = {
        **{k: v for k, v in base.items() if k not in ("library", "library_sha256")},
        "input_source_sha256": base["source_sha256"],
        "source_sha256": digest,
        "module_name": module_name,
        "scope": "Private physical N64 schedule; both logical N32 updates retained",
        "physical_tile_n": 64,
        "logical_update_n": 32,
        "prefetch_v": args.v_loading != "after-qk",
        "raw_v_prefetch": args.v_loading != "after-qk",
        "v_loading": args.v_loading,
        "source_files": {
            str(p.relative_to(sources)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(sources.rglob("*"))
            if p.is_file()
        },
    }
    if args.build:
        from torch.utils.cpp_extension import load

        build = directory / "build"
        build.mkdir()
        module = load(
            name=module_name,
            sources=[str(path)],
            extra_include_paths=[str(sources / "include"), str(sources / "kernel")],
            extra_cuda_cflags=base["extra_cuda_cflags"],
            build_directory=str(build),
            verbose=True,
        )
        manifest["library"] = str(Path(module.__file__).resolve())
        manifest["library_sha256"] = hashlib.sha256(
            Path(module.__file__).read_bytes()
        ).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
