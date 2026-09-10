# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolate the traced FP32 scalar q1 fallback and its ordered PV loop.

This builder installs no backend. Candidates keep each head's scalar FP32
operations and the frozen partition/merge order. Independent load scheduling
and six-head KV reuse are screened separately.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from benchmarks.kernels.build_sm70_grouped_attention_candidate import replace_once

SHARED_HEADS_BODY = r"""{
  static_assert(D == 256 && PARTITION_SIZE == 1024 &&
      KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E4M3 &&
      SEQ_LEN_ROUTE == 0 && !ANCHORED_SWA &&
      std::is_same_v<PARTIAL_T, float>, "Private scalar six-head contract");
  const int partition_idx = blockIdx.z;
  const int seq_len = seq_lens[0];
  const int start_token_idx = partition_idx * PARTITION_SIZE;
  if (seq_len <= 0 || start_token_idx >= seq_len ||
      partition_idx >= max_num_partitions) return;
  const int effective_num_partitions = min(max_num_partitions,
      max(active_num_partitions[0], (seq_len + PARTITION_SIZE - 1) / PARTITION_SIZE));
  if (partition_idx >= effective_num_partitions) return;
  const int part_tokens = min(PARTITION_SIZE, seq_len - start_token_idx);
  const int lane = threadIdx.x % kWarpSize;
  const int warp_idx = threadIdx.x / kWarpSize;
  const float score_scale = softmax_scale * k_scale;
  __shared__ __half q_shared[6][D];
  __shared__ float scores_shared[6][PARTITION_SIZE];
  __shared__ int block_idx_shared[PARTITION_SIZE];
  __shared__ int block_offset_shared[PARTITION_SIZE];
  for (int i = threadIdx.x; i < 6 * D; i += blockDim.x)
    q_shared[i / D][i % D] = q[(i / D) * q_stride1 + i % D];
  for (int i = threadIdx.x; i < part_tokens; i += blockDim.x) {
    const int token_idx = start_token_idx + i;
    const int logical_block = token_idx / block_size;
    block_idx_shared[i] = block_table[logical_block];
    block_offset_shared[i] = token_idx - logical_block * block_size;
  }
  __syncthreads();
  float local_max[6];
#pragma unroll
  for (int head = 0; head < 6; ++head) local_max[head] = -1.0e20f;
  for (int token_local = warp_idx; token_local < part_tokens;
       token_local += kWarpsPerBlock) {
    const int64_t k_index =
        static_cast<int64_t>(block_idx_shared[token_local]) * k_block_stride +
        static_cast<int64_t>(block_offset_shared[token_local]) * k_token_stride;
    float score[6] = {};
    // Each head retains d=lane,lane+32,... and the original warp reduction.
    // Only independent heads share the decoded K value.
#pragma unroll
    for (int d = lane; d < D; d += kWarpSize) {
      const float kv = flash_v100::load_kv_cache_float_unscaled<KV_DTYPE>(
          k_cache, k_index + d);
#pragma unroll
      for (int head = 0; head < 6; ++head)
        score[head] = fmaf(__half2float(q_shared[head][d]), kv, score[head]);
    }
#pragma unroll
    for (int head = 0; head < 6; ++head) {
      const float reduced = warp_reduce_sum(score[head]);
      if (lane == 0) {
        const float value = reduced * score_scale;
        scores_shared[head][token_local] = value;
        local_max[head] = fmaxf(local_max[head], value);
      }
    }
  }
  float inv_sum[6];
#pragma unroll
  for (int head = 0; head < 6; ++head) {
    const float part_max = block_reduce_max<kWarpsPerBlock>(local_max[head]);
    float local_sum = 0.f;
    for (int i = threadIdx.x; i < part_tokens; i += blockDim.x) {
      const float p = __expf(scores_shared[head][i] - part_max);
      scores_shared[head][i] = p;
      local_sum += p;
    }
    const float part_sum = block_reduce_sum<kWarpsPerBlock>(local_sum);
    inv_sum[head] = part_sum > 0.f ? 1.f / part_sum : 0.f;
    if (threadIdx.x == 0) {
      const int64_t stats_index = head * stats_stride1 + partition_idx;
      max_logits[stats_index] = part_max;
      exp_sums[stats_index] = part_sum;
    }
    // Complete reads of the reduction helper's shared result before the
    // next head reuses it; all 256 threads participate in every reduction.
    __syncthreads();
  }
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float acc[6] = {};
    for (int i = 0; i < part_tokens; ++i) {
      const int64_t v_index =
          static_cast<int64_t>(block_idx_shared[i]) * v_block_stride +
          static_cast<int64_t>(block_offset_shared[i]) * v_token_stride + d;
      const float vv =
          flash_v100::load_kv_cache_float_unscaled<KV_DTYPE>(v_cache, v_index);
#pragma unroll
      for (int head = 0; head < 6; ++head)
        acc[head] = fmaf(scores_shared[head][i], vv, acc[head]);
    }
#pragma unroll
    for (int head = 0; head < 6; ++head) {
      const float out_scale = inv_sum[head] * v_scale;
      const int64_t tmp_out_base = head * tmp_out_stride1 +
          static_cast<int64_t>(partition_idx) * tmp_out_stride2;
      tmp_out[tmp_out_base + d] = acc[head] * out_scale;
    }
  }
}
"""


def compact_page_map(partition: str) -> str:
    """Keep two page IDs instead of per-token IDs/offsets; preserve FMA order.

    The host validates page3296 and partition1024, so a partition spans at most
    two physical pages. Both PV segments visit exactly the original token order.
    """
    partition = replace_once(
        partition,
        "  __shared__ int block_idx_shared[PARTITION_SIZE];\n"
        "  __shared__ int block_offset_shared[PARTITION_SIZE];",
        "  __shared__ int block_idx_shared[2];\n"
        "  const int first_block = start_token_idx / block_size;\n"
        "  const int first_offset = start_token_idx - first_block * block_size;\n"
        "  const int first_page_tokens = min(part_tokens, block_size - first_offset);",
    )
    a = partition.index("  for (int i = threadIdx.x; i < part_tokens;")
    b = partition.index("  __syncthreads();", a)
    partition = (
        partition[:a]
        + """  if (threadIdx.x == 0) {
    block_idx_shared[0] = block_table[first_block];
    block_idx_shared[1] = first_page_tokens < part_tokens
        ? block_table[first_block + 1] : block_idx_shared[0];
  }
"""
        + partition[b:]
    )
    partition = replace_once(
        partition,
        """    const int64_t k_index =
        static_cast<int64_t>(block_idx_shared[token_local]) * k_block_stride +
        static_cast<int64_t>(block_offset_shared[token_local]) * k_token_stride;""",
        """    const bool second_page = token_local >= first_page_tokens;
    const int token_offset = second_page ? token_local - first_page_tokens
                                         : first_offset + token_local;
    const int64_t k_index =
        static_cast<int64_t>(block_idx_shared[second_page]) * k_block_stride +
        static_cast<int64_t>(token_offset) * k_token_stride;""",
    )
    a = partition.index("    for (int i = 0; i < part_tokens; ++i) {")
    b = partition.index("#pragma unroll\n    for (int head = 0;", a)
    partition = (
        partition[:a]
        + """    for (int page = 0; page < 2; ++page) {
      const int begin = page == 0 ? 0 : first_page_tokens;
      const int end = page == 0 ? first_page_tokens : part_tokens;
      int64_t v_index = static_cast<int64_t>(block_idx_shared[page]) *
          v_block_stride + (page == 0 ? first_offset * v_token_stride : 0) + d;
      for (int i = begin; i < end; ++i, v_index += v_token_stride) {
        const float vv =
            flash_v100::load_kv_cache_float_unscaled<KV_DTYPE>(v_cache, v_index);
#pragma unroll
        for (int head = 0; head < 6; ++head)
          acc[head] = fmaf(scores_shared[head][i], vv, acc[head]);
      }
    }
"""
        + partition[b:]
    )
    return partition


def template_function(source: str, name: str) -> tuple[int, str]:
    symbol = source.index("void " + name + "(")
    start = source.rfind("template <", 0, symbol)
    brace = source.index("{", symbol)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return start, source[start:end]


HOST = r"""
}  // namespace
at::Tensor scalar_attention_candidate(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    at::Tensor& out, const at::Tensor& table, const at::Tensor& lengths,
    at::Tensor& partial, at::Tensor& maximum, at::Tensor& sums,
    const at::Tensor& active, float scale, float k_scale, float v_scale) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kHalf &&
              q.sizes() == at::IntArrayRef({1, 6, 256}) && q.is_contiguous(),
              "Private scalar q1 probe requires contiguous FP16 [1,6,256]");
  TORCH_CHECK(k.dim() == 4 && k.size(2) == 1 && k.size(3) == 256 &&
              k.size(1) > 0 && k.scalar_type() == at::kByte &&
              v.sizes() == k.sizes() && v.scalar_type() == at::kByte,
              "Private scalar q1 probe requires E4M3 [pages,page,1,256]");
  for (const auto* t : {&k, &v})
    TORCH_CHECK(t->stride(3) == 1, "Head dimension must be contiguous");
  TORCH_CHECK(out.sizes() == q.sizes() && out.scalar_type() == at::kHalf &&
              out.is_contiguous(), "Output shape/dtype mismatch");
  TORCH_CHECK(table.dim() == 2 && table.size(0) == 1 && table.size(1) > 0 &&
              table.scalar_type() == at::kInt && table.is_contiguous() &&
              table.size(1) * k.size(1) <= 266240 &&
              lengths.sizes() == at::IntArrayRef({1}) &&
              lengths.scalar_type() == at::kInt && lengths.is_contiguous() &&
              active.sizes() == at::IntArrayRef({1}) &&
              active.scalar_type() == at::kInt && active.is_contiguous(),
              "Invalid scalar q1 page/length metadata");
  TORCH_CHECK(partial.sizes() == at::IntArrayRef({1, 6, 256, 256}) &&
              partial.scalar_type() == at::kFloat && partial.is_contiguous() &&
              maximum.sizes() == at::IntArrayRef({1, 6, 256}) &&
              sums.sizes() == maximum.sizes() &&
              maximum.scalar_type() == at::kFloat &&
              sums.scalar_type() == at::kFloat &&
              maximum.is_contiguous() && sums.is_contiguous(),
              "Scalar q1 requires unchanged FP32 partition workspaces");
  for (const auto* t : {&k, &v, static_cast<const at::Tensor*>(&out), &table,
      &lengths, static_cast<const at::Tensor*>(&partial),
      static_cast<const at::Tensor*>(&maximum),
      static_cast<const at::Tensor*>(&sums), &active})
    TORCH_CHECK(t->device() == q.device(), "Device mismatch");
  TORCH_CHECK(std::isfinite(scale) && std::isfinite(k_scale) &&
              std::isfinite(v_scale) && k_scale > 0 && v_scale > 0,
              "Finite positive KV scales required");
  c10::cuda::CUDAGuard guard(q.device());
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 7 && properties->minor == 0, "SM70 only");
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  flash_attention_decode_partition_kernel<256, 1024,
      flash_v100::KV_CACHE_DTYPE_FP8_E4M3, 0, false, float>
      <<<dim3(1, 6, 256), 256, 0, stream>>>(
      reinterpret_cast<const __half*>(q.data_ptr()), k.data_ptr(), v.data_ptr(),
      partial.data_ptr<float>(), maximum.data_ptr<float>(), sums.data_ptr<float>(),
      table.data_ptr<int>(), lengths.data_ptr<int>(), active.data_ptr<int>(),
      1, table.size(1), 256, 6, 1, k.size(1), q.stride(0), q.stride(1),
      partial.stride(0), partial.stride(1), partial.stride(2),
      maximum.stride(0), maximum.stride(1),
      k.stride(0), k.stride(1), k.stride(2),
      v.stride(0), v.stride(1), v.stride(2),
      scale, k_scale, v_scale, -1, -1, 0, 0, 0, nullptr, 0);
  flash_attention_decode_reduce_kernel<256, 1024, 0, float>
      <<<dim3(1, 6, 1), 256, 2 * 256 * sizeof(float), stream>>>(
      partial.data_ptr<float>(), maximum.data_ptr<float>(), sums.data_ptr<float>(),
      lengths.data_ptr<int>(), active.data_ptr<int>(),
      reinterpret_cast<__half*>(out.data_ptr()), 1, 256, 6,
      partial.stride(0), partial.stride(1), partial.stride(2),
      maximum.stride(0), maximum.stride(1), out.stride(0), out.stride(1),
      0, 0, 0, 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &scalar_attention_candidate);
}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pv-unroll", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--pv-prefetch", action="store_true")
    parser.add_argument("--share-kv-six-heads", action="store_true")
    parser.add_argument("--e4m3-lut", action="store_true")
    parser.add_argument("--compact-page-map", action="store_true")
    parser.add_argument(
        "--dynamic-shared-bytes", type=int, choices=(0, 4096), default=0
    )
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    original = args.source.read_text()
    start, partition = template_function(
        original, "flash_attention_decode_partition_kernel"
    )
    _, reduce = template_function(original, "flash_attention_decode_reduce_kernel")
    if args.pv_unroll != 1:
        partition = replace_once(
            partition,
            "    for (int i = 0; i < part_tokens; ++i) {",
            f"#pragma unroll {args.pv_unroll}\n"
            "    for (int i = 0; i < part_tokens; ++i) {",
        )
    if args.pv_prefetch:
        if args.pv_unroll != 1:
            parser.error("Isolate explicit prefetch from the unroll-hint experiment")
        a = partition.index("    for (int i = 0; i < part_tokens; ++i) {")
        b = partition.index("    const float out_scale =", a)
        partition = (
            partition[:a]
            + r"""    for (int base = 0; base < part_tokens; base += 4) {
      float values[4], weights[4];
#pragma unroll
      for (int stage = 0; stage < 4; ++stage) {
        const int i = base + stage;
        if (i < part_tokens) {
          const int physical_block = block_idx_shared[i];
          const int block_offset = block_offset_shared[i];
          const int64_t v_index =
              static_cast<int64_t>(physical_block) * v_block_stride +
              static_cast<int64_t>(block_offset) * v_token_stride +
              static_cast<int64_t>(kv_head_idx) * v_head_stride + d;
          values[stage] =
              flash_v100::load_kv_cache_float_unscaled<KV_DTYPE>(v_cache, v_index);
          weights[stage] = scores_shared[i];
        }
      }
#pragma unroll
      for (int stage = 0; stage < 4; ++stage) {
        if (base + stage < part_tokens)
          acc = fmaf(weights[stage], values[stage], acc);
      }
    }
"""
            + partition[b:]
        )
    host = HOST
    if args.share_kv_six_heads:
        if args.pv_prefetch or args.pv_unroll != 1:
            parser.error("Screen shared-head KV reuse independently")
        partition = partition[: partition.index("{")] + SHARED_HEADS_BODY
        host = replace_once(host, "<<<dim3(1, 6, 256),", "<<<dim3(1, 1, 256),")
    if args.compact_page_map:
        if not args.share_kv_six_heads:
            parser.error("The compact page map requires six-head KV sharing")
        partition = compact_page_map(partition)
    if args.e4m3_lut:
        if not args.share_kv_six_heads:
            parser.error("The LUT probe currently requires six-head KV sharing")
        partition = replace_once(
            partition,
            "  __shared__ __half q_shared[6][D];",
            "  __shared__ float kv_lut[256];\n"
            "  kv_lut[threadIdx.x] = flash_v100::fp8_e4m3fn_to_float(\n"
            "      static_cast<uint8_t>(threadIdx.x));\n"
            "  __shared__ __half q_shared[6][D];",
        )
        partition = replace_once(
            partition,
            "flash_v100::load_kv_cache_float_unscaled<KV_DTYPE>(\n"
            "          k_cache, k_index + d)",
            "kv_lut[static_cast<const uint8_t*>(k_cache)[k_index + d]]",
        )
        partition = replace_once(
            partition,
            "flash_v100::load_kv_cache_float_unscaled<KV_DTYPE>(v_cache, v_index)",
            "kv_lut[static_cast<const uint8_t*>(v_cache)[v_index]]",
        )
    if args.dynamic_shared_bytes:
        if not (args.compact_page_map and args.e4m3_lut):
            parser.error("Shared-memory occupancy tuning requires the compact LUT")
        # No data is stored here. The extra reservation tests the two-block
        # resource limit against the compact layout's three-block limit.
        host = replace_once(
            host,
            "<<<dim3(1, 1, 256), 256, 0, stream>>>",
            f"<<<dim3(1, 1, 256), 256, {args.dynamic_shared_bytes}, stream>>>",
        )
    source = original[:start] + partition + "\n" + reduce + "\n" + host
    directory = args.output_dir.resolve()
    if directory.exists():
        parser.error("Use a new directory for each native variant")
    sources = directory / "sources"
    root = args.source.parent.parent
    for sub in ("kernel", "include"):
        target = sources / sub
        target.mkdir(parents=True)
        for pattern in ("*.h", "*.cuh"):
            for header in (root / sub).glob(pattern):
                shutil.copy2(header, target)
    shutil.copy2(root / "LICENSE", sources)
    path = sources / "kernel/scalar-attention.cu"
    path.write_text(source)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    name = "sm70_scalar_attention_" + digest[:12]
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
    manifest = dict(
        input_source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        source_sha256=digest,
        module_name=name,
        pv_unroll=args.pv_unroll,
        pv_prefetch=args.pv_prefetch,
        share_kv_six_heads=args.share_kv_six_heads,
        e4m3_lut=args.e4m3_lut,
        compact_page_map=args.compact_page_map,
        dynamic_shared_bytes=args.dynamic_shared_bytes,
        max_context=262144,
        source_files={
            str(p.relative_to(sources)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(sources.rglob("*"))
            if p.is_file()
        },
        extra_cuda_cflags=flags,
        scope="Private scalar q1 fallback screen; no serving route",
    )
    if args.build:
        from torch.utils.cpp_extension import load

        build = directory / "build"
        build.mkdir()
        library = Path(
            load(
                name=name,
                sources=[str(path)],
                build_directory=str(build),
                extra_cuda_cflags=flags,
                extra_include_paths=[str(sources / "kernel"), str(sources / "include")],
                verbose=True,
            ).__file__
        )
        manifest.update(
            library=str(library),
            library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        )
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
