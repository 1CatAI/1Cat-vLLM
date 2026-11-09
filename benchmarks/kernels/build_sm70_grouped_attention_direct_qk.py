# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolate exact QK scores with one warp per M16/N16 tile.

Read native SM70 operand words directly from paged KV and contiguous Q. This
removes the Q/K shared panels from the independent score producer. The reference
is the existing staged producer using the original compensated QK helper.
This is a QK-stage experiment, not complete attention or serving admission.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from benchmarks.kernels.build_sm70_grouped_attention_candidate import replace_once
from benchmarks.kernels.build_sm70_grouped_attention_staged import PRODUCER

DIRECT = r"""
template <int PAGE_SIZE>
__global__ __launch_bounds__(32) void grouped_direct_qk_kernel(
    const __half* q, const uint8_t* k_cache, const int* page_ids,
    const int* row_lengths, float* scores, int page_block_size,
    int64_t k_block_stride, int64_t k_token_stride, float qk_scale) {
  int total_kv = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) total_kv = max(total_kv, row_lengths[i]);
  const int tile_start = blockIdx.x * 32;
  if (tile_start >= total_kv) return;
  const int lane = threadIdx.x;
  const int m_tile = blockIdx.y / 2;
  const int n_tile = blockIdx.y % 2;
  const int a_row = (lane & 3) + ((lane >> 4) & 1) * 4 + ((lane >> 2) & 1) * 8;
  const int b_row = (lane & 3) + ((lane >> 4) & 1) * 4 + ((lane >> 3) & 1) * 8;
  const int key_token = tile_start + n_tile * 16 + b_row;
  const bool valid = key_token < total_kv;
  int64_t key_offset = 0;
  if (valid) {
    const int size = PAGE_SIZE ? PAGE_SIZE : page_block_size;
    const int logical_page = key_token / size;
    const int offset = key_token - logical_page * size;
    key_offset = static_cast<int64_t>(page_ids[logical_page]) * k_block_stride +
                 static_cast<int64_t>(offset) * k_token_stride;
  }
  const __half* query_row = q + (m_tile * 16 + a_row) * 256;
  volta::fragment<volta::accumulator, 16, 16, 16, float> score;
  volta::fill_fragment(score, 0.0f);
  float correction[8] = {};
#pragma unroll 2
  for (int offset = 0; offset < 256; offset += 16) {
    volta::fragment<volta::matrix_a, 16, 16, 16, half, volta::row_major> af;
    volta::fragment<volta::matrix_b, 16, 16, 16, half, volta::col_major> bf;
    const uint4 qa = __ldg(reinterpret_cast<const uint4*>(query_row + offset));
    const uint4 qb = __ldg(reinterpret_cast<const uint4*>(query_row + offset + 8));
    af.x[0] = qa.x; af.x[1] = qa.y; af.x[2] = qa.z; af.x[3] = qa.w;
    af.x[4] = qb.x; af.x[5] = qb.y; af.x[6] = qb.z; af.x[7] = qb.w;
    uint64_t k0 = 0, k1 = 0;
    if (valid) {
      const uint64_t* key = reinterpret_cast<const uint64_t*>(
          k_cache + key_offset + offset);
      k0 = __ldg(key); k1 = __ldg(key + 1);
    }
    const uint4 ka = fp8_e4m3fn_vector_to_half8_fast(k0);
    const uint4 kb = fp8_e4m3fn_vector_to_half8_fast(k1);
    bf.x[0] = ka.x; bf.x[1] = ka.y; bf.x[2] = ka.z; bf.x[3] = ka.w;
    bf.x[4] = kb.x; bf.x[5] = kb.y; bf.x[6] = kb.z; bf.x[7] = kb.w;
    volta::fragment<volta::accumulator, 16, 16, 16, float> tile;
    volta::fill_fragment(tile, 0.0f);
    volta::mma_sync(tile, af, bf, tile);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float y = __fsub_rn(tile.x[i], correction[i]);
      const float sum = __fadd_rn(score.x[i], y);
      correction[i] = __fsub_rn(__fsub_rn(sum, score.x[i]), y);
      score.x[i] = sum;
    }
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) score.x[i] *= qk_scale;
  volta::store_matrix_sync(scores + static_cast<int64_t>(blockIdx.x) * 48 * 32 +
      m_tile * 16 * 32 + n_tile * 16, score, 32, volta::mem_row_major);
}
"""

HOST = r"""
void private_qk_stage(const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& table, const at::Tensor& lengths, at::Tensor& scores,
    float scale, bool direct) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kHalf && q.is_contiguous() &&
      q.sizes() == at::IntArrayRef({8,6,256}) &&
      reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0, "aligned Q [8,6,256]");
  TORCH_CHECK(k.dim() == 4 && k.size(1) > 0 && k.size(2) == 1 && k.size(3) == 256 &&
      k.scalar_type() == at::kByte && k.stride(3) == 1 &&
      k.stride(0) % 8 == 0 && k.stride(1) % 8 == 0 &&
      reinterpret_cast<uintptr_t>(k.data_ptr()) % 8 == 0, "aligned E4M3 K");
  TORCH_CHECK(table.dim() == 2 && table.size(0) == 1 && table.size(1) > 0 &&
      table.is_contiguous() && table.scalar_type() == at::kInt &&
      lengths.sizes() == at::IntArrayRef({8}) && lengths.is_contiguous() &&
      lengths.scalar_type() == at::kInt, "one paged sequence and eight lengths");
  const int tiles = (table.numel() * k.size(1) + 31) / 32;
  TORCH_CHECK(scores.is_contiguous() && scores.scalar_type() == at::kFloat &&
      scores.sizes() == at::IntArrayRef({tiles,48,32}) && std::isfinite(scale),
      "full FP32 tiled scores and finite scale");
  for (const auto* t : {&k, &table, &lengths, static_cast<const at::Tensor*>(&scores)})
    TORCH_CHECK(t->device() == q.device(), "same CUDA device");
  c10::cuda::CUDAGuard guard(q.device());
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0, "SM70 only");
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  if (direct) {
    auto kernel = grouped_direct_qk_kernel<0>;
    if (k.size(1) == 1648) kernel = grouped_direct_qk_kernel<1648>;
    if (k.size(1) == 3296) kernel = grouped_direct_qk_kernel<3296>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout, 0));
    kernel<<<dim3(tiles,6),32,0,stream>>>(
        reinterpret_cast<const __half*>(q.data_ptr()), k.data_ptr<uint8_t>(),
        table.data_ptr<int>(), lengths.data_ptr<int>(), scores.data_ptr<float>(),
        k.size(1), k.stride(0), k.stride(1), scale);
  } else {
    const bool paired = k.stride(0) % 16 == 0 && k.stride(1) % 16 == 0 &&
                        reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 == 0;
    auto kernel = paired ? grouped_staged_qk_kernel<0,true>
                         : grouped_staged_qk_kernel<0,false>;
    if (k.size(1) == 1648) kernel = paired ? grouped_staged_qk_kernel<1648,true>
                                         : grouped_staged_qk_kernel<1648,false>;
    if (k.size(1) == 3296) kernel = paired ? grouped_staged_qk_kernel<3296,true>
                                         : grouped_staged_qk_kernel<3296,false>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    kernel<<<tiles,256,0,stream>>>(
        reinterpret_cast<const __half*>(q.data_ptr()), k.data_ptr(),
        table.data_ptr<int>(), lengths.data_ptr<int>(), scores.data_ptr<float>(),
        k.size(1), k.stride(0), k.stride(1), k.stride(2), scale);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

PANEL = r"""
template <int PAGE_SIZE, int N, bool PAIRED>
__global__ __launch_bounds__(N * 6, 1) void grouped_panel_qk_kernel(
    const __half* q, const uint8_t* k_cache, const int* page_ids,
    const int* row_lengths, float* scores, int page_block_size,
    int64_t k_block_stride, int64_t k_token_stride, float qk_scale) {
  int total_kv = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) total_kv = max(total_kv, row_lengths[i]);
  const int tile_start = blockIdx.x * N;
  if (tile_start >= total_kv) return;
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int m_tile = warp / (N / 16);
  const int n_tile = warp % (N / 16);
  __shared__ __align__(16) __half shared_q[48 * 136];
  __shared__ __align__(16) __half shared_k[N * 136];
  __shared__ uint16_t lut[256];
  if (tid < 256) lut[tid] = fp8_e4m3fn_to_half_bits(static_cast<uint8_t>(tid));
  __syncthreads();
  volta::fragment<volta::accumulator, 16, 16, 16, float> score;
  volta::fill_fragment(score, 0.0f);
  float correction[8] = {};
  const int valid_rows = min(N, total_kv - tile_start);
  for (int panel_offset = 0; panel_offset < 256; panel_offset += 128) {
    for (int i = tid; i < 48 * 16; i += N * 6) {
      const int row = i / 16, vec = i % 16;
      reinterpret_cast<uint4*>(shared_q)[row * 17 + vec] =
          __ldg(reinterpret_cast<const uint4*>(q) + row * 32 +
                panel_offset / 8 + vec);
    }
    load_xqa_tc_kv_panel<PAGE_SIZE, false, N * 6,
        flash_v100::KV_CACHE_DTYPE_FP8_E4M3, PAIRED, true>(
        shared_k, k_cache, page_ids, valid_rows, 16, 17, tile_start, 0,
        page_block_size, 0, k_block_stride, k_token_stride, 0,
        panel_offset, tid, lut);
    for (int i = tid + valid_rows * 17; i < N * 17; i += N * 6)
      reinterpret_cast<uint4*>(shared_k)[i] = make_uint4(0,0,0,0);
    __syncthreads();
#pragma unroll 2
    for (int offset = 0; offset < 128; offset += 16) {
      volta::fragment<volta::matrix_a,16,16,16,half,volta::row_major> af;
      volta::fragment<volta::matrix_b,16,16,16,half,volta::col_major> bf;
      volta::load_matrix_sync(af, shared_q + m_tile * 16 * 136 + offset, 136);
      volta::load_matrix_sync(bf, shared_k + n_tile * 16 * 136 + offset, 136);
      volta::fragment<volta::accumulator,16,16,16,float> tile;
      volta::fill_fragment(tile, 0.0f);
      volta::mma_sync(tile, af, bf, tile);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float y = __fsub_rn(tile.x[i], correction[i]);
        const float sum = __fadd_rn(score.x[i], y);
        correction[i] = __fsub_rn(__fsub_rn(sum, score.x[i]), y);
        score.x[i] = sum;
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) score.x[i] *= qk_scale;
  const int output_tile = tile_start / 32 + n_tile / 2;
  if (output_tile * 32 < total_kv)
    volta::store_matrix_sync(scores + static_cast<int64_t>(output_tile) * 48 * 32 +
        m_tile * 16 * 32 + (n_tile % 2) * 16, score, 32, volta::mem_row_major);
}
"""


def reuse_panel_keys(source: str, separate_products: bool = False) -> str:
    """Reuse one K fragment across three independent compensated QK sums."""
    source = source.replace("N * 6", "N * 2")
    source = replace_once(
        source,
        "  const int m_tile = warp / (N / 16);\n  const int n_tile = warp % (N / 16);",
        "  const int n_tile = warp;",
    )
    source = replace_once(
        source,
        "  if (tid < 256) lut[tid] = "
        "fp8_e4m3fn_to_half_bits(static_cast<uint8_t>(tid));",
        "  for (int i = tid; i < 256; i += N * 2)\n"
        "    lut[i] = fp8_e4m3fn_to_half_bits(static_cast<uint8_t>(i));",
    )
    source = replace_once(
        source,
        "  volta::fragment<volta::accumulator, 16, 16, 16, float> score;\n"
        "  volta::fill_fragment(score, 0.0f);\n"
        "  float correction[8] = {};",
        "  volta::fragment<volta::accumulator, 16, 16, 16, float> score[3];\n"
        "#pragma unroll\n"
        "  for (int m = 0; m < 3; ++m) volta::fill_fragment(score[m], 0.0f);\n"
        "  float correction[3][8] = {};",
    )
    begin = source.index("#pragma unroll 2\n    for (int offset")
    end = source.index("    __syncthreads();\n  }", begin)
    source = (
        source[:begin]
        + r"""#pragma unroll 1
    for (int offset = 0; offset < 128; offset += 16) {
      volta::fragment<volta::matrix_b,16,16,16,half,volta::col_major> bf;
      volta::load_matrix_sync(bf, shared_k + n_tile * 16 * 136 + offset, 136);
#pragma unroll
      for (int m = 0; m < 3; ++m) {
        volta::fragment<volta::matrix_a,16,16,16,half,volta::row_major> af;
        volta::load_matrix_sync(af, shared_q + m * 16 * 136 + offset, 136);
        volta::fragment<volta::accumulator,16,16,16,float> tile;
        volta::fill_fragment(tile, 0.0f);
        volta::mma_sync(tile, af, bf, tile);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const float y = __fsub_rn(tile.x[i], correction[m][i]);
          const float sum = __fadd_rn(score[m].x[i], y);
          correction[m][i] = __fsub_rn(__fsub_rn(sum, score[m].x[i]), y);
          score[m].x[i] = sum;
        }
      }
    }
"""
        + source[end:]
    )
    if separate_products:
        source = replace_once(
            source,
            "#pragma unroll\n      for (int m = 0; m < 3; ++m) {",
            "      volta::fragment<volta::accumulator,16,16,16,float> products[3];\n"
            "#pragma unroll\n      for (int m = 0; m < 3; ++m) {",
        )
        source = replace_once(
            source,
            "        volta::fragment<volta::accumulator,16,16,16,float> tile;\n"
            "        volta::fill_fragment(tile, 0.0f);\n"
            "        volta::mma_sync(tile, af, bf, tile);\n#pragma unroll",
            "        volta::fill_fragment(products[m], 0.0f);\n"
            "        volta::mma_sync(products[m], af, bf, products[m]);\n"
            "      }\n#pragma unroll\n      for (int m = 0; m < 3; ++m) {\n"
            "#pragma unroll",
        )
        source = replace_once(
            source, "tile.x[i], correction[m][i]", "products[m].x[i], correction[m][i]"
        )
    begin = source.index("#pragma unroll\n  for (int i = 0; i < 8; ++i) score.x[i]")
    end = source.index("  __syncthreads();", begin)
    return (
        source[:begin]
        + r"""#pragma unroll
  for (int m = 0; m < 3; ++m) {
#pragma unroll
    for (int i = 0; i < 8; ++i) score[m].x[i] *= qk_scale;
    volta::store_matrix_sync(storage.scores + m * 16 * (N + 4) + n_tile * 16,
        score[m], N + 4, volta::mem_row_major);
  }
"""
        + source[end:]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-major", action="store_true")
    parser.add_argument("--warps-per-cta", type=int, choices=(1, 6), default=1)
    parser.add_argument("--predecoded-keys", action="store_true")
    parser.add_argument("--context-panel", type=int, choices=(32, 64, 128), default=32)
    parser.add_argument("--coalesced-output", action="store_true")
    parser.add_argument("--reuse-k", action="store_true")
    parser.add_argument("--separate-products", action="store_true")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if args.context_major and args.warps_per_cta != 1:
        parser.error("Six-warp CTAs already retain all Q tiles for one K tile")
    if args.context_panel != 32 and (
        args.context_major or args.warps_per_cta != 1 or args.predecoded_keys
    ):
        parser.error("Shared context panels use their own warp and KV layout")
    if args.coalesced_output and args.context_panel == 32:
        parser.error("Coalesced output reuses the shared context-panel storage")
    if args.reuse_k and not args.coalesced_output:
        parser.error("K-fragment reuse requires coalesced panel output")
    if args.separate_products and not args.reuse_k:
        parser.error("Separate independent products require K-fragment reuse")
    base = json.loads(args.base_manifest.read_text())
    if base["source_sha256"] != (
        "eb7a85511f581fcd22cf13619c85ed2f42a8cbc8b216bb3e632bf448f6b820e1"
    ):
        parser.error("Use the frozen service-checked P-layout source")
    parent = args.base_manifest.parent / "sources"
    for rel, digest in base["source_files"].items():
        if hashlib.sha256((parent / rel).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Parent source hash mismatch: {rel}")
    output = args.output_dir.resolve()
    sources = output / "sources"
    shutil.copytree(parent, sources)
    path = sources / "kernel/grouped-attention.cu"
    source = path.read_text()
    reference = PRODUCER[: PRODUCER.index("constexpr int kStagedPVThreads")]
    marker = "}  // namespace\n\nat::Tensor private_grouped_e4m3_fp32_paged("
    direct, host = DIRECT, HOST
    if args.context_panel != 32:
        direct = PANEL
        if args.coalesced_output:
            direct = replace_once(
                direct,
                "  __shared__ __align__(16) __half shared_q[48 * 136];\n"
                "  __shared__ __align__(16) __half shared_k[N * 136];",
                r"""  __shared__ __align__(16) union {
    struct {
      __half q[48 * 136];
      __half k[N * 136];
    } input;
    float scores[48 * (N + 4)];
  } storage;
  __half* shared_q = storage.input.q;
  __half* shared_k = storage.input.k;""",
            )
            begin = direct.index("  const int output_tile = tile_start / 32")
            direct = (
                direct[:begin]
                + r"""
  volta::store_matrix_sync(storage.scores + m_tile * 16 * (N + 4) + n_tile * 16,
      score, N + 4, volta::mem_row_major);
  __syncthreads();
  for (int i = tid; i < 48 * N; i += N * 6) {
    const int row = i / N, col = i % N;
    const int output_tile = tile_start / 32 + col / 32;
    if (output_tile * 32 < total_kv)
      scores[static_cast<int64_t>(output_tile) * 48 * 32 + row * 32 + col % 32] =
          storage.scores[row * (N + 4) + col];
  }
}
"""
            )
        if args.reuse_k:
            direct = reuse_panel_keys(direct, args.separate_products)
        a = host.index("    auto kernel = grouped_direct_qk_kernel<0>;")
        b = host.index("    kernel<<<", a)
        declarations = [
            "    const bool paired = k.stride(0) % 16 == 0 && k.stride(1) % 16 == 0;"
        ]
        for page in (0, 1648, 3296):
            prefix = (
                "auto kernel =" if page == 0 else f"if (k.size(1) == {page}) kernel ="
            )
            declarations.append(
                f"    {prefix} paired ? grouped_panel_qk_kernel<"
                f"{page},{args.context_panel},true>"
                f" : grouped_panel_qk_kernel<{page},{args.context_panel},false>;"
            )
        declarations.append(
            "    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,\n"
            "        cudaFuncAttributePreferredSharedMemoryCarveout, 100));\n"
        )
        host = host[:a] + "\n".join(declarations) + host[b:]
        width = args.context_panel // 32
        host = replace_once(
            host,
            "dim3(tiles,6),32",
            f"dim3((tiles + {width - 1}) / {width}),"
            f"{args.context_panel * (2 if args.reuse_k else 6)}",
        )
    if args.context_major:
        direct = direct.replace("blockIdx.x * 32", "blockIdx.y * 32")
        direct = direct.replace("blockIdx.y / 2", "blockIdx.x / 2")
        direct = direct.replace("blockIdx.y % 2", "blockIdx.x % 2")
        direct = direct.replace("(blockIdx.x) * 48", "(blockIdx.y) * 48")
        host = replace_once(host, "dim3(tiles,6),32", "dim3(6,tiles),32")
    if args.warps_per_cta == 6:
        direct = direct.replace("__launch_bounds__(32)", "__launch_bounds__(192)")
        direct = direct.replace("lane = threadIdx.x;", "lane = threadIdx.x & 31;")
        direct = direct.replace("blockIdx.y / 2", "(threadIdx.x / 32) / 2")
        direct = direct.replace("blockIdx.y % 2", "(threadIdx.x / 32) % 2")
        host = replace_once(host, "dim3(tiles,6),32", "dim3(tiles),192")
    if args.predecoded_keys:
        direct = replace_once(direct, "const uint8_t* k_cache", "const __half* k_cache")
        begin = direct.index("    uint64_t k0 = 0, k1 = 0;")
        end = direct.index("    bf.x[0] = ka.x;", begin)
        direct = (
            direct[:begin]
            + r"""
    uint4 ka = make_uint4(0,0,0,0), kb = make_uint4(0,0,0,0);
    if (valid) {
      const __half* key = k_cache + key_offset + offset;
      ka = __ldg(reinterpret_cast<const uint4*>(key));
      kb = __ldg(reinterpret_cast<const uint4*>(key + 8));
    }
"""
            + direct[end:]
        )
        host = replace_once(
            host,
            "k.scalar_type() == at::kByte",
            "k.scalar_type() == (direct ? at::kHalf : at::kByte)",
        )
        host = replace_once(
            host,
            "k.data_ptr<uint8_t>()",
            "reinterpret_cast<const __half*>(k.data_ptr())",
        )
    source = replace_once(source, marker, reference + direct + marker)
    marker = "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {"
    source = replace_once(
        source, marker, host + marker + '\n  m.def("qk_stage", &private_qk_stage);'
    )
    path.write_text(source)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    name = "sm70_direct_qk_" + digest[:12]
    manifest = {
        **{k: v for k, v in base.items() if k not in ("library", "library_sha256")},
        "input_source_sha256": base["source_sha256"],
        "source_sha256": digest,
        "module_name": name,
        "warps_per_cta": args.warps_per_cta,
        "context_major": args.context_major,
        "predecoded_keys": args.predecoded_keys,
        "context_panel": args.context_panel,
        "coalesced_output": args.coalesced_output,
        "reuse_k": args.reuse_k,
        "separate_products": args.separate_products,
        "scope": "Independent QK scores only; not full attention performance",
        "source_files": {
            str(p.relative_to(sources)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(sources.rglob("*"))
            if p.is_file()
        },
    }
    if args.build:
        from torch.utils.cpp_extension import load

        build = output / "build"
        build.mkdir()
        library = Path(
            load(
                name=name,
                sources=[str(path)],
                extra_include_paths=[str(sources / "include"), str(sources / "kernel")],
                extra_cuda_cflags=base["extra_cuda_cflags"],
                build_directory=str(build),
                verbose=True,
            ).__file__
        ).resolve()
        manifest["library"] = str(library)
        manifest["library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
