# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline-compile the DeepSeek V4 Triton route for SM70.

This check does not need a working CUDA driver. It lowers and assembles the
critical kernels to an SM70 cubin and rejects accidental native-FP8 PTX.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource, make_backend

SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE_ROOT))

import vllm.triton_utils as vllm_triton  # noqa: E402


@dataclass(frozen=True)
class KernelSpec:
    label: str
    module: str
    function: str
    runtime_types: tuple[str, ...]
    constants: dict[str, Any]
    num_warps: int = 4


def _specs() -> list[KernelSpec]:
    compressor_runtime = (
        "*fp32",
        "i64",
        "i64",
        "*fp32",
        "i64",
        "*fp32",
        "i64",
        "*fp32",
        "i64",
        "*i32",
        "*i64",
        "*i64",
        "*i32",
        "i64",
        "i32",
        "*fp16",
        "fp32",
        "*fp16",
        "i64",
        "*u8",
        "*i64",
        "i32",
    )
    return [
        KernelSpec(
            "q_kv_rmsnorm",
            "vllm.models.deepseek_v4.common.ops.fused_qk_rmsnorm",
            "_fused_q_kv_rmsnorm_kernel",
            (
                "*fp16",
                "*fp16",
                "*fp16",
                "i64",
                "i64",
                "*fp16",
                "*fp16",
                "*fp16",
                "i64",
                "i64",
                "fp32",
            ),
            {"Q_SIZE": 1536, "KV_SIZE": 512, "BLOCK_SIZE": 2048},
        ),
        KernelSpec(
            "mtp_input_rmsnorm",
            "vllm.models.deepseek_v4.common.ops.fused_mtp_input_rmsnorm",
            "_fused_mtp_input_rmsnorm_kernel",
            (
                "*fp16",
                "*i64",
                "*fp16",
                "*fp16",
                "*fp16",
                "*fp16",
                "*fp16",
                "fp32",
            ),
            {"HIDDEN": 4096, "HC_MULT": 4, "BLOCK_SIZE": 4096},
        ),
        KernelSpec(
            "mtp_head_rmsnorm",
            "vllm.models.deepseek_v4.common.ops.fused_mtp_input_rmsnorm",
            "_mtp_shared_head_rmsnorm_kernel",
            ("*fp16", "*fp16", "*fp16", "fp32"),
            {"HIDDEN": 4096, "BLOCK_SIZE": 4096},
        ),
        KernelSpec(
            "kv_pack",
            "vllm.models.deepseek_v4.common.ops.cache_utils",
            "quantize_and_insert_k_kernel",
            ("*fp16", "*i64", "*u8", "i32"),
            {
                "input_dim": 512,
                "fp8_dim": 448,
                "bf16_dim": 64,
                "scale_dim": 8,
                "quant_block": 64,
                "cache_block_size": 64,
                "token_data_size": 576,
                "block_stride": 64 * 584,
                "fp8_max": 448.0,
                "n_quant_blocks": 8,
                "use_software_fp8": True,
            },
        ),
        KernelSpec(
            "kv_gather",
            "vllm.models.deepseek_v4.common.ops.cache_utils",
            "_dequantize_and_gather_k_kernel",
            ("*fp16", "i64", "i64", "*u8", "*i32", "*i32", "i32", "*i32"),
            {
                "max_blocks_per_seq": 4096,
                "fp8_dim": 448,
                "bf16_dim": 64,
                "scale_dim": 8,
                "quant_block": 64,
                "cache_block_size": 64,
                "token_data_size": 576,
                "block_stride": 64 * 584,
                "output_dim": 512,
                "fp8_max": 448.0,
                "n_quant_blocks": 7,
                "use_software_fp8": True,
            },
        ),
        KernelSpec(
            "qnorm_rope",
            "vllm.models.deepseek_v4.sm70.qnorm_rope_kv_fp8_insert",
            "_sm70_qnorm_rope_kernel",
            ("*fp16", "*fp16", "*fp16", "*i64", "*fp16", "i32"),
            {
                "eps": 1e-6,
                "num_heads": 16,
                "HEAD_DIM": 512,
                "ROPE_DIM": 64,
                "NOPE_DIM": 448,
                "HALF_ROPE": 32,
            },
        ),
        KernelSpec(
            "inverse_rope",
            "vllm.models.deepseek_v4.sm70.projection",
            "_sm70_inverse_rope_kernel",
            ("*fp16", "*fp16", "*i64", "*fp16"),
            {
                "num_heads": 16,
                "head_dim": 512,
                "rope_dim": 64,
                "nope_dim": 448,
                "half_rope": 32,
            },
            1,
        ),
        KernelSpec(
            "sparse_prefill",
            "vllm.models.deepseek_v4.sm70.sparse_kernels",
            "_sm70_sparse_gathered_kernel",
            (
                "*fp16",
                "*fp16",
                "*i32",
                "*i32",
                "*fp32",
                "*fp16",
                "i64",
                "i64",
                "i64",
                "i64",
                "i64",
                "i64",
                "i32",
                "i32",
                "fp32",
            ),
            {"INDEX_WIDTH": 640, "BLOCK_H": 8, "BLOCK_K": 16, "BLOCK_D": 512},
        ),
        KernelSpec(
            "sparse_decode_c128",
            "vllm.models.deepseek_v4.sm70.sparse_kernels",
            "_sm70_sparse_paged_fp8_kernel",
            (
                "*fp16",
                "*u8",
                "*i32",
                "*i32",
                "*u8",
                "*i32",
                "*i32",
                "*fp32",
                "*fp16",
                "i64",
                "i64",
                "i64",
                "i64",
                "i64",
                "i64",
                "i64",
                "i64",
                "i32",
                "i32",
                "i32",
                "i32",
                "fp32",
                "i32",
            ),
            {
                "HAS_EXTRA": True,
                "MAIN_WIDTH": 128,
                "BLOCK_H": 8,
                "BLOCK_K": 16,
                "NOPE_DIM": 448,
                "NOPE_BLOCK": 512,
                "ROPE_DIM": 64,
            },
        ),
        KernelSpec(
            "sparse_decode_swa",
            "vllm.models.deepseek_v4.sm70.sparse_kernels",
            "_sm70_sparse_paged_fp8_kernel",
            (
                "*fp16",
                "*u8",
                "*i32",
                "*i32",
                "*u8",
                "*i32",
                "*i32",
                "*fp32",
                "*fp16",
                "i64",
                "i64",
                "i64",
                "i64",
                "i64",
                "i64",
                "i64",
                "i64",
                "i32",
                "i32",
                "i32",
                "i32",
                "fp32",
                "i32",
            ),
            {
                "HAS_EXTRA": False,
                "MAIN_WIDTH": 128,
                "BLOCK_H": 8,
                "BLOCK_K": 16,
                "NOPE_DIM": 448,
                "NOPE_BLOCK": 512,
                "ROPE_DIM": 64,
            },
        ),
        KernelSpec(
            "index_q_rope",
            "vllm.models.deepseek_v4.common.ops.fused_indexer_q",
            "_sm70_indexer_q_rope_kernel",
            (
                "*i64",
                "*fp16",
                "i64",
                "i64",
                "*fp16",
                "i64",
                "*fp32",
                "i64",
                "*fp16",
                "i64",
                "i64",
                "*fp32",
                "i64",
                "fp32",
                "fp32",
            ),
            {"HEAD_DIM": 128, "HALF_ROPE": 32},
            1,
        ),
        KernelSpec(
            "index_weighted_q",
            "vllm.models.deepseek_v4.sm70.indexer",
            "_weighted_query_kernel",
            ("*fp16", "*fp32", "*fp16", "i64", "i64", "i64", "i64"),
            {"num_heads": 64, "head_dim": 128, "BLOCK_D": 32},
        ),
        KernelSpec(
            "index_k_contiguous",
            "vllm.models.deepseek_v4.sm70.indexer",
            "_dequant_contiguous_index_k_kernel",
            ("*u8", "*fp32", "*fp16", "i64", "i64"),
            {"head_dim": 128},
        ),
        KernelSpec(
            "short_context_topk",
            "vllm.models.deepseek_v4.attention",
            "_fill_short_context_topk_indices",
            ("*i32", "*i64"),
            {"TOP_K": 512, "COMPRESS_RATIO": 4, "PADDED_TOP_K": 512},
            8,
        ),
        KernelSpec(
            "index_k_paged",
            "vllm.models.deepseek_v4.sm70.indexer",
            "_dequant_paged_index_k_kernel",
            (
                "*u8",
                "*i32",
                "*i32",
                "*fp16",
                "i64",
                "i64",
                "i64",
                "i64",
                "i64",
                "i32",
                "i32",
            ),
            {"head_dim": 128, "BLOCK_N": 16},
        ),
        KernelSpec(
            "compress_main_c4",
            "vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache",
            "_fused_kv_compress_norm_rope_insert_sparse_attn",
            compressor_runtime,
            {
                "HEAD_SIZE": 512,
                "TRITON_BLOCK_SIZE": 512,
                "STATE_WIDTH": 1024,
                "COMPRESS_RATIO": 4,
                "OVERLAP": True,
                "ROPE_HEAD_DIM": 64,
                "FP8_MAX": 448.0,
                "QUANT_BLOCK": 64,
                "TOKEN_STRIDE": 576,
                "SCALE_DIM": 8,
                "KV_BLOCK_STRIDE": 64 * 584,
                "FUSE_SAVE": False,
                "FRESH_BLOCK_SIZE": 1024,
                "USE_SOFTWARE_FP8": True,
            },
        ),
        KernelSpec(
            "compress_index_c4",
            "vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache",
            "_fused_kv_compress_norm_rope_insert_indexer_attn",
            compressor_runtime,
            {
                "HEAD_SIZE": 128,
                "TRITON_BLOCK_SIZE": 128,
                "STATE_WIDTH": 256,
                "COMPRESS_RATIO": 4,
                "OVERLAP": True,
                "ROPE_HEAD_DIM": 64,
                "FP8_MAX": 448.0,
                "QUANT_BLOCK": 128,
                "TOKEN_STRIDE": 128,
                "SCALE_DIM": 4,
                "KV_BLOCK_STRIDE": 64 * 132,
                "FUSE_SAVE": False,
                "FRESH_BLOCK_SIZE": 256,
                "USE_SOFTWARE_FP8": True,
            },
            1,
        ),
    ]


def main() -> None:
    # vLLM deliberately disables its Triton wrapper when no active driver is
    # present. Restore real Triton objects for this driver-free AOT check.
    vllm_triton.triton = triton
    vllm_triton.tl = tl

    target = GPUTarget("cuda", 70, 32)
    backend = make_backend(target)
    modules: dict[str, Any] = {}
    for spec in _specs():
        module = modules.get(spec.module)
        if module is None:
            module = importlib.import_module(spec.module)
            modules[spec.module] = module
        function = getattr(module, spec.function)
        runtime_types = iter(spec.runtime_types)
        signature = {
            name: ("constexpr" if name in spec.constants else next(runtime_types))
            for name in function.arg_names
        }
        options = backend.parse_options({"num_warps": spec.num_warps, "num_stages": 1})
        compiled = triton.compile(
            ASTSource(function, signature, spec.constants),
            target=target,
            options=options.__dict__,
        )
        ptx = compiled.asm["ptx"]
        if ".target sm_70" not in ptx:
            raise RuntimeError(f"{spec.label} did not target SM70")
        if "e4m3" in ptx:
            raise RuntimeError(f"{spec.label} emitted a native FP8 instruction")
        print(
            f"{spec.label:22s} ptx={len(ptx):7d} "
            f"cubin={len(compiled.asm['cubin']):7d} "
            f"shared={compiled.metadata.shared:6d}"
        )


if __name__ == "__main__":
    main()
