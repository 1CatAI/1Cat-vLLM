# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 FP16 Tensor Core GEMM from unchanged FP8/UE8M0 32x32 blocks.

Dequantization is local to each MMA tile. There is no second resident dense
weight bank, and the checkpoint's native FP8 bytes/scales remain on device.
"""

import torch

import vllm.envs as envs
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_scale_parameter,
    create_fp8_weight_parameter,
    validate_fp8_block_shape,
)
from vllm.model_executor.parameter import BlockQuantScaleParameter
from vllm.models.deepseek_v4.common.ops.fp8_software import (
    fp8_e4m3fn_bits_to_fp32_bitcast,
)
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["M"])
def _linear(
    x,
    weight,
    scales,
    bias,
    out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BIAS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    columns = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, 32)
    accumulator = tl.zeros((BM, BN), tl.float32)
    for group in range(K // 32):
        at = group * 32 + k
        a = tl.load(x + rows[:, None] * K + at[None, :], rows[:, None] < M, 0)
        codes = tl.load(
            weight + columns[:, None] * K + at[None, :], columns[:, None] < N, 0
        )
        exponent = tl.load(
            scales + (columns // 32) * (K // 32) + group, columns < N, 127
        ).to(tl.uint32)
        scale = tl.where(exponent == 0, 1 << 22, exponent << 23).to(
            tl.float32, bitcast=True
        )
        b = (fp8_e4m3fn_bits_to_fp32_bitcast(codes) * scale[:, None]).to(tl.float16)
        accumulator = tl.dot(a, tl.trans(b), accumulator)
    if BIAS:
        accumulator += tl.load(bias + columns, columns < N, 0)[None, :]
    tl.store(
        out + rows[:, None] * N + columns[None, :],
        accumulator,
        (rows[:, None] < M) & (columns[None, :] < N),
    )


@triton.jit
def _linear_split_k(
    x,
    weight,
    scales,
    partials,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SPLIT: tl.constexpr,
):
    rows = tl.arange(0, 16)
    columns = tl.program_id(0) * 64 + tl.arange(0, 64)
    split = tl.program_id(1)
    k = tl.arange(0, 32)
    groups: tl.constexpr = triton.cdiv(K // 32, SPLIT)
    accumulator = tl.zeros((16, 64), tl.float32)
    for offset in range(groups):
        group = split * groups + offset
        at = group * 32 + k
        a = tl.load(
            x + rows[:, None] * K + at[None, :],
            (rows[:, None] < M) & (at[None, :] < K),
            0,
        )
        codes = tl.load(
            weight + columns[:, None] * K + at[None, :],
            (columns[:, None] < N) & (at[None, :] < K),
            0,
        )
        exponent = tl.load(
            scales + (columns // 32) * (K // 32) + group,
            (columns < N) & (group < K // 32),
            127,
        ).to(tl.uint32)
        scale = tl.where(exponent == 0, 1 << 22, exponent << 23).to(
            tl.float32, bitcast=True
        )
        b = (fp8_e4m3fn_bits_to_fp32_bitcast(codes) * scale[:, None]).to(tl.float16)
        accumulator = tl.dot(a, tl.trans(b), accumulator)
    tl.store(
        partials + split * M * N + rows[:, None] * N + columns[None, :],
        accumulator,
        (rows[:, None] < M) & (columns[None, :] < N),
    )


@triton.jit
def _reduce_split_k(
    partials,
    bias,
    out,
    N: tl.constexpr,
    ELEMENTS: tl.constexpr,
    SPLIT: tl.constexpr,
    BIAS: tl.constexpr,
):
    index = tl.program_id(0) * 256 + tl.arange(0, 256)
    value = tl.full((256,), 0, tl.float32)
    for split in range(SPLIT):
        value += tl.load(partials + split * ELEMENTS + index, index < ELEMENTS, 0)
    if BIAS:
        value += tl.load(bias + index % N, index < ELEMENTS, 0)
    tl.store(out + index, value, index < ELEMENTS)


def _small_split(n, k):
    # SM70 V100 has 80 SMs. Two CTAs per SM expose the short-M K loop's
    # memory/decode latency. The policy is independent of M, so individual
    # rows retain the same FP32 accumulation boundaries in every M1..6 batch.
    return min(64, k // 32, triton.cdiv(160, triton.cdiv(n, 64)))


def _gemv_ready():
    from vllm.model_executor.layers.quantization.gguf_v41_moe import _expert_major_ready

    return envs.VLLM_V41_FP8_GEMV == "1" and _expert_major_ready()


def fp8_block32_linear(x, weight, scales, bias=None, *, split_k=None):
    if (
        not x.is_cuda
        or x.dtype != torch.float16
        or x.ndim != 2
        or not x.is_contiguous()
        or weight.ndim != 2
        or weight.dtype != torch.float8_e4m3fn
        or not weight.is_contiguous()
        or scales.dtype != torch.float8_e8m0fnu
        or not scales.is_contiguous()
        or weight.shape[1] != x.shape[1]
        or x.shape[1] % 32
        or scales.shape != ((weight.shape[0] + 31) // 32, x.shape[1] // 32)
        or weight.device != x.device
        or scales.device != x.device
    ):
        raise ValueError("V4.1 draft linear requires raw FP8/UE8M0 32x32 blocks")
    result = torch.empty(x.shape[0], weight.shape[0], dtype=x.dtype, device=x.device)
    if 1 <= x.shape[0] <= 8 and _gemv_ready():
        # Bandwidth-bound small-M GEMV (exact e4m3 bit placement, FP32 sums).
        m, n, k = x.shape[0], weight.shape[0], x.shape[1]
        ctas = (n + 31) // 32
        split = max(1, min(k // 512 if k >= 512 else 1, (240 + ctas - 1) // ctas))
        partials = torch.empty(
            split * m * n if (split > 1 or bias is not None) else 1,
            device=x.device,
            dtype=torch.float32,
        )
        torch.ops.v41k.fp8_gemv(
            result,
            x,
            weight.view(torch.uint8),
            scales.view(torch.uint8),
            partials,
            split,
            bias,
        )
        return result
    if 1 <= x.shape[0] <= 6:
        m, n, k = x.shape[0], weight.shape[0], x.shape[1]
        split = _small_split(n, k) if split_k is None else split_k
        if not 1 <= split <= 64 or split * m * n * 4 > 32 * 1024**2:
            raise ValueError("V4.1 FP8 partial workspace exceeds 32 MiB")
        partials = torch.empty((split, m, n), device=x.device, dtype=torch.float32)
        _linear_split_k[(triton.cdiv(n, 64), split)](
            x,
            weight.view(torch.uint8),
            scales.view(torch.uint8),
            partials,
            m,
            n,
            k,
            split,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
        _reduce_split_k[(triton.cdiv(m * n, 256),)](
            partials,
            bias if bias is not None else result,
            result,
            n,
            m * n,
            split,
            bias is not None,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return result
    if x.shape[0]:
        _linear[(triton.cdiv(x.shape[0], 16), triton.cdiv(weight.shape[0], 64))](
            x,
            weight.view(torch.uint8),
            scales.view(torch.uint8),
            bias if bias is not None else result,
            result,
            M=x.shape[0],
            N=weight.shape[0],
            K=x.shape[1],
            BIAS=bias is not None,
            BM=16,
            BN=64,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return result


@register_weight_loader_v2_supported_method
class V41Fp8LinearMethod(LinearMethodBase):
    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **attributes,
    ):
        if params_dtype != torch.float16:
            raise ValueError("V4.1 draft linears require FP16 computation")
        validate_fp8_block_shape(
            layer,
            input_size,
            output_size,
            input_size_per_partition,
            output_partition_sizes,
            [32, 32],
        )
        loader = attributes["weight_loader"]
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = sum(output_partition_sizes)
        layer.weight_block_size = [32, 32]
        layer.register_parameter(
            "weight",
            create_fp8_weight_parameter(
                sum(output_partition_sizes), input_size_per_partition, loader
            ),
        )
        layer.register_parameter(
            "weight_scale_inv",
            create_fp8_scale_parameter(
                BlockQuantScaleParameter,
                output_partition_sizes,
                input_size_per_partition,
                [32, 32],
                loader,
                scale_dtype=torch.float8_e8m0fnu,
            ),
        )

    def process_weights_after_loading(self, layer):
        if (layer.weight_scale_inv.view(torch.uint8) == 255).any():
            raise ValueError("Nonfinite UE8M0 scale in V4.1 draft weights")

    def apply(self, layer, x, bias=None):
        return fp8_block32_linear(x, layer.weight, layer.weight_scale_inv, bias)

    def apply_grouped(self, layer, x):
        if (
            x.ndim != 3
            or x.shape[2] != 4096
            or layer.weight.shape[0] != x.shape[1] * 1024
        ):
            raise ValueError("V4.1 grouped FP8 projection requires 1024 rows/group")
        return torch.cat(
            [
                fp8_block32_linear(
                    x[:, group].contiguous(),
                    layer.weight[group * 1024 : (group + 1) * 1024],
                    layer.weight_scale_inv[group * 32 : (group + 1) * 32],
                    # Retain the full projection's K partitions when selecting
                    # disjoint output groups. Only redundant rows disappear.
                    split_k=_small_split(*layer.weight.shape),
                )
                for group in range(x.shape[1])
            ],
            dim=-1,
        )
