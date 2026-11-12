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


# ---- v41q8: QPN8 HMMA small-M GEMM on 32x32 UE8M0 blocks (VLLM_V41_Q8) ----
_Q8: dict = {}


def q8_ready():
    """The v41q8 JIT extension (benchmarks/deepseek_v41/v41k/build.py)."""
    if "ok" not in _Q8:
        _Q8["ok"] = False
        if envs.VLLM_V41_Q8 == "1":
            from vllm.model_executor.layers.quantization.gguf_v41_moe import (
                default_v41k_library,
            )

            path = default_v41k_library().replace("v41k.so", "v41q8.so")
            if __import__("os").path.exists(path):
                torch.ops.load_library(path)
                _Q8["ok"] = True
    return _Q8["ok"]


def q8_split(n, k):
    """Measured on V100 at M=6 (V4.1 layer-3 projections, 877 MHz memory)."""
    groups = k // 16
    order = (16, 8, 32, 4) if n >= 4096 and k <= 2048 else (32, 16, 8, 4)
    if n >= 4096 and k <= 1280:
        order = (8, 16, 4, 32)
    for split in order:
        if groups % split == 0:
            return split
    return None


def q8_scales(scales):
    """[N/32, K/32] UE8M0 -> [K/32, N/32] FP16 holding 256 * 2^(e - 127)."""
    exponent = scales.view(torch.uint8).to(torch.int32) - 127 + 8
    if bool((exponent < -14).any()) or bool((exponent > 15).any()):
        return None
    return torch.pow(2.0, exponent.float()).half().t().contiguous()


def q8_prepare(weight, scales):
    """(packed codes, FP16 scales, split) or None when the shape or range fails."""
    n, k = weight.shape
    split = q8_split(n, k)
    half_scales = q8_scales(scales[: (n + 31) // 32])
    if n % 32 or k % 32 or split is None or half_scales is None:
        return None
    packed = torch.empty(n * k, dtype=torch.uint8, device=weight.device)
    torch.ops.v41q8.prepack(packed, weight.view(torch.uint8))
    return packed, half_scales, split


_ABLATE_DENSE = "dense" in envs.VLLM_V41_ABLATE
# Grouped wo_a in one QPN8 launch instead of per-group copies + GEMMs + cat.
_Q8_GROUPED = envs.VLLM_V41_Q8_GROUPED == "1"


def q8_linear(x, prepared, n, bias=None):
    packed, half_scales, split = prepared
    if _ABLATE_DENSE:  # timing experiments only (VLLM_V41_ABLATE)
        return torch.zeros(x.shape[0], n, dtype=torch.float16, device=x.device)
    out = torch.empty(x.shape[0], n, dtype=torch.float16, device=x.device)
    torch.ops.v41q8.gemm(out, x, packed, half_scales, split, 2)
    return out if bias is None else out + bias


# Batched decode (VLLM_V41_BATCH): M in 9..64 on the multi-row QPN8 kernel
# (v41q8.gemm_rows), which decodes each weight fragment once for all rows.
_ROWS_MAX = int(envs.VLLM_V41_Q8_ROWS_MAX)
_BATCH_ROWS = envs.VLLM_V41_BATCH == "1"
_ROWS_TICKETS: dict = {}


def q8_rows_config(m, n, k):
    """(warps, parts, row_tiles) measured on V4.1 shapes (rows_bench.py)."""
    row_tiles = 2 if m <= 16 else 4 if m <= 32 else 8
    warps = {2: 16 if n <= 2048 else 8, 4: 8, 8: 4}[row_tiles]
    if n >= 4096 and row_tiles == 2:
        warps = 8
    groups = k // 16
    while groups % warps:
        warps //= 2
    target = max(1, 256 // max(1, n // 32))
    parts = 1
    for candidate in range(1, target + 1):
        if groups % (warps * candidate) == 0:
            parts = candidate
    return warps, parts, row_tiles


def q8_rows(x, packed, half_scales, n, tickets=None):
    """x [M, K] (or grouped [M, G, K]) @ W^T for 8 < M <= 64."""
    m, k = x.shape[0], x.shape[-1]
    if tickets is None:
        tickets = _ROWS_TICKETS.get(packed.data_ptr())
        if tickets is None:
            tickets = _ROWS_TICKETS[packed.data_ptr()] = torch.zeros(
                n // 32, dtype=torch.int32, device=x.device
            )
    warps, parts, row_tiles = q8_rows_config(m, n, k)
    out = torch.empty(m, n, dtype=torch.float16, device=x.device)
    torch.ops.v41q8.gemm_rows(
        out, x.contiguous(), packed, half_scales, tickets, warps, parts, row_tiles
    )
    return out


def _rows_eligible(x):
    return 8 < x.shape[0] <= _ROWS_MAX and x.dtype == torch.float16 and x.is_cuda


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
    if _DEQUANT_MM_ROWS and x.shape[0] >= _DEQUANT_MM_ROWS:
        return _fp8_dequant_mm(x, weight, scales, bias)
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


@triton.jit
def _dequant_fp16(weight, scales, out, N: tl.constexpr, K: tl.constexpr):
    """One 32x32 block: E4M3 codes * 2^(e - 127) -> FP16 (as _linear's b)."""
    nb, kb = tl.program_id(0), tl.program_id(1)
    rows = nb * 32 + tl.arange(0, 32)
    cols = kb * 32 + tl.arange(0, 32)
    codes = tl.load(weight + rows[:, None] * K + cols[None, :], rows[:, None] < N, 0)
    exponent = tl.load(scales + nb * (K // 32) + kb).to(tl.uint32)
    scale = tl.where(exponent == 0, 1 << 22, exponent << 23)
    scale = scale.to(tl.float32, bitcast=True)
    value = (fp8_e4m3fn_bits_to_fp32_bitcast(codes) * scale).to(tl.float16)
    tl.store(out + rows[:, None] * K + cols[None, :], value, rows[:, None] < N)


# Prefill (M >= _DEQUANT_MM_ROWS): dequantize the weight once to FP16 and use
# cuBLAS tensor-core GEMM instead of the 16-row Triton tiles, which re-decode
# every weight tile once per 16 activation rows.
_DEQUANT_MM_ROWS = int(envs.VLLM_V41_DEQUANT_MM_ROWS)


def _fp8_dequant_mm(x, weight, scales, bias):
    n, k = weight.shape
    w16 = torch.empty(n, k, dtype=torch.float16, device=x.device)
    _dequant_fp16[(triton.cdiv(n, 32), k // 32)](
        weight.view(torch.uint8), scales.view(torch.uint8), w16, N=n, K=k, num_warps=4
    )
    out = torch.mm(x, w16.t())
    return out if bias is None else out + bias


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
        self.prepare_q8(layer)

    @staticmethod
    def prepare_q8(layer):
        """A second, QPN8 fragment-order copy for 1..8-row decode (HMMA).

        The checkpoint layout stays for prefill. Grouped ``wo_a`` keeps one
        prepared copy per 1024-row O group. Re-run after the scales change.
        """
        layer.v41_q8 = layer.v41_q8_groups = layer.v41_q8_grouped = None
        if not q8_ready() or not layer.weight.is_cuda:
            return
        weight, scales = layer.weight.data, layer.weight_scale_inv.data
        layer.v41_q8 = q8_prepare(weight, scales)
        if weight.shape[0] % 1024 == 0 and weight.shape[1] == 4096:
            layer.v41_q8_groups = [
                q8_prepare(
                    weight[g * 1024 : (g + 1) * 1024],
                    scales[g * 32 : (g + 1) * 32],
                )
                for g in range(weight.shape[0] // 1024)
            ]
            if _Q8_GROUPED and layer.v41_q8 is not None:
                # All groups in one launch (gemm_grouped) on the full prepack;
                # tickets re-arm themselves after every launch.
                tickets = torch.zeros(
                    weight.shape[0] // 32, dtype=torch.int32, device=weight.device
                )
                layer.v41_q8_grouped = (*layer.v41_q8[:2], tickets)

    def apply(self, layer, x, bias=None):
        prepared = getattr(layer, "v41_q8", None)
        if prepared is not None and 1 <= x.shape[0] <= 8 and x.is_contiguous():
            return q8_linear(x, prepared, layer.weight.shape[0], bias)
        if prepared is not None and _rows_eligible(x):
            out = q8_rows(x, prepared[0], prepared[1], layer.weight.shape[0])
            return out if bias is None else out + bias
        return fp8_block32_linear(x, layer.weight, layer.weight_scale_inv, bias)

    def apply_grouped(self, layer, x):
        if (
            x.ndim != 3
            or x.shape[2] != 4096
            or layer.weight.shape[0] != x.shape[1] * 1024
        ):
            raise ValueError("V4.1 grouped FP8 projection requires 1024 rows/group")
        grouped = getattr(layer, "v41_q8_grouped", None)
        if grouped is not None and 1 <= x.shape[0] <= 8:
            packed, half_scales, tickets = grouped
            out = torch.empty(
                x.shape[0], layer.weight.shape[0], dtype=torch.float16, device=x.device
            )
            # Measured best at K=4096, 2 groups/rank: 4 warps x 8 K-parts.
            torch.ops.v41q8.gemm_grouped(
                out, x.contiguous(), packed, half_scales, tickets, 4, 8, 2
            )
            return out
        if grouped is not None and _rows_eligible(x):
            packed, half_scales, tickets = grouped
            return q8_rows(x, packed, half_scales, layer.weight.shape[0], tickets)
        groups = getattr(layer, "v41_q8_groups", None)
        if groups is not None and 1 <= x.shape[0] <= 8 and all(groups):
            return torch.cat(
                [
                    q8_linear(x[:, group].contiguous(), groups[group], 1024)
                    for group in range(x.shape[1])
                ],
                dim=-1,
            )
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


class Q8Fused:
    """Same-input FP8 projections as one v41q8 GEMM (``dense_tm.fused``).

    Concatenated rows of e.g. wq_a+wkv or the shared expert's gate+up; the
    outputs are column views of one result.
    """

    def __init__(self, prepared, widths):
        self.prepared, self.widths = prepared, tuple(widths)
        self.n = sum(widths)
        # VLLM_V41_BATCH decode rows (9..64) take the multi-row kernel.
        self.max_m = _ROWS_MAX if _BATCH_ROWS else 8

    def __call__(self, x):
        if 8 < x.shape[0] <= self.max_m and not _ABLATE_DENSE:
            out = q8_rows(x, self.prepared[0], self.prepared[1], self.n)
            return [part.contiguous() for part in out.split(self.widths, dim=-1)]
        if len(self.widths) == 2 and 1 <= x.shape[0] <= 8 and not _ABLATE_DENSE:
            # Two parts: the kernel writes each contiguously (no split copies).
            packed, half_scales, split = self.prepared
            rows, dev = x.shape[0], x.device
            a = torch.empty(rows, self.widths[0], dtype=torch.float16, device=dev)
            b = torch.empty(rows, self.widths[1], dtype=torch.float16, device=dev)
            torch.ops.v41q8.gemm2(a, b, x.contiguous(), packed, half_scales, split, 2)
            return [a, b]
        out = q8_linear(x.contiguous(), self.prepared, self.n)
        # Consumers (shared SwiGLU, short RMSNorm) take contiguous matrices.
        return [part.contiguous() for part in out.split(self.widths, dim=-1)]


def attach_q8_fused(groups):
    """groups: iterable of same-input FP8 linear tuples. Returns the count."""
    count = 0
    for linears in groups:
        if not q8_ready() or any(
            getattr(layer, "v41_q8", None) is None for layer in linears
        ):
            continue
        weight = torch.cat([layer.weight.data for layer in linears])
        scales = torch.cat(
            [
                layer.weight_scale_inv.data[: layer.weight.shape[0] // 32]
                for layer in linears
            ]
        )
        prepared = q8_prepare(weight, scales)
        del weight, scales
        if prepared is None:
            continue
        linears[0].v41_small_m = Q8Fused(
            prepared, [layer.weight.shape[0] for layer in linears]
        )
        for layer in linears:
            layer.v41_q8 = None  # the fused copy serves decode
        count += 1
    return count


def q8_channel_prepare(weight):
    """FP16 [N, K] -> E4M3 codes (QPN8 order) with one scale per row.

    Lossy (round to nearest E4M3): for draft-only projections, whose outputs
    are verified by the target. Returns (packed, FP16 scales * 256, split).
    """
    n, k = weight.shape
    if not q8_ready() or n % 32 or k % 16 or (k // 16) % 16:
        return None
    scale = weight.float().abs().amax(dim=1).clamp_min(1e-30) / 448.0
    codes = (weight.float() / scale[:, None]).to(torch.float8_e4m3fn)
    packed = torch.empty(n * k, dtype=torch.uint8, device=weight.device)
    torch.ops.v41q8.prepack(packed, codes.view(torch.uint8).contiguous())
    split = 32 if (k // 16) % 32 == 0 else 16
    return packed, (scale * 256.0).half().contiguous(), split


def q8_channel_linear(x, prepared, n):
    packed, scales, split = prepared
    out = torch.empty(x.shape[0], n, dtype=torch.float16, device=x.device)
    torch.ops.v41q8.gemm_channel(out, x.contiguous(), packed, scales, split)
    return out
