# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Literal

import torch

from vllm import envs
from vllm.platforms import current_platform

U4_GROUP_SIZES = (32, 64, 128)
GPTQ_GROUP_SIZES = (128,)
COMPRESSED_UINT4_GROUP_SIZES = (32, 128)
MXFP4_GROUP_SIZE = 32
NVFP4_GROUP_SIZE = 16
# SM70 packed NVFP4 GEMM needs complete 32-column tiles. N=8240 (Qwen
# GDN on TP2) is 16-aligned but corrupts the result without this padding.
NVFP4_OUTPUT_ALIGNMENT = 32
NVFP4_QPN4_DENSE_WORKSPACE_ELEMENTS = 5120 * 8704
STATE_ATTR = "_sm70_turbomind_linear"
SM70QuantBackend = Literal["auto", "marlin", "turbomind"]


@dataclass
class SM70TurboMindLinearState:
    weight: torch.Tensor
    scales: torch.Tensor
    group_size: int
    k_ld: int
    q_ld: int
    output_size: int
    op_kind: Literal["uint4", "mxfp4", "nvfp4", "nvfp4_qpn4", "nvfp4_qpn2_dense"]
    gated_silu: bool = False
    dense_weight_ptr: int = 0
    global_scale: float = 0.0
    use_scale_code: bool = False
    padded_output_size: int = 0
    # QPN2 launch configuration (split-K, independent accumulator chains).
    split_k: int = 0
    accumulator_chains: int = 0


# States retain only data_ptr(), so this cache owns the bounded allocation.
_nvfp4_qpn4_dense_workspaces: dict[tuple[int, torch.dtype], torch.Tensor] = {}


def clear_sm70_turbomind_workspaces() -> None:
    """Release process-global NVFP4 QPN4 dense workspaces."""
    _nvfp4_qpn4_dense_workspaces.clear()
    from vllm.model_executor.layers.quantization.utils.nvfp4_qpn2_dequant import (
        clear_nvfp4_qpn2_dense_workspaces,
    )

    clear_nvfp4_qpn2_dense_workspaces()
    _fp8_qpn8_dense_workspaces.clear()


def quant_backend() -> SM70QuantBackend:
    return envs.get_sm70_quant_backend()


def use_turbomind(default_enabled: bool) -> bool:
    return envs.use_sm70_turbomind(default_enabled)


def forces_marlin() -> bool:
    return envs.force_sm70_marlin()


def is_exact_sm70_cuda(tensor: torch.Tensor, enabled: bool) -> bool:
    if not enabled or not tensor.is_cuda:
        return False
    return torch.cuda.get_device_capability(tensor.device) == (7, 0)


def is_exact_sm70_cuda_platform() -> bool:
    """Return true only for Volta SM70 CUDA workers.

    Quant-method selection runs before a layer owns a CUDA tensor, so it
    cannot use :func:`is_exact_sm70_cuda`. Keep this platform check separate
    from the tensor-based helpers used by linear weight preparation.
    """
    return current_platform.is_cuda() and current_platform.is_device_capability((7, 0))


def should_use_mxfp4_moe_turbomind() -> bool:
    """Select the native MXFP4 MoE path only on exact SM70."""
    return is_exact_sm70_cuda_platform() and use_turbomind(
        envs.VLLM_SM70_MXFP4_TURBOMIND
    )


def should_use_nvfp4_moe_turbomind() -> bool:
    """Select the native NVFP4 MoE path only on exact SM70."""
    return is_exact_sm70_cuda_platform() and use_turbomind(
        envs.VLLM_SM70_NVFP4_TURBOMIND
    )


def should_prepare_turbomind(
    tensor: torch.Tensor,
    default_enabled: bool,
) -> bool:
    return is_exact_sm70_cuda(tensor, use_turbomind(default_enabled))


def should_prepare_turbomind_or_marlin(
    tensor: torch.Tensor,
    default_enabled: bool,
) -> bool:
    return is_exact_sm70_cuda(tensor, use_turbomind(default_enabled) or forces_marlin())


def is_turing_cuda(tensor: torch.Tensor, enabled: bool) -> bool:
    if not enabled or not tensor.is_cuda:
        return False
    return torch.cuda.get_device_capability(tensor.device) == (7, 5)


def is_pre_ampere_cuda_platform() -> bool:
    """Return true for Volta and Turing workers, judged on the worker's device.

    Quant-method selection runs before a layer owns a CUDA tensor. The
    capability is read from the device this process computes on: on a node
    that mixes card generations, device 0 of the visibility list answers for
    another card.
    """
    if not current_platform.is_cuda():
        return False
    device_id = (
        torch.accelerator.current_device_index() if torch.cuda.is_initialized() else 0
    )
    return current_platform.has_device_capability(70, device_id=device_id) and (
        not current_platform.has_device_capability(80, device_id=device_id)
    )


def is_turing_cuda_platform() -> bool:
    """Return true for Turing workers, judged on the worker's device."""
    if not current_platform.is_cuda():
        return False
    device_id = (
        torch.accelerator.current_device_index() if torch.cuda.is_initialized() else 0
    )
    return current_platform.is_device_capability((7, 5), device_id=device_id)


def should_prepare_turing_qpn2(
    tensor: torch.Tensor,
    default_enabled: bool,
) -> bool:
    """Turing takes the QPN2 decode kernels with a dense fp16 prefill.

    The TurboMind GEMMs are registered for exact SM70 only (``Sm70`` is
    ``Arch<700, 750>``), so Turing cannot take the Volta path; the QPN2
    kernels and the dequantization do not depend on the TurboMind registry.
    The same switches as on Volta apply.
    """
    return is_turing_cuda(tensor, use_turbomind(default_enabled))


def _get_u4_slices(x: torch.Tensor, dtype: torch.dtype) -> list[torch.Tensor]:
    if x.dtype == torch.int32:
        count = 8
    elif x.dtype == torch.uint8:
        count = 2
    else:
        raise TypeError(f"expected int32 or uint8 packed int4 tensor, got {x.dtype}")
    xs = []
    for _ in range(count):
        xs.append((x & 15).to(dtype))
        x = x >> 4
    return xs


def unpack_gptq_weight(qweight: torch.Tensor) -> torch.Tensor:
    xs = _get_u4_slices(qweight, torch.uint8)
    return torch.stack(xs, dim=1).reshape(-1, qweight.size(-1)).contiguous()


def unpack_gptq_zeros(qzeros: torch.Tensor) -> torch.Tensor:
    xs = _get_u4_slices(qzeros, torch.uint8)
    zeros = torch.stack(xs, dim=-1).reshape(qzeros.size(0), -1)
    return (zeros + 1).to(torch.float16).contiguous()


def unpack_compressed_weight(weight_packed: torch.Tensor) -> torch.Tensor:
    xs = _get_u4_slices(weight_packed, torch.uint8)
    weight = torch.stack(xs, dim=-1).reshape(*weight_packed.shape[:-1], -1)
    return weight.t().contiguous()


def unpack_compressed_zeros(weight_zero_point: torch.Tensor) -> torch.Tensor:
    xs = _get_u4_slices(weight_zero_point, torch.uint8)
    zeros = torch.stack(xs, dim=1).reshape(-1, weight_zero_point.size(-1))
    return zeros.t().to(torch.float16).contiguous()


def unpack_mxfp4_weight(weight_packed: torch.Tensor) -> torch.Tensor:
    if weight_packed.dim() > 2:
        weight_packed = torch.flatten(weight_packed, start_dim=-2)
    xs = _get_u4_slices(weight_packed, torch.uint8)
    weight = torch.flatten(
        torch.stack(xs, dim=-1),
        start_dim=-2,
    )
    return weight.t().contiguous()


def symmetric_int4_zeros_like(scales: torch.Tensor) -> torch.Tensor:
    return torch.full_like(scales, 8, dtype=torch.float16)


def _store_state(
    layer: torch.nn.Module,
    weight: torch.Tensor,
    scales: torch.Tensor,
    meta: torch.Tensor | None,
    group_size: int,
    output_size: int,
    op_kind: Literal["uint4", "mxfp4", "nvfp4", "nvfp4_qpn4", "nvfp4_qpn2_dense"],
    gated_silu: bool = False,
    dense_weight_ptr: int = 0,
    global_scale: float = 0.0,
    use_scale_code: bool = False,
    padded_output_size: int = 0,
    split_k: int = 0,
    accumulator_chains: int = 0,
) -> None:
    state = SM70TurboMindLinearState(
        weight=weight,
        scales=scales,
        group_size=group_size,
        k_ld=0 if meta is None else int(meta[0]),
        q_ld=0 if meta is None else int(meta[1]),
        output_size=output_size,
        op_kind=op_kind,
        gated_silu=gated_silu,
        dense_weight_ptr=dense_weight_ptr,
        global_scale=global_scale,
        use_scale_code=use_scale_code,
        padded_output_size=padded_output_size,
        split_k=split_k,
        accumulator_chains=accumulator_chains,
    )
    setattr(layer, STATE_ATTR, state)


def has_prepared_linear(layer: torch.nn.Module) -> bool:
    return getattr(layer, STATE_ATTR, None) is not None


def prepare_gptq_linear(
    layer: torch.nn.Module,
    group_size: int,
    interleave_gated_silu: bool = False,
) -> None:
    if group_size not in GPTQ_GROUP_SIZES:
        raise RuntimeError(
            f"SM70 TurboMind GPTQ supports group_size 128, but got {group_size}."
        )
    if not hasattr(torch.ops._C, "uint4_sm70_prepare"):
        raise RuntimeError(
            "VLLM_SM70_GPTQ_TURBOMIND=1 requires a build with CUDA arch 7.0 "
            "and the SM70 TurboMind extension."
        )
    from vllm import _sm70_ops as sm70_ops

    qweight = unpack_gptq_weight(layer.qweight.data)
    scales = layer.scales.data.to(torch.float16).contiguous()
    zeros = unpack_gptq_zeros(layer.qzeros.data)
    tm_weight, tm_scales, meta = sm70_ops.uint4_sm70_prepare(
        qweight, scales, zeros, group_size, interleave_gated_silu
    )
    _store_state(
        layer,
        tm_weight,
        tm_scales,
        meta,
        group_size,
        qweight.size(1),
        "uint4",
    )


def prepare_compressed_uint4_linear(
    layer: torch.nn.Module,
    group_size: int,
    symmetric: bool,
    interleave_gated_silu: bool = False,
) -> None:
    if group_size not in COMPRESSED_UINT4_GROUP_SIZES:
        raise RuntimeError(
            "SM70 TurboMind compressed-tensors int4 supports "
            f"group_size 32/128, but got {group_size}."
        )
    if not hasattr(torch.ops._C, "uint4_sm70_prepare"):
        raise RuntimeError(
            "VLLM_SM70_COMPRESSED_TENSORS_TURBOMIND=1 requires a build with "
            "CUDA arch 7.0 and the SM70 TurboMind extension."
        )
    from vllm import _sm70_ops as sm70_ops

    qweight = unpack_compressed_weight(layer.weight_packed.data)
    scales = layer.weight_scale.data.t().to(torch.float16).contiguous()
    if symmetric:
        zeros = symmetric_int4_zeros_like(scales)
    else:
        zeros = unpack_compressed_zeros(layer.weight_zero_point.data)
    tm_weight, tm_scales, meta = sm70_ops.uint4_sm70_prepare(
        qweight, scales, zeros, group_size, interleave_gated_silu
    )
    _store_state(
        layer,
        tm_weight,
        tm_scales,
        meta,
        group_size,
        qweight.size(1),
        "uint4",
    )


def prepare_mxfp4_linear(
    layer: torch.nn.Module,
    interleave_gated_silu: bool = False,
) -> None:
    if not hasattr(torch.ops._C, "mxfp4_sm70_prepare"):
        raise RuntimeError(
            "VLLM_SM70_MXFP4_TURBOMIND=1 requires a build with CUDA arch 7.0 "
            "and the SM70 TurboMind extension."
        )
    from vllm import _sm70_ops as sm70_ops

    qweight = unpack_mxfp4_weight(layer.weight_packed.data)
    scales = layer.weight_scale.data.t().contiguous()
    tm_weight, tm_scales, meta = sm70_ops.mxfp4_sm70_prepare(
        qweight, scales, MXFP4_GROUP_SIZE, interleave_gated_silu
    )
    _store_state(
        layer,
        tm_weight,
        tm_scales,
        meta,
        MXFP4_GROUP_SIZE,
        qweight.size(1),
        "mxfp4",
    )


def prepare_nvfp4_linear(
    layer: torch.nn.Module,
    interleave_gated_silu: bool = False,
) -> None:
    if not hasattr(torch.ops._C, "nvfp4_sm70_prepare"):
        raise RuntimeError(
            "VLLM_SM70_NVFP4_TURBOMIND=1 requires a build with CUDA arch 7.0 "
            "and the SM70 TurboMind NVFP4 extension."
        )
    from vllm import _sm70_ops as sm70_ops

    qweight = unpack_mxfp4_weight(layer.weight.data)
    scales = (
        (
            layer.weight_scale.data.t().to(torch.float32)
            * layer.weight_global_scale.to(torch.float32)
        )
        .to(torch.float16)
        .contiguous()
    )
    output_size = qweight.size(1)
    padded_output_size = (
        (output_size + NVFP4_OUTPUT_ALIGNMENT - 1) // NVFP4_OUTPUT_ALIGNMENT
    ) * NVFP4_OUTPUT_ALIGNMENT
    if padded_output_size != output_size:
        if interleave_gated_silu:
            raise RuntimeError(
                "SM70 TurboMind NVFP4 gated-SiLU does not support output padding."
            )
        padded_qweight = torch.zeros(
            (qweight.size(0), padded_output_size),
            dtype=qweight.dtype,
            device=qweight.device,
        )
        padded_scales = torch.zeros(
            (scales.size(0), padded_output_size),
            dtype=scales.dtype,
            device=scales.device,
        )
        padded_qweight[:, :output_size].copy_(qweight)
        padded_scales[:, :output_size].copy_(scales)
        qweight = padded_qweight
        scales = padded_scales
    tm_weight, tm_scales, meta = sm70_ops.nvfp4_sm70_prepare(
        qweight, scales, NVFP4_GROUP_SIZE, interleave_gated_silu
    )
    _store_state(
        layer,
        tm_weight,
        tm_scales,
        meta,
        NVFP4_GROUP_SIZE,
        output_size,
        "nvfp4",
        interleave_gated_silu,
        padded_output_size=padded_output_size,
    )


# Mirror of ``kQpn2DispatchMaxRows`` in nvfp4_qpn2_sm70.cu: rows up to this
# take the QPN2 decode kernels, larger M takes the dense prefill.
QPN2_DISPATCH_MAX_ROWS = 32
QPN2_GROUP_SIZE = 16

# Launch configurations (split-K, accumulator chains) measured on the
# Qwen3.8 TP2 shapes; other shapes take the heuristic below.
_QPN2_LAUNCH_TABLE: dict[tuple[int, int], tuple[int, int]] = {
    (1536, 5120): (16, 2),
    (4352, 5120): (16, 2),
    (5120, 8704): (8, 2),
    (5120, 4096): (16, 2),
    (5120, 2048): (32, 2),
    (5120, 62080): (8, 1),
    (5120, 3584): (16, 2),
}


def qpn2_launch_config(k: int, n: int) -> tuple[int, int]:
    """Split-K and accumulator chains for a QPN2 GEMM of ``[n, k]``."""
    groups = k // QPN2_GROUP_SIZE
    config = _QPN2_LAUNCH_TABLE.get((k, n))
    if config is not None and groups % config[0] == 0:
        return config
    # Smallest split that puts about 640 warps in flight (80 SMs x 8).
    for split_k in (8, 16, 32):
        if groups % split_k == 0 and (n // 32) * split_k >= 640:
            return split_k, 2 if split_k >= 16 else 1
    for split_k in (16, 8):
        if groups % split_k == 0:
            return split_k, 2 if split_k >= 16 else 1
    raise RuntimeError(f"no QPN2 launch configuration for K={k}, N={n}")


def pad_qpn2_output_rows(
    weight: torch.Tensor, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pad checkpoint-native output rows to QPN2's 32-column contract."""
    logical_n = weight.shape[0]
    physical_n = (logical_n + 31) // 32 * 32
    if physical_n == logical_n:
        return weight, scales, physical_n
    padded_weight = weight.new_zeros((physical_n, weight.shape[1]))
    padded_scales = scales.new_zeros((physical_n, scales.shape[1]))
    padded_weight[:logical_n].copy_(weight)
    padded_scales[:logical_n].copy_(scales)
    return padded_weight, padded_scales, physical_n


def prepare_nvfp4_qpn2_dense_linear(layer: torch.nn.Module) -> None:
    """Prepare the QPN2 prepack as the only resident layout of an NVFP4 linear.

    Decode (M <= ``QPN2_DISPATCH_MAX_ROWS``) runs the QPN2 kernels on it;
    larger M dequantizes into a transient fp16 buffer and runs cuBLAS. No
    TurboMind weight is built, so this does not need the TurboMind registry.
    """
    if not hasattr(torch.ops._C, "nvfp4_qpn2_prepare_sm70"):
        raise RuntimeError(
            "The pre-Ampere NVFP4 QPN2 path requires a build with CUDA arch "
            "7.0 and the SM70 TurboMind NVFP4 extension."
        )
    from vllm import _sm70_ops as sm70_ops

    # Registers the dispatch op now, at weight loading, so it exists before
    # the first forward is traced.
    from vllm.model_executor.layers.quantization.utils import (  # noqa: F401
        nvfp4_qpn2_dequant,
    )

    weight, scales, padded_output_size = pad_qpn2_output_rows(
        layer.weight.data, layer.weight_scale.data
    )
    codes, qpn2_scales = sm70_ops.nvfp4_qpn2_prepare_sm70(weight, scales)
    output_size = int(layer.weight.shape[0])
    input_size = int(layer.weight.shape[1]) * 2
    split_k, accumulator_chains = qpn2_launch_config(input_size, padded_output_size)
    _store_state(
        layer,
        codes,
        qpn2_scales,
        None,
        QPN2_GROUP_SIZE,
        output_size,
        "nvfp4_qpn2_dense",
        global_scale=float(layer.weight_global_scale.item()),
        padded_output_size=padded_output_size,
        split_k=split_k,
        accumulator_chains=accumulator_chains,
    )


# FP8 weights on Turing: QPN8 decode kernels plus the QPN8 dense prefill
# (dequantization into a transient fp16 [K, N] workspace and cuBLAS), both
# provided by fp8_qpn8_sm70.cu without the TurboMind registry.
_fp8_qpn8_dense_workspaces: dict[tuple[int, int, torch.device], torch.Tensor] = {}
FP8_QPN8_STATE_ATTR = "_sm70_fp8_qpn8_state"


class FP8QPN8LinearState:
    def __init__(
        self,
        codes: torch.Tensor,
        group_scales: torch.Tensor,
        output_size: int,
        split_k: int,
        accumulator_chains: int,
        prefetch_codes: bool,
        dense_weight_ptr: int,
    ) -> None:
        self.codes = codes
        self.group_scales = group_scales
        self.output_size = output_size
        self.split_k = split_k
        self.accumulator_chains = accumulator_chains
        self.prefetch_codes = prefetch_codes
        self.dense_weight_ptr = dense_weight_ptr


def fp8_qpn8_launch_config(k: int) -> tuple[int, int, bool]:
    """Split-K, accumulator chains and code prefetch for a QPN8 GEMM.

    The dispatcher admits split-K up to 16; the measured Qwen3.8 shapes use
    16 (K = 5120) and 12 (K = 1536) with two accumulator chains.
    """
    groups = k // QPN2_GROUP_SIZE
    for split_k in (16, 12, 8):
        if groups % split_k == 0:
            return split_k, 2, False
    raise RuntimeError(f"no QPN8 launch configuration for K={k}")


def get_fp8_qpn8_dense_workspace(k: int, n: int, device: torch.device) -> torch.Tensor:
    key = (k, n, device)
    workspace = _fp8_qpn8_dense_workspaces.get(key)
    if workspace is None:
        workspace = torch.empty(k, n, dtype=torch.float16, device=device)
        _fp8_qpn8_dense_workspaces[key] = workspace
    return workspace


def prepare_fp8_qpn8_dense_linear(
    layer: torch.nn.Module, weight: torch.Tensor, weight_scale: torch.Tensor
) -> None:
    """Prepare a per-tensor FP8 linear as QPN8 codes with channel scales.

    ``weight`` is the checkpoint-native fp8-e4m3fn ``[N, K]`` tensor,
    ``weight_scale`` its single scale. The QPN8 prepack takes channel scales,
    so the scale is broadcast over the rows; the dequantization is then the
    same product the reference path computes.
    """
    if not hasattr(torch.ops._C, "fp8_qpn8_prepare_sm70"):
        raise RuntimeError(
            "The Turing FP8 QPN8 path requires a build with CUDA arch 7.0 and "
            "the SM70 TurboMind FP8 extension."
        )
    from vllm import _sm70_ops as sm70_ops

    n, k = (int(dim) for dim in weight.shape)
    if n % 32 != 0 or k % 16 != 0:
        raise RuntimeError(f"QPN8 needs N % 32 == 0 and K % 16 == 0, got N={n}, K={k}.")
    channel_scales = (
        weight_scale.to(torch.float32).reshape(1, 1).expand(n, 1).contiguous()
    )
    codes, group_scales = sm70_ops.fp8_qpn8_prepare_sm70(
        weight.contiguous(), channel_scales
    )
    split_k, accumulator_chains, prefetch_codes = fp8_qpn8_launch_config(k)
    workspace = get_fp8_qpn8_dense_workspace(k, n, weight.device)
    setattr(
        layer,
        FP8_QPN8_STATE_ATTR,
        FP8QPN8LinearState(
            codes,
            group_scales,
            n,
            split_k,
            accumulator_chains,
            prefetch_codes,
            workspace.data_ptr(),
        ),
    )


def has_prepared_fp8_qpn8_linear(layer: torch.nn.Module) -> bool:
    return getattr(layer, FP8_QPN8_STATE_ATTR, None) is not None


def apply_prepared_fp8_qpn8_linear(
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    state = getattr(layer, FP8_QPN8_STATE_ATTR)
    if x.dtype != torch.float16:
        raise RuntimeError(
            f"The Turing FP8 QPN8 path requires float16 activations, got {x.dtype}."
        )
    reshaped_x = x.reshape(-1, x.shape[-1])
    if reshaped_x.stride(-1) != 1:
        reshaped_x = reshaped_x.contiguous()
    out = torch.empty(
        (reshaped_x.shape[0], state.output_size), dtype=x.dtype, device=x.device
    )
    from vllm import _sm70_ops as sm70_ops

    sm70_ops.fp8_qpn8_dispatch_sm70_out(
        out,
        state.dense_weight_ptr,
        reshaped_x,
        state.codes,
        state.group_scales,
        state.split_k,
        state.accumulator_chains,
        state.prefetch_codes,
        False,
    )
    if bias is not None:
        out.add_(bias)
    return out.reshape(x.shape[:-1] + (state.output_size,))


def get_nvfp4_qpn4_dense_workspace(weight: torch.Tensor) -> torch.Tensor | None:
    device_index = weight.device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    cache_key = (device_index, torch.float16)
    workspace = _nvfp4_qpn4_dense_workspaces.get(cache_key)
    if workspace is not None:
        return workspace
    try:
        workspace = torch.empty(
            (NVFP4_QPN4_DENSE_WORKSPACE_ELEMENTS,),
            dtype=torch.float16,
            device=weight.device,
        )
    except torch.OutOfMemoryError:
        return None
    _nvfp4_qpn4_dense_workspaces[cache_key] = workspace
    return workspace


def prepare_nvfp4_qpn4_linear(
    layer: torch.nn.Module,
    workspace: torch.Tensor,
    gated_silu: bool,
) -> None:
    """Replace one accepted NVFP4 dense weight with memory-neutral QPN4."""
    from vllm import _sm70_ops as sm70_ops

    qweight = unpack_mxfp4_weight(layer.weight.data)
    use_scale_code = gated_silu or envs.VLLM_SM70_NVFP4_QPN4_DOWN_SCALE_CODE
    global_scale = 0.0
    if use_scale_code:
        global_scale = float(layer.weight_global_scale.detach().float().item())
        raw_scale_codes = layer.weight_scale.data.t().contiguous()
        packed_weight, packed_scales = sm70_ops.nvfp4_qpn4_prepare_scale_code_sm70(
            qweight, raw_scale_codes
        )
    else:
        fp16_scales = (
            (
                layer.weight_scale.data.t().to(torch.float32)
                * layer.weight_global_scale.to(torch.float32)
            )
            .to(torch.float16)
            .contiguous()
        )
        packed_weight, packed_scales = sm70_ops.nvfp4_qpn4_prepare_sm70(
            qweight, fp16_scales
        )
    _store_state(
        layer,
        packed_weight,
        packed_scales,
        None,
        NVFP4_GROUP_SIZE,
        qweight.size(1),
        "nvfp4_qpn4",
        gated_silu=gated_silu,
        dense_weight_ptr=workspace.data_ptr(),
        global_scale=global_scale,
        use_scale_code=use_scale_code,
    )


def apply_prepared_linear(
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    state = getattr(layer, STATE_ATTR)
    reshaped_x = x.reshape(-1, x.shape[-1])
    out_shape = x.shape[:-1] + (state.output_size,)
    kernel_output_size = state.padded_output_size or state.output_size
    out = torch.empty(
        (reshaped_x.shape[0], kernel_output_size),
        dtype=x.dtype,
        device=x.device,
    )
    from vllm import _sm70_ops as sm70_ops

    if state.op_kind == "uint4":
        sm70_ops.awq_gemm_sm70_out(
            out,
            reshaped_x,
            state.weight,
            state.scales,
            state.group_size,
            state.k_ld,
            state.q_ld,
        )
    elif state.op_kind == "mxfp4":
        sm70_ops.mxfp4_gemm_sm70_out(
            out,
            reshaped_x,
            state.weight,
            state.scales,
            state.group_size,
            state.k_ld,
            state.q_ld,
        )
    elif state.op_kind == "nvfp4":
        sm70_ops.nvfp4_gemm_sm70_out(
            out,
            reshaped_x,
            state.weight,
            state.scales,
            state.group_size,
            state.k_ld,
            state.q_ld,
        )
    elif state.op_kind == "nvfp4_qpn4":
        if reshaped_x.dtype != torch.float16:
            raise RuntimeError(
                f"SM70 NVFP4 QPN4 requires float16 activations, got {reshaped_x.dtype}."
            )
        if reshaped_x.stride(-1) != 1:
            reshaped_x = reshaped_x.contiguous()
        sm70_ops.nvfp4_qpn4_dispatch_sm70_out(
            out,
            state.dense_weight_ptr,
            reshaped_x,
            state.weight,
            state.scales,
            state.global_scale,
            state.use_scale_code,
            False,
        )
    elif state.op_kind == "nvfp4_qpn2_dense":
        if reshaped_x.dtype != torch.float16:
            raise RuntimeError(
                "The pre-Ampere NVFP4 QPN2 path requires float16 activations, "
                f"got {reshaped_x.dtype}."
            )
        if reshaped_x.stride(-1) != 1:
            reshaped_x = reshaped_x.contiguous()
        from vllm.model_executor.layers.quantization.utils import (
            nvfp4_qpn2_dequant,
        )

        out = nvfp4_qpn2_dequant.nvfp4_qpn2_dispatch_linear(
            reshaped_x,
            state.weight,
            state.scales,
            state.global_scale,
            kernel_output_size,
            reshaped_x.shape[1],
            state.split_k,
            state.accumulator_chains,
        )
    else:
        raise AssertionError(f"unknown SM70 TurboMind op kind: {state.op_kind}")
    if kernel_output_size != state.output_size:
        out = out[:, : state.output_size]
    if state.gated_silu and state.op_kind == "nvfp4":
        out_features = state.output_size // 2
        out = (
            out.reshape(reshaped_x.shape[0], out_features, 2)
            .transpose(1, 2)
            .reshape(reshaped_x.shape[0], state.output_size)
        )
    if bias is not None:
        out.add_(bias)
    return out.reshape(out_shape)


def apply_prepared_fused_silu_and_mul(
    layer: torch.nn.Module,
    x: torch.Tensor,
) -> torch.Tensor | None:
    state = getattr(layer, STATE_ATTR, None)
    if (
        state is None
        or state.op_kind not in ("nvfp4", "nvfp4_qpn4")
        or not state.gated_silu
    ):
        return None
    if x.dtype != torch.float16:
        raise RuntimeError(
            "SM70 TurboMind NVFP4 gated-SiLU requires float16 activations, "
            f"got {x.dtype}."
        )

    reshaped_x = x.reshape(-1, x.shape[-1])
    if reshaped_x.stride(-1) != 1:
        reshaped_x = reshaped_x.contiguous()
    out_features = state.output_size // 2
    out = torch.empty(
        (reshaped_x.shape[0], out_features),
        dtype=x.dtype,
        device=x.device,
    )
    if reshaped_x.shape[0] == 0:
        return out.reshape(*x.shape[:-1], out_features)

    from vllm import _sm70_ops as sm70_ops

    if state.op_kind == "nvfp4_qpn4":
        sm70_ops.nvfp4_qpn4_dispatch_sm70_out(
            out,
            state.dense_weight_ptr,
            reshaped_x,
            state.weight,
            state.scales,
            state.global_scale,
            state.use_scale_code,
            True,
        )
    else:
        sm70_ops.nvfp4_gemm_sm70_out(
            out,
            reshaped_x,
            state.weight,
            state.scales,
            state.group_size,
            state.k_ld,
            state.q_ld,
            True,
        )
    return out.reshape(*x.shape[:-1], out_features)
