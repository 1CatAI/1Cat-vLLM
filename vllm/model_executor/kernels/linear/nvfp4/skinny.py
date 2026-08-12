# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math

import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig

logger = init_logger(__name__)

_SELF_CHECK_TOL = 3e-2
_runtime_ok = True
_validated_shapes: set[tuple[int, int]] = set()
_route_log_seen: set[tuple[str, int, torch.dtype]] = set()


def qpn_prepack(
    codes: torch.Tensor, scales: torch.Tensor
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Prepack checkpoint-native NVFP4 bytes for the QPN fragment map.

    The result is a pure permutation of weight nibbles and scale bytes. It is
    built once at weight load and consumed by the M=4..16 QPN kernel.
    """
    if codes.dim() != 2 or scales.dim() != 2:
        raise ValueError("NVFP4 codes and scales must both be two-dimensional.")
    if codes.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise TypeError("QPN prepack expects uint8 views of NVFP4 codes and scales.")
    if codes.device != scales.device:
        raise ValueError("NVFP4 codes and scales must be on the same device.")

    n, k2 = codes.shape
    k = k2 * 2
    if scales.shape != (n, k // 16):
        raise ValueError(
            "NVFP4 scale shape mismatch: expected "
            f"{(n, k // 16)}, got {tuple(scales.shape)}."
        )
    if n % 32 or k % 64:
        return None, None

    device = codes.device
    tiles, groups = n // 32, k // 16
    lane = torch.arange(32, device=device)
    col = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) > 0).long() * 4
    korder = torch.tensor(
        [0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15],
        device=device,
    )
    nibbles = torch.stack([codes & 0xF, codes >> 4], dim=-1).view(n, k)
    group = torch.arange(groups, device=device)
    kidx = group.view(groups, 1) * 16 + korder.view(1, 16)
    qcodes = torch.empty(tiles, groups, 32, 8, dtype=torch.uint8, device=device)
    qscales = torch.empty(tiles, groups, 32, dtype=torch.uint8, device=device)

    # Broadcast indices are much larger than the payload. Bound temporary
    # storage to roughly 300 MiB even for a large vocabulary projection.
    tile_chunk = max(1, 36864 // groups)
    for tile_start in range(0, tiles, tile_chunk):
        tile_end = min(tile_start + tile_chunk, tiles)
        tile_count = tile_end - tile_start
        ncol = torch.arange(tile_start, tile_end, device=device).view(
            tile_count, 1
        ) * 32 + col.view(1, 32)
        selected = nibbles[
            ncol.view(tile_count, 1, 32, 1).expand(tile_count, groups, 32, 16),
            kidx.view(1, groups, 1, 16).expand(tile_count, groups, 32, 16),
        ]
        qcodes[tile_start:tile_end] = selected[..., 0::2] | (selected[..., 1::2] << 4)
        qscales[tile_start:tile_end] = scales[
            ncol.view(tile_count, 1, 32).expand(tile_count, groups, 32),
            group.view(1, groups, 1).expand(tile_count, groups, 32),
        ]

    return qcodes.view(-1).contiguous(), qscales.view(-1).contiguous()


def _turbomind_fallback(
    x: torch.Tensor,
    tm_weight: torch.Tensor,
    tm_scales: torch.Tensor,
    n: int,
    group_size: int,
    k_ld: int,
    q_ld: int,
) -> torch.Tensor:
    from vllm import _sm70_ops as sm70_ops

    out = torch.empty((x.shape[0], n), dtype=x.dtype, device=x.device)
    sm70_ops.nvfp4_gemm_sm70_out(
        out,
        x,
        tm_weight,
        tm_scales,
        group_size,
        k_ld,
        q_ld,
    )
    return out


def _skinny_nvfp4_linear_impl(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    qpn_codes: torch.Tensor,
    qpn_scales: torch.Tensor,
    tm_weight: torch.Tensor,
    tm_scales: torch.Tensor,
    global_scale: float,
    n: int,
    k: int,
    group_size: int,
    k_ld: int,
    q_ld: int,
) -> torch.Tensor:
    global _runtime_ok

    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(
            f"SM70 skinny NVFP4 supports FP16 or BF16 activations, but got {x.dtype}."
        )

    # Both the small-M kernels and the existing TurboMind fallback execute in
    # FP16 on Volta. For a BF16 model, make that conversion explicit at this
    # adapter boundary and restore the public output dtype afterwards.
    output_dtype = x.dtype
    kernel_x = x if x.dtype == torch.float16 else x.to(torch.float16)
    m = kernel_x.shape[0]

    use_qpn = (
        _runtime_ok
        and 4 <= m <= 16
        and k % 64 == 0
        and n % 32 == 0
        and qpn_codes.numel() == n * (k // 2)
        and qpn_scales.numel() == n * (k // 16)
    )
    use_simt = (
        _runtime_ok
        and 1 <= m <= 3
        and k % 128 == 0
        and n % 8 == 0
        and codes.numel() == n * (k // 2)
        and scales.numel() == n * (k // 16)
    )

    if use_qpn:
        route = "qpn"
    elif use_simt:
        route = "simt"
    else:
        route = "turbomind"
    route_key = (route, m, output_dtype)
    if route_key not in _route_log_seen:
        _route_log_seen.add(route_key)
        logger.info(
            "SM70 skinny NVFP4 route: M=%d N=%d K=%d dtype=%s -> %s",
            m,
            n,
            k,
            output_dtype,
            route,
        )

    from vllm import _sm70_ops as sm70_ops

    if use_qpn:
        out = sm70_ops.skinny_nvfp4_gemm_qpn(
            kernel_x, qpn_codes, qpn_scales, global_scale, n
        )
    elif use_simt:
        out = sm70_ops.skinny_nvfp4_gemm_simt(kernel_x, codes, scales, global_scale)
    else:
        out = _turbomind_fallback(
            kernel_x, tm_weight, tm_scales, n, group_size, k_ld, q_ld
        )

    return out if output_dtype == torch.float16 else out.to(output_dtype)


def _skinny_nvfp4_linear_fake(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    qpn_codes: torch.Tensor,
    qpn_scales: torch.Tensor,
    tm_weight: torch.Tensor,
    tm_scales: torch.Tensor,
    global_scale: float,
    n: int,
    k: int,
    group_size: int,
    k_ld: int,
    q_ld: int,
) -> torch.Tensor:
    del (
        codes,
        scales,
        qpn_codes,
        qpn_scales,
        tm_weight,
        tm_scales,
        global_scale,
        k,
        group_size,
        k_ld,
        q_ld,
    )
    return x.new_empty((x.shape[0], n))


direct_register_custom_op(
    op_name="sm70_skinny_nvfp4_linear",
    op_func=_skinny_nvfp4_linear_impl,
    fake_impl=_skinny_nvfp4_linear_fake,
)


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = expected.float().abs().max().clamp(min=1e-6)
    error = (actual.float() - expected.float()).abs().max() / denominator
    return float(error.item())


class SkinnyNvFp4LinearKernel(NvFp4LinearKernel):
    """SM70 small-M NVFP4 overlay with a TurboMind fallback."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if compute_capability is None:
            if (
                not current_platform.is_cuda()
                or not current_platform.is_device_capability((7, 0))
            ):
                return False, "requires exact CUDA capability 7.0"
        elif compute_capability != 70:
            return False, "requires exact CUDA capability 7.0"

        required_ops = (
            "skinny_nvfp4_gemm_simt",
            "skinny_nvfp4_gemm_qpn",
            "nvfp4_sm70_prepare",
            "nvfp4_gemm_sm70_out",
        )
        missing = [name for name in required_ops if not hasattr(torch.ops._C, name)]
        if missing:
            return False, "missing SM70 extension ops: " + ", ".join(missing)
        return True, None

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        del config
        return True, None

    def _validate_shape(self, layer: torch.nn.Module) -> None:
        global _runtime_ok

        n = layer.output_size_per_partition
        k = layer.input_size_per_partition
        shape = (n, k)
        if not _runtime_ok or shape in _validated_shapes:
            return

        state = sm70_tm.get_prepared_linear_state(layer)
        cases = [(1, "simt")]
        if layer.skinny_qpn_codes.numel() > 0:
            cases.append((8, "qpn"))

        try:
            for m, route in cases:
                values = torch.arange(
                    m * k, device=layer.skinny_codes.device, dtype=torch.int32
                )
                x = ((values.remainder(31) - 15).to(torch.float16) * 1e-3).view(m, k)
                reference = _turbomind_fallback(
                    x,
                    state.weight,
                    state.scales,
                    n,
                    state.group_size,
                    state.k_ld,
                    state.q_ld,
                )
                if route == "simt":
                    actual = torch.ops._C.skinny_nvfp4_gemm_simt(
                        x,
                        layer.skinny_codes,
                        layer.skinny_scales,
                        layer.skinny_global_scale,
                    )
                else:
                    actual = torch.ops._C.skinny_nvfp4_gemm_qpn(
                        x,
                        layer.skinny_qpn_codes,
                        layer.skinny_qpn_scales,
                        layer.skinny_global_scale,
                        n,
                    )
                relative_error = _relative_error(actual, reference)
                if (
                    not math.isfinite(relative_error)
                    or relative_error > _SELF_CHECK_TOL
                ):
                    raise RuntimeError(
                        f"{route} relative error {relative_error:.3e} exceeds "
                        f"{_SELF_CHECK_TOL:.3e}"
                    )
        except Exception:
            _runtime_ok = False
            logger.exception(
                "SM70 skinny NVFP4 self-check failed for N=%d K=%d; "
                "disabling skinny routes and retaining TurboMind.",
                n,
                k,
            )
            return

        _validated_shapes.add(shape)
        logger.info(
            "SM70 skinny NVFP4 self-check passed for N=%d K=%d (%d route(s)).",
            n,
            k,
            len(cases),
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if layer.weight.dtype != torch.uint8 or layer.weight.dim() != 2:
            raise TypeError("Skinny NVFP4 expects native uint8 [N, K/2] weights.")

        n = layer.output_size_per_partition
        k = layer.input_size_per_partition
        if layer.weight.shape != (n, k // 2):
            raise ValueError(
                f"Skinny NVFP4 expected weight shape {(n, k // 2)}, "
                f"got {tuple(layer.weight.shape)}."
            )

        # Keep aliases of the checkpoint-native tensors; no clone is needed.
        # TurboMind prepare only reads them before replacing layer.weight.
        layer.register_parameter(
            "skinny_codes", Parameter(layer.weight.data, requires_grad=False)
        )
        layer.register_parameter(
            "skinny_scales",
            Parameter(
                layer.weight_scale.data.view(torch.uint8).contiguous(),
                requires_grad=False,
            ),
        )
        layer.skinny_global_scale = float(
            layer.weight_global_scale.data.float().max().item()
        )

        qcodes, qscales = qpn_prepack(layer.skinny_codes.data, layer.skinny_scales.data)
        empty = layer.skinny_codes.data.new_empty(0)
        layer.register_parameter(
            "skinny_qpn_codes",
            Parameter(qcodes if qcodes is not None else empty, requires_grad=False),
        )
        layer.register_parameter(
            "skinny_qpn_scales",
            Parameter(qscales if qscales is not None else empty, requires_grad=False),
        )

        sm70_tm.prepare_nvfp4_linear(layer)
        weight_device = layer.weight.device
        scale_device = layer.weight_scale.device
        scale_dtype = layer.weight_scale.dtype
        layer.weight = Parameter(
            torch.empty(0, dtype=torch.uint8, device=weight_device),
            requires_grad=False,
        )
        layer.weight_scale = Parameter(
            torch.empty(0, dtype=scale_dtype, device=scale_device),
            requires_grad=False,
        )

        self._validate_shape(layer)
        logger.info_once(
            "SM70 skinny NVFP4 dense backend enabled: SIMT M<=3, "
            "QPN M=4..16, TurboMind fallback."
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        state = sm70_tm.get_prepared_linear_state(layer)
        k = layer.input_size_per_partition
        n = layer.output_size_per_partition
        reshaped_x = x.reshape(-1, k).contiguous()
        out = torch.ops.vllm.sm70_skinny_nvfp4_linear(
            reshaped_x,
            layer.skinny_codes,
            layer.skinny_scales,
            layer.skinny_qpn_codes,
            layer.skinny_qpn_scales,
            state.weight,
            state.scales,
            layer.skinny_global_scale,
            n,
            k,
            state.group_size,
            state.k_ld,
            state.q_ld,
        )
        if bias is not None:
            out = out + bias
        return out.reshape(x.shape[:-1] + (n,))
