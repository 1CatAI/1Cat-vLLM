# SPDX-License-Identifier: Apache-2.0
# SM70 (V100) WNA16 kernel for compressed-tensors and legacy-AWQ formats.
#
# Reads pack-quantized weights at load time, transcodes to legacy AWQ pack
# layout (interleave order [0,2,4,6,1,3,5,7] along the output dim), then
# dispatches through the existing 1Cat TurboMind s884h kernels via
# awq_sm70_prepare / awq_gemm_sm70.
#
# Why: on SM70 (V100) Cutlass/Machete (CC>=90), AllSpark/Conch (>=80),
# Marlin (>=75) all reject. Exllama runs at CC>=60 but only accepts
# uint4b8 / uint8b128 scalar types. cyankiwi's Qwen3.6-27B quants are
# compressed-tensors uint4 (asymmetric) — nothing currently picks them.
# This kernel fills that gap.

from __future__ import annotations

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

logger = init_logger(__name__)

# AWQ unpack uses GATHER with awq_order [0,4,1,5,2,6,3,7]; the inverse
# permutation [0,2,4,6,1,3,5,7] is what packing must use for round-trip
# correctness. Same constant as the MoE SM70 transcoder uses.
_AWQ_PACK_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)


def _awq_pack_last_dim(unpacked: torch.Tensor) -> torch.Tensor:
    """[..., X, Y] uint8 -> [..., X, Y/8] int32 with AWQ interleave."""
    *prefix, x, y = unpacked.shape
    assert y % 8 == 0
    grouped = unpacked.view(*prefix, x, y // 8, 8)
    res = grouped[..., _AWQ_PACK_ORDER[7]].to(torch.int32)
    for i in range(6, -1, -1):
        res = (res << 4) | grouped[..., _AWQ_PACK_ORDER[i]].to(torch.int32)
    return res


def _ct_qweight_to_awq(ct_q: torch.Tensor) -> torch.Tensor:
    """CT [N, K/8] (sequential pack along K) -> AWQ [K, N/8] (interleave pack along N)."""
    n, k_div_8 = ct_q.shape
    k = k_div_8 * 8
    unpacked = torch.empty(n, k, dtype=torch.uint8, device=ct_q.device)
    tmp = ct_q.clone()
    for i in range(8):
        unpacked[:, i::8] = (tmp & 0xF).to(torch.uint8)
        tmp = tmp >> 4
    return _awq_pack_last_dim(unpacked.t().contiguous())


def _ct_qzeros_to_awq(ct_zp: torch.Tensor) -> torch.Tensor:
    """CT [N/8, K/gs] (pack along N) -> AWQ [K/gs, N/8] (interleave pack along N)."""
    n_div_8, k_gs = ct_zp.shape
    n = n_div_8 * 8
    unpacked = torch.empty(n, k_gs, dtype=torch.uint8, device=ct_zp.device)
    tmp = ct_zp.clone()
    for i in range(8):
        unpacked[i::8, :] = (tmp & 0xF).to(torch.uint8)
        tmp = tmp >> 4
    return _awq_pack_last_dim(unpacked.t().contiguous())


class SM70TurboMindLinearKernel(MPLinearKernel):
    """V100 dense WNA16 kernel: CT/legacy pack-quant -> TurboMind s884h GEMM."""

    SUPPORTED_QUANT_TYPES = [scalar_types.uint4, scalar_types.uint4b8]
    SUPPORTED_GROUP_SIZES = (32, 64, 128)

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_cuda_alike():
            return False, 'SM70TurboMind requires CUDA'
        if c.act_type != torch.float16:
            return False, 'SM70TurboMind requires float16 activations'
        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (False,
                    f'SM70TurboMind: weight type {c.weight_type} not supported'
                    f' (need {cls.SUPPORTED_QUANT_TYPES})')
        if c.group_size not in cls.SUPPORTED_GROUP_SIZES:
            return (False,
                    f'SM70TurboMind: group_size={c.group_size} not in '
                    f'{cls.SUPPORTED_GROUP_SIZES}')
        k_part, n_part = c.partition_weight_shape
        if k_part % 8 != 0 or n_part % 8 != 0:
            return False, 'SM70TurboMind: K and N must be multiples of 8'
        if k_part % c.group_size != 0:
            return (False,
                    f'SM70TurboMind: K={k_part} not divisible by '
                    f'group_size={c.group_size}')
        if c.has_g_idx:
            return False, 'SM70TurboMind: act-reorder (g_idx) not supported'
        if not hasattr(torch.ops._C, 'awq_sm70_prepare'):
            return False, 'SM70TurboMind: awq_sm70_prepare op missing'
        # Only run on actual SM70 hardware - SM75+ should pick a faster kernel
        cap = current_platform.get_device_capability()
        if cap is None or not (cap[0] == 7 and cap[1] == 0):
            return False, 'SM70TurboMind: only used on V100 (CC 7.0)'
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        c = self.config
        k_part, n_part = c.partition_weight_shape
        gs = c.group_size

        ct_q = getattr(layer, self.w_q_name).data
        ct_s = getattr(layer, self.w_s_name).data
        ct_zp = (getattr(layer, self.w_zp_name).data
                 if c.zero_points and self.w_zp_name else None)
        device = ct_q.device

        assert ct_q.shape == (n_part, k_part // 8), (
            f'SM70TurboMind: expected CT qweight [{n_part}, {k_part//8}], '
            f'got {tuple(ct_q.shape)}')
        assert ct_s.shape[0] == n_part, (
            f'SM70TurboMind: expected scale dim 0 == {n_part}, '
            f'got {tuple(ct_s.shape)}')
        k_gs = ct_s.shape[1]

        logger.info(
            'SM70TurboMind: layer %s K=%d N=%d gs=%d K/gs=%d asym=%s',
            getattr(layer, '_layer_name', '?'), k_part, n_part, gs, k_gs,
            c.zero_points)

        awq_q = _ct_qweight_to_awq(ct_q)
        awq_s = ct_s.t().contiguous().to(torch.float16)

        if c.zero_points and ct_zp is not None:
            assert ct_zp.shape == (n_part // 8, k_gs), (
                f'SM70TurboMind: expected CT qzeros [{n_part//8}, {k_gs}], '
                f'got {tuple(ct_zp.shape)}')
            awq_zp = _ct_qzeros_to_awq(ct_zp)
        else:
            zp_val = torch.tensor(
                [0x88888888], dtype=torch.uint32).view(torch.int32).item()
            awq_zp = torch.full((k_gs, n_part // 8), zp_val,
                                dtype=torch.int32, device=device)

        tm_w, tm_s, meta = ops.awq_sm70_prepare(awq_q, awq_s, awq_zp, gs)

        layer._awq_sm70_weight = torch.nn.Parameter(tm_w, requires_grad=False)
        layer._awq_sm70_scales = torch.nn.Parameter(tm_s, requires_grad=False)
        meta0 = meta[0].item() if torch.is_tensor(meta[0]) else meta[0]
        meta1 = meta[1].item() if torch.is_tensor(meta[1]) else meta[1]
        layer._awq_sm70_k_ld = int(meta0)
        layer._awq_sm70_q_ld = int(meta1)
        layer._awq_sm70_group_size = gs
        layer._awq_sm70_prepared = True

        empty_i32 = torch.empty(0, dtype=torch.int32, device=device)
        empty_fp16 = torch.empty(0, dtype=torch.float16, device=device)
        replace_parameter(
            layer, self.w_q_name,
            torch.nn.Parameter(empty_i32, requires_grad=False))
        replace_parameter(
            layer, self.w_s_name,
            torch.nn.Parameter(empty_fp16, requires_grad=False))
        if c.zero_points and self.w_zp_name:
            replace_parameter(
                layer, self.w_zp_name,
                torch.nn.Parameter(empty_i32, requires_grad=False))

        del awq_q, awq_s, awq_zp

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out_shape = x.shape[:-1] + (layer._awq_sm70_weight.shape[-1] * 8,)
        x_2d = x.reshape(-1, x.shape[-1])
        out = ops.awq_gemm_sm70(
            x_2d,
            layer._awq_sm70_weight,
            layer._awq_sm70_scales,
            layer._awq_sm70_group_size,
            layer._awq_sm70_k_ld,
            layer._awq_sm70_q_ld,
        )
        if bias is not None:
            out.add_(bias)
        return out.reshape(out_shape)
