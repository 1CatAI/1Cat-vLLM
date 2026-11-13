# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Weight-only block-scaled FP8 through the native QPN8 operators.

The kernel admits static E4M3 weights with 128 x 128 scales, aligned positive
N/K, and FP16 activations/output on Volta and Turing. KernelConfig owns the
policy; no model identity or parallel layout limits this admission.

M=0 returns an empty output. M=1..8 uses packed GEMM. Volta retains its
TurboMind layout for larger M, avoiding a full dense reconstruction on short
prefills, when decode can exceed M=8 or
kernel_config.sm70_fp8.block_qpn8_volta_turbomind_prefill asks for it.
Turing, and Volta without that second layout, use invocation-private
dense scratch for the existing FP16 prefill operator. Native arithmetic is
unchanged.
"""

from collections.abc import Sequence

import torch
from torch.library import custom_op

from vllm import _sm70_ops as sm70_ops
from vllm._sm70.policy import NativeBindings, call_native
from vllm.config import get_current_vllm_config_or_none
from vllm.config.kernel import Sm70Fp8Config
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.qpn.fp8 import (
    TurboMindFp8LinearKernel,
    _sm70_fp8_qpn8_config,
)
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.sm70_provider import (
    flatten_linear_input,
    restore_linear_output,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Static128BlockSym,
)
from vllm.platforms import current_platform

# Block-scale QPN8 single-tile GEMM bound; M=9..40 uses native two-phase/chunked
# QPN8 when the optional second TurboMind layout is omitted.
_QPN8_MAX_M = 8
_QPN8_BATCH_NATIVE_MAX_M = 40
_turbomind_headroom_cache: dict[tuple[int, int], bool] = {}
logger = init_logger(__name__)


def _qpn8_native_linear_write(
    out: torch.Tensor,
    x: torch.Tensor,
    codes: torch.Tensor,
    group_scales: torch.Tensor,
    split_k: int,
    accumulator_chains: int,
    prefetch_codes: bool,
    fallback_weight: torch.Tensor | None = None,
    fallback_scales: torch.Tensor | None = None,
    fallback_k_ld: int = 0,
    fallback_q_ld: int = 0,
    native_policy: list[str] | None = None,
) -> None:
    # The M decision stays inside the opaque op: a compiled graph covers a
    # dynamic M range, so a Python branch traced at small M would be reused
    # for prefill.
    k_dim, n_dim = codes.shape
    if x.shape[0] == 0:
        return
    if x.shape[0] <= _QPN8_MAX_M:
        call_native(
            sm70_ops.fp8_qpn8_gemm_sm70_out,
            native_policy,
            out,
            x,
            codes,
            group_scales,
            split_k,
            accumulator_chains,
            True,
            prefetch_codes,
        )
        return
    if fallback_weight is not None:
        call_native(
            sm70_ops.fp8_gemm_sm70_out,
            native_policy,
            out,
            x,
            fallback_weight,
            fallback_scales,
            128,
            fallback_k_ld,
            fallback_q_ld,
            False,
        )
        return
    if x.shape[0] <= _QPN8_BATCH_NATIVE_MAX_M and current_platform.is_device_capability(
        70
    ):
        if (
            split_k in (12, 16)
            and accumulator_chains == 2
            and not prefetch_codes
            and k_dim % (16 * split_k) == 0
            and n_dim % 128 == 0
            and hasattr(torch.ops._C, "fp8_qpn8_block_native_sm70_out")
        ):
            torch.ops._C.fp8_qpn8_block_native_sm70_out(
                out,
                x,
                codes,
                group_scales,
                split_k,
                0,
                False,
                False,
            )
            return
        if (
            split_k in (4, 8, 12, 16, 32)
            and accumulator_chains in (1, 2)
            and (split_k != 12 or (accumulator_chains == 2 and not prefetch_codes))
            and k_dim % (16 * split_k) == 0
            and n_dim % 128 == 0
        ):
            for row in range(0, x.shape[0], _QPN8_MAX_M):
                end = min(row + _QPN8_MAX_M, x.shape[0])
                call_native(
                    sm70_ops.fp8_qpn8_gemm_sm70_out,
                    native_policy,
                    out[row:end],
                    x[row:end],
                    codes,
                    group_scales,
                    split_k,
                    accumulator_chains,
                    True,
                    prefetch_codes,
                )
            return
    dense = torch.empty((k_dim * n_dim,), dtype=torch.float16, device=x.device)
    call_native(
        sm70_ops.fp8_qpn8_prefill_sm70_out,
        native_policy,
        out,
        dense.data_ptr(),
        x,
        codes,
        group_scales,
        False,
    )


@custom_op(
    "sm70_fp8::qpn8_native_linear",
    mutates_args=(),
    schema="(Tensor x, Tensor codes, Tensor group_scales, int split_k, "
    "int accumulator_chains, bool prefetch_codes, Tensor? fallback_weight=None, "
    "Tensor? fallback_scales=None, int fallback_k_ld=0, int fallback_q_ld=0, "
    "str[]? native_policy=None) -> Tensor",
)
def _qpn8_native_linear(
    x: torch.Tensor,
    codes: torch.Tensor,
    group_scales: torch.Tensor,
    split_k: int,
    accumulator_chains: int,
    prefetch_codes: bool,
    fallback_weight: torch.Tensor | None = None,
    fallback_scales: torch.Tensor | None = None,
    fallback_k_ld: int = 0,
    fallback_q_ld: int = 0,
    native_policy: list[str] | None = None,
) -> torch.Tensor:
    out = x.new_empty((x.shape[0], codes.shape[1]))
    _qpn8_native_linear_write(
        out,
        x,
        codes,
        group_scales,
        split_k,
        accumulator_chains,
        prefetch_codes,
        fallback_weight,
        fallback_scales,
        fallback_k_ld,
        fallback_q_ld,
        native_policy,
    )
    return out


@custom_op(
    "sm70_fp8::qpn8_native_linear_out",
    mutates_args=("out",),
    schema="(Tensor(a!) out, Tensor x, Tensor codes, Tensor group_scales, "
    "int split_k, int accumulator_chains, bool prefetch_codes, "
    "Tensor? fallback_weight=None, Tensor? fallback_scales=None, "
    "int fallback_k_ld=0, int fallback_q_ld=0, "
    "str[]? native_policy=None) -> ()",
)
def _qpn8_native_linear_out(
    out: torch.Tensor,
    x: torch.Tensor,
    codes: torch.Tensor,
    group_scales: torch.Tensor,
    split_k: int,
    accumulator_chains: int,
    prefetch_codes: bool,
    fallback_weight: torch.Tensor | None = None,
    fallback_scales: torch.Tensor | None = None,
    fallback_k_ld: int = 0,
    fallback_q_ld: int = 0,
    native_policy: list[str] | None = None,
) -> None:
    _qpn8_native_linear_write(
        out,
        x,
        codes,
        group_scales,
        split_k,
        accumulator_chains,
        prefetch_codes,
        fallback_weight,
        fallback_scales,
        fallback_k_ld,
        fallback_q_ld,
        native_policy,
    )


@custom_op(
    "sm70_fp8::qpn8_native_gated_silu",
    mutates_args=(),
    schema="(Tensor x, Tensor codes, Tensor group_scales, int split_k, "
    "int accumulator_chains, bool prefetch_codes, Tensor? fallback_weight=None, "
    "Tensor? fallback_scales=None, int fallback_k_ld=0, int fallback_q_ld=0, "
    "str[]? native_policy=None) -> Tensor",
)
def _qpn8_native_gated_silu(
    x: torch.Tensor,
    codes: torch.Tensor,
    group_scales: torch.Tensor,
    split_k: int,
    accumulator_chains: int,
    prefetch_codes: bool,
    fallback_weight: torch.Tensor | None = None,
    fallback_scales: torch.Tensor | None = None,
    fallback_k_ld: int = 0,
    fallback_q_ld: int = 0,
    native_policy: list[str] | None = None,
) -> torch.Tensor:
    k_dim, n_dim = codes.shape
    hidden = n_dim // 2
    out = x.new_empty((x.shape[0], hidden))
    if x.shape[0] == 0:
        return out
    if x.shape[0] <= _QPN8_MAX_M:
        call_native(
            sm70_ops.fp8_qpn8_gated_pair_sm70_out,
            native_policy,
            out,
            x,
            codes,
            group_scales,
            split_k,
            accumulator_chains,
            True,
            prefetch_codes,
        )
        return out
    if (
        x.shape[0] <= _QPN8_BATCH_NATIVE_MAX_M
        and split_k == 8
        and k_dim % 128 == 0
        and n_dim % 256 == 0
        and current_platform.is_device_capability(70)
        and hasattr(torch.ops._C, "fp8_qpn8_block_native_sm70_out")
    ):
        torch.ops._C.fp8_qpn8_block_native_sm70_out(
            out,
            x,
            codes,
            group_scales,
            8,
            0,
            False,
            True,
        )
        return out
    if fallback_weight is not None:
        gate_up = x.new_empty((x.shape[0], n_dim))
        call_native(
            sm70_ops.fp8_gemm_sm70_out,
            native_policy,
            gate_up,
            x,
            fallback_weight,
            fallback_scales,
            128,
            fallback_k_ld,
            fallback_q_ld,
            False,
        )
        torch.ops._C.silu_and_mul(out, gate_up)
        return out
    dense = torch.empty((k_dim * n_dim,), dtype=torch.float16, device=x.device)
    call_native(
        sm70_ops.fp8_qpn8_prefill_sm70_out,
        native_policy,
        out,
        dense.data_ptr(),
        x,
        codes,
        group_scales,
        True,
    )
    return out


@_qpn8_native_gated_silu.register_fake
def _qpn8_native_gated_silu_fake(
    x,
    codes,
    group_scales,
    split_k,
    accumulator_chains,
    prefetch_codes,
    fallback_weight=None,
    fallback_scales=None,
    fallback_k_ld=0,
    fallback_q_ld=0,
    native_policy=None,
):
    return x.new_empty((x.shape[0], codes.shape[1] // 2))


@_qpn8_native_linear.register_fake
def _qpn8_native_linear_fake(
    x,
    codes,
    group_scales,
    split_k,
    accumulator_chains,
    prefetch_codes,
    fallback_weight=None,
    fallback_scales=None,
    fallback_k_ld=0,
    fallback_q_ld=0,
    native_policy=None,
):
    return x.new_empty((x.shape[0], codes.shape[1]))


@_qpn8_native_linear_out.register_fake
def _qpn8_native_linear_out_fake(
    out,
    x,
    codes,
    group_scales,
    split_k,
    accumulator_chains,
    prefetch_codes,
    fallback_weight=None,
    fallback_scales=None,
    fallback_k_ld=0,
    fallback_q_ld=0,
    native_policy=None,
):
    return None


class QPN8Fp8BlockScaledMMLinearKernel(TurboMindFp8LinearKernel):
    """Block-scaled FP8 on SM70/SM75 via 1Cat's native QPN8 operators."""

    # fp16 activations go straight into the GEMM; no input quantization.
    apply_input_quant = False

    @classmethod
    def is_supported(cls, compute_capability=None):
        if not current_platform.is_cuda():
            return False, "requires CUDA"
        if compute_capability is None:
            capability = current_platform.get_device_capability()
            compute_capability = capability.to_int() if capability else None
        if compute_capability not in (70, 72, 75):
            return False, "requires a Volta or Turing CUDA device"
        return True, None

    @staticmethod
    def _policy(config):
        if hasattr(config, "policy"):
            return config.policy
        engine = get_current_vllm_config_or_none()
        return engine.kernel_config.sm70_fp8 if engine else Sm70Fp8Config()

    @classmethod
    def can_implement(cls, config):
        policy = cls._policy(config)
        policy.resolve()
        if not policy.block_qpn8:
            return False, "disabled by kernel_config.sm70_fp8.block_qpn8"
        if set(policy.explicit_enables).intersection(
            {"gated_silu", "prefill_prescaled", "prescaled_decode"}
        ):
            return False, "retains the explicitly configured fused or prescaled variant"
        if policy.force_marlin:
            return False, "Marlin selected by legacy backend compatibility policy"
        if getattr(config, "is_bmm", False):
            return False, "grouped BMM retains its existing grouped implementation"
        if config.input_dtype != torch.float16 or config.out_dtype != torch.float16:
            return False, "requires FP16 activations and output"
        if config.weight_quant_key != kFp8Static128BlockSym:
            return False, "requires static E4M3 weights with 128 x 128 block scales"
        if len(config.weight_shape) != 2:
            return False, "requires a rank-two weight matrix"
        n, k = config.weight_shape
        if n <= 0 or k <= 0 or n % 128 or k % 128:
            return False, f"requires positive N,K divisible by 128, got ({n},{k})"
        missing = [
            name
            for name in (
                "fp8_qpn8_prepare_sm70",
                "fp8_qpn8_gemm_sm70_out",
                "fp8_qpn8_prefill_sm70_out",
            )
            if not hasattr(torch.ops._C, name)
        ]
        if missing:
            return False, f"missing native operators: {missing}"
        return True, None

    def __init__(self, config, layer_param_names: Sequence[str] = ()):
        ScaledMMLinearKernel.__init__(self, config, layer_param_names)
        self.policy = self._policy(config)
        self.policy.resolve()
        self.policy.native.resolve("fp8")
        self.native_ops = NativeBindings(self.policy.native.values)

    def _keep_volta_turbomind_copy(self, device: torch.device | None = None) -> bool:
        explicit = self.policy.block_qpn8_volta_turbomind_prefill
        if explicit is not None:
            return explicit
        engine = get_current_vllm_config_or_none()
        if engine is None:
            return True
        spec = engine.speculative_config
        verify_rows = 1 + (spec.num_speculative_tokens if spec is not None else 0)
        if engine.scheduler_config.max_num_seqs * verify_rows <= _QPN8_MAX_M:
            return False
        if device is not None and device.type == "cuda":
            dev_idx = (
                device.index
                if device.index is not None
                else torch.accelerator.current_device_index()
            )
            key = (id(engine), dev_idx)
            fits = _turbomind_headroom_cache.get(key)
            if fits is None:
                free_bytes, _ = torch.cuda.mem_get_info(dev_idx)
                # Raw checkpoint weights are already resident on GPU when the
                # first layer converts; require >=18 GiB free before duplicating
                # an entire large block-FP8 checkpoint on 32 GiB Volta shards.
                fits = free_bytes >= 18 * 1024**3
                _turbomind_headroom_cache[key] = fits
                if not fits:
                    logger.info_once(
                        "Skipping optional Volta TurboMind block-FP8 prefill "
                        "copy due to memory headroom (%.2f GiB free); M=9..40 "
                        "uses native block QPN8 and larger M uses prefill "
                        "scratch.",
                        free_bytes / 1024**3,
                    )
            return fits
        return True

    def process_weights_after_loading(self, layer: torch.nn.Module):
        weight = layer.weight.data
        n, k = weight.shape
        scale = getattr(layer, "weight_scale_inv", None)
        if scale is None:
            scale = getattr(layer, "weight_scale", None)
        assert scale is not None, "block-FP8 layer without a weight scale"
        block_scales = scale.data.detach().float().contiguous()
        if tuple(block_scales.shape) != (n // 128, k // 128):
            raise ValueError(
                f"QPN8: scale raster {tuple(block_scales.shape)} does not match "
                f"weights {n}x{k} at block [128,128]"
            )
        if weight.dtype != torch.float8_e4m3fn:
            weight = weight.view(torch.float8_e4m3fn)

        codes, group_scales = self.native_ops.fp8_qpn8_prepare_sm70(
            weight, block_scales
        )
        k_dim, n_dim = (int(dim) for dim in codes.shape)
        layer.register_buffer(
            "_sm70_block_fp8_qpn8_packed_codes", codes, persistent=False
        )
        layer.register_buffer(
            "_sm70_block_fp8_qpn8_packed_scales", group_scales, persistent=False
        )
        capability = current_platform.get_device_capability()
        if (
            capability is not None
            and capability.to_int() in (70, 72)
            and self._keep_volta_turbomind_copy(codes.device)
        ):
            packed, scales, metadata = self.native_ops.fp8_sm70_prepare(
                weight, block_scales, 128, False
            )
            layer.register_buffer(
                "_sm70_block_fp8_turbomind_packed_weight", packed, persistent=False
            )
            layer.register_buffer(
                "_sm70_block_fp8_turbomind_packed_scales", scales, persistent=False
            )
            layer._qpn8_fallback_k_ld = int(metadata[0].item())
            layer._qpn8_fallback_q_ld = int(metadata[1].item())
        if hasattr(layer, "_qpn8_fallback_k_ld"):
            logger.info_once(
                "Block FP8 QPN8 native GEMM supports M=1..8; larger M uses a "
                "second, TurboMind-packed copy of each weight. Set "
                "kernel_config.sm70_fp8.block_qpn8_volta_turbomind_prefill=false "
                "to hold one layout and use the FP16 prefill instead."
            )
        else:
            logger.info_once(
                "Block FP8 QPN8 native GEMM supports M=1..40; larger M uses the "
                "FP16 prefill, which needs no second copy of the weights."
            )
        layer._qpn8_out_features = n
        layer._qpn8_cfg = _sm70_fp8_qpn8_config(k_dim, n_dim, False)
        if (
            self._is_sm70_gated_silu_layer(layer)
            and bool(self.policy.gated_silu)
            and n_dim % 256 == 0
            and k_dim % 128 == 0
            and hasattr(torch.ops._C, "fp8_qpn8_gated_pair_sm70_out")
        ):
            layer._qpn8_gated_cfg = _sm70_fp8_qpn8_config(k_dim, n_dim, True)
            layer.sm70_fp8_gated_silu = True
        layer.sm70_fp8_turbomind = True
        # The startup memory profile accounts for both packed layouts on
        # Volta. Raw checkpoint weights are no longer resident.
        layer.weight = torch.nn.Parameter(
            torch.empty(0, dtype=torch.uint8, device=codes.device), requires_grad=False
        )

    def apply_block_scaled_mm(self, A, B, As, Bs):
        # Satisfies the ABC; apply_weights below bypasses the base-class
        # A/As machinery entirely (weight-only path, fp16 activations).
        raise RuntimeError("unreachable: QPN8 overrides apply_weights")

    def supports_direct_output(self, layer: torch.nn.Module) -> bool:
        return hasattr(layer, "_sm70_block_fp8_qpn8_packed_codes")

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        output: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        codes = layer._sm70_block_fp8_qpn8_packed_codes
        split_k, accumulator_chains, prefetch_codes = layer._qpn8_cfg
        x_2d = flatten_linear_input(x, codes.shape[0]).contiguous()
        if output is not None:
            expected_shape = (*x.shape[:-1], layer._qpn8_out_features)
            if (
                output.shape != expected_shape
                or output.dtype != x.dtype
                or output.device != x.device
                or not output.is_contiguous()
            ):
                raise ValueError(
                    "Incompatible direct output tensor for SM70 block-FP8 QPN8."
                )
            out_2d = output.reshape(x_2d.shape[0], layer._qpn8_out_features)
            self.native_ops.invoke(
                torch.ops.sm70_fp8.qpn8_native_linear_out,
                out_2d,
                x_2d,
                codes,
                layer._sm70_block_fp8_qpn8_packed_scales,
                split_k,
                accumulator_chains,
                prefetch_codes,
                getattr(layer, "_sm70_block_fp8_turbomind_packed_weight", None),
                getattr(layer, "_sm70_block_fp8_turbomind_packed_scales", None),
                getattr(layer, "_qpn8_fallback_k_ld", 0),
                getattr(layer, "_qpn8_fallback_q_ld", 0),
                self.native_ops.arguments,
            )
            if bias is not None:
                out_2d.add_(bias)
            return output
        y = self.native_ops.invoke(
            torch.ops.sm70_fp8.qpn8_native_linear,
            x_2d,
            codes,
            layer._sm70_block_fp8_qpn8_packed_scales,
            split_k,
            accumulator_chains,
            prefetch_codes,
            getattr(layer, "_sm70_block_fp8_turbomind_packed_weight", None),
            getattr(layer, "_sm70_block_fp8_turbomind_packed_scales", None),
            getattr(layer, "_qpn8_fallback_k_ld", 0),
            getattr(layer, "_qpn8_fallback_q_ld", 0),
            self.native_ops.arguments,
        )
        if bias is not None:
            y = y + bias
        return restore_linear_output(y, x, layer._qpn8_out_features)

    def apply_fused_silu_and_mul(
        self, layer: torch.nn.Module, x: torch.Tensor
    ) -> torch.Tensor | None:
        if not getattr(layer, "sm70_fp8_gated_silu", False):
            return None
        codes = layer._sm70_block_fp8_qpn8_packed_codes
        split_k, accumulator_chains, prefetch_codes = layer._qpn8_gated_cfg
        y = self.native_ops.invoke(
            torch.ops.sm70_fp8.qpn8_native_gated_silu,
            flatten_linear_input(x, codes.shape[0]).contiguous(),
            codes,
            layer._sm70_block_fp8_qpn8_packed_scales,
            split_k,
            accumulator_chains,
            prefetch_codes,
            getattr(layer, "_sm70_block_fp8_turbomind_packed_weight", None),
            getattr(layer, "_sm70_block_fp8_turbomind_packed_scales", None),
            getattr(layer, "_qpn8_fallback_k_ld", 0),
            getattr(layer, "_qpn8_fallback_q_ld", 0),
            self.native_ops.arguments,
        )
        return restore_linear_output(y, x, layer._qpn8_out_features // 2)
