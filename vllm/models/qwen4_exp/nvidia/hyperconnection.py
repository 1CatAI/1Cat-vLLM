# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HyperConnection (Gated Residual) utilities — NVIDIA model variant.

Implements the HyperConnection residual scheme proposed in
"HyperConnections" (https://arxiv.org/abs/2409.19606). This NVIDIA variant
delays each HC combine to the following HC mix boundary. HC glue kernels,
including fused combine+RMSNorm, live in ``ops/hc.py``; projections remain
standard vLLM Linear modules.

Hidden states between layers have shape ``[..., HC*HS]`` with HS inner
(HC outer, HS inner — checkpoint-native layout).

Typical usage inside a transformer decoder layer::

    self.attn_hc = GatedResidual(hc_config)

    hidden_states, block_input, injection = self.attn_hc.mix(hidden_states)
    attention_output = attention(block_input)
    hidden_states, block_input, injection = self.mlp_hc.combine_and_mix(
        hidden_states, attention_output, injection
    )
"""

import torch
from torch import nn

from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.layers.quantization.sm70_online_qpn8 import (
    maybe_apply_fused_hc,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.utils.torch_utils import direct_register_custom_op

from ..common.hyperconnection import (
    GroupedGemmaRMSNorm,
    HyperConnectionConfig,
)
from .ops.hc import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
    hc_gate_mix,
    hc_silu,
)
from .sm70_fp16_hc import maybe_apply_qwen38_sm70_fp16_fused_hc


# ---------------------------------------------------------------------------
# Gated-residual variant
# ---------------------------------------------------------------------------
class GatedResidual(nn.Module):
    """Gated HyperConnection with learnable low-rank mixing and injection.

    ``combine_and_mix()`` runs the pre pipeline (grouped GemmaRMSNorm -> merged
    low-rank down+inject GEMM -> silu -> up GEMM -> sigmoid -> gated mean
    over the HC streams). When passed a pending block output and an injection,
    it fuses their residual combine with the RMSNorm. Final mixers use
    ``use_combine=False`` and do not produce a new injection.

    Weights: the norm owns the grouped GemmaRMSNorm affine; the projections
    are vLLM Linear modules (merged replicated linear for down+inject), so
    GEMM dispatch (e.g. the low-latency skinny GEMM) applies through the
    standard quant_method mechanism.
    """

    def __init__(
        self,
        config: HyperConnectionConfig,
        use_combine: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.lora_rank = config.hc_lowrank
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.use_combine = use_combine

        norm_size = (
            self.hyper_hidden_size if config.hc_per_branch_norm else config.hidden_size
        )
        group_size = config.hidden_size if config.hc_per_branch_norm else None
        # Normalize each H-sized HC stream independently while retaining a
        # separate affine weight for every element of the HC*H layout.
        self.hc_norm = GroupedGemmaRMSNorm(
            norm_size,
            eps=config.rms_norm_eps,
            group_size=group_size,
            dtype=config.params_dtype,
        )

        # -- vLLM Linear weights --------------------------------------------
        # The merged skinny-GEMM shape is physically padded to 16 rows to ensure
        # good alignment and performant implementation chosen by CuBLAS heuristics.
        self.pad_size = (-(self.lora_rank + self.hc_count)) % 16 if use_combine else 0
        if use_combine:
            self.input_mix_weight_down_block_inject = MergedColumnParallelLinear(
                self.hyper_hidden_size,
                [self.lora_rank, self.hc_count]
                + ([self.pad_size] if self.pad_size else []),
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_down_block_inject"),
                return_bias=False,
                disable_tp=True,
            )
        else:
            self.input_mix_weight_down = ReplicatedLinear(
                self.hyper_hidden_size,
                self.lora_rank,
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_down"),
                return_bias=False,
            )
        self.input_mix_weight_up = ReplicatedLinear(
            self.lora_rank,
            self.hyper_hidden_size,
            bias=False,
            params_dtype=config.params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "input_mix_weight_up"),
            return_bias=False,
        )

    def _project(self, xn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.use_combine:
            fused_fp16 = maybe_apply_qwen38_sm70_fp16_fused_hc(
                self.input_mix_weight_down_block_inject,
                self.input_mix_weight_up,
                xn,
                getattr(self, "_sm70_qwen38_fp16_fused_hc", False),
            )
            if fused_fp16 is not None:
                return fused_fp16

            fused = maybe_apply_fused_hc(
                self.input_mix_weight_down_block_inject,
                self.input_mix_weight_up,
                xn,
            )
            if fused is not None:
                return fused

            split_sizes = [self.lora_rank, self.hc_count, self.pad_size]
            down_and_injection = self.input_mix_weight_down_block_inject(xn)
            lora, injection, _ = down_and_injection.split(split_sizes, dim=-1)
        else:
            lora = self.input_mix_weight_down(xn)
            injection = None

        lora = hc_silu(lora, self.hc_count)
        gate = self.input_mix_weight_up(lora)
        return hc_gate_mix(xn, gate, self.hc_count), injection

    def mix(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        xn = grouped_gemma_rmsnorm(
            hidden_states,
            self.hc_norm.weight,
            self.config.rms_norm_eps,
            self.hc_count,
        )

        block_input, injection = self._project(xn)

        return hidden_states, block_input, injection

    def enable_partial_inputs(self, name: str, runtime=None) -> None:
        """Block outputs reaching this module are TP partials from now on."""
        self._partial_inputs = True
        self._hcx_name = name
        _PARTIAL_MODULES[name] = self
        if runtime is not None and self.use_combine:
            from .sm70_hcx import pack_down, pack_up

            down = self.input_mix_weight_down_block_inject.weight
            up = self.input_mix_weight_up.weight
            if down.dtype == torch.float16 and tuple(down.shape) == (336, 10240):
                self._hcx = runtime
                self._hcx_down = pack_down(down.data, runtime.logical_rank)
                self._hcx_up = pack_up(up.data, runtime.logical_rank)

    def _partial_pair(self, block_output):
        if getattr(self, "_hcx_moe_payload", False):
            rows = block_output.shape[0] // 2
            return block_output[:rows], block_output[rows:]
        return block_output, None

    def _reduce_partial(self, block_output: torch.Tensor) -> torch.Tensor:
        if not getattr(self, "_partial_inputs", False):
            return block_output
        first, second = self._partial_pair(block_output)
        if first.shape[0] > 8:
            return first
        if second is not None:
            from vllm.distributed import tensor_model_parallel_all_reduce_sum2

            return tensor_model_parallel_all_reduce_sum2(first, second)
        from vllm.distributed import tensor_model_parallel_all_reduce

        return tensor_model_parallel_all_reduce(first)

    def combine_and_mix(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor,
        prev_injection: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Consume a pending combine, then prepare the next block input.

        ``hidden_states`` is the multi-stream state from before the pending
        block's mix. Its combine with ``block_output`` is fused with this
        module's input RMSNorm.
        """
        if getattr(self, "_partial_inputs", False):
            if not self.use_combine:
                hidden, block = torch.ops.vllm.qwen38_sm70_hcx_final_mix(
                    hidden_states, prev_block_output, prev_injection, self._hcx_name
                )
                return hidden, block, None
            return torch.ops.vllm.qwen38_sm70_hcx_combine_and_mix(
                hidden_states, prev_block_output, prev_injection, self._hcx_name
            )
        prev_block_output = self._reduce_partial(prev_block_output)
        return self._combine_and_mix_reduced(
            hidden_states, prev_block_output, prev_injection
        )

    def _combine_and_mix_reduced(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor,
        prev_injection: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # HCX already supplies an opaque runtime boundary for these modules.
        # Keep full outputs but avoid full-batch norm and gate intermediates.
        from .sm70_fp16_hc import _hc_prefill_chunk_size

        chunk_size = _hc_prefill_chunk_size(hidden_states.shape[0])
        if (
            chunk_size
            and self.use_combine
            and getattr(self, "_partial_inputs", False)
            and hidden_states.is_cuda
            and hidden_states.dtype == torch.float16
        ):
            hidden = torch.empty_like(hidden_states)
            block = hidden_states.new_empty((hidden_states.shape[0], self.hidden_size))
            injection = hidden_states.new_empty((hidden_states.shape[0], self.hc_count))
            for start in range(0, hidden_states.shape[0], chunk_size):
                stop = min(start + chunk_size, hidden_states.shape[0])
                part, xn = hc_combine_norm(
                    hidden_states[start:stop],
                    prev_block_output[start:stop],
                    prev_injection[start:stop],
                    self.hc_norm.weight,
                    self.config.rms_norm_eps,
                    self.hc_count,
                )
                block_part, inject_part = self._project(xn)
                hidden[start:stop].copy_(part)
                block[start:stop].copy_(block_part)
                injection[start:stop].copy_(inject_part)
                del part, xn, block_part, inject_part
            return hidden, block, injection
        hidden_states, xn = hc_combine_norm(
            hidden_states,
            prev_block_output,
            prev_injection,
            self.hc_norm.weight,
            self.config.rms_norm_eps,
            self.hc_count,
        )

        block_input, injection = self._project(xn)

        return hidden_states, block_input, injection

    def combine(
        self,
        hidden_states: torch.Tensor,
        block_output: torch.Tensor,
        injection: torch.Tensor,
    ) -> torch.Tensor:
        if getattr(self, "_partial_inputs", False):
            return torch.ops.vllm.qwen38_sm70_hcx_combine(
                hidden_states, block_output, injection, self._hcx_name
            )
        block_output = self._reduce_partial(block_output)
        return hc_combine(hidden_states, block_output, injection, self.hc_count)

    @property
    def hyper_hidden_size(self) -> int:
        return self.hc_count * self.hidden_size


_PARTIAL_MODULES: dict[str, "GatedResidual"] = {}


def _hcx_combine_and_mix(
    hidden_states: torch.Tensor,
    block_output: torch.Tensor,
    injection: torch.Tensor,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    module = _PARTIAL_MODULES[name]
    runtime = getattr(module, "_hcx", None)
    oproj = getattr(module, "_hcx_oproj", None)
    block_output, secondary = module._partial_pair(block_output)
    if runtime is not None and 0 < block_output.shape[0] <= 8:
        diagnostic = (
            {"snapshot_name": name} if getattr(runtime, "diagnostic", False) else {}
        )
        return runtime.run(
            block_output,
            hidden_states,
            injection,
            module.hc_norm.weight,
            module.config.rms_norm_eps,
            module._hcx_down,
            module._hcx_up,
            None if oproj is None else oproj[1:],
            secondary=secondary,
            **diagnostic,
        )
    from vllm.distributed import tensor_model_parallel_all_reduce

    if oproj is not None and block_output.shape[0] <= 8:
        # Deferred projection: the block output carries the o-proj input.
        layer, k = oproj[0], oproj[1]
        block_output, _ = layer(block_output[:, :k].contiguous())
    reduced = block_output
    if block_output.shape[0] <= 8:
        if secondary is not None:
            from vllm.distributed import tensor_model_parallel_all_reduce_sum2

            reduced = tensor_model_parallel_all_reduce_sum2(block_output, secondary)
        else:
            reduced = tensor_model_parallel_all_reduce(block_output)
    hidden, block, inj = module._combine_and_mix_reduced(
        hidden_states, reduced, injection
    )
    assert inj is not None
    # Match the fake's contiguous outputs. The original large-M injection
    # is a view of the padded 336-column projection, with row stride 336.
    return hidden.contiguous(), block.contiguous(), inj.contiguous()


def _hcx_combine_and_mix_fake(hidden_states, block_output, injection, name):
    m = hidden_states.shape[0]
    return (
        torch.empty_like(hidden_states),
        block_output.new_empty((m, block_output.shape[1])),
        injection.new_empty((m, injection.shape[1])),
    )


def _hcx_combine(
    hidden_states: torch.Tensor,
    block_output: torch.Tensor,
    injection: torch.Tensor,
    name: str,
) -> torch.Tensor:
    # Keep the M-dependent TP reduction opaque to range compilation. The
    # prefill trace can contain already-reduced payloads, whereas the same
    # compiled range executes un-reduced small verification batches.
    module = _PARTIAL_MODULES[name]
    reduced = module._reduce_partial(block_output)
    return hc_combine(hidden_states, reduced, injection, module.hc_count)


def _hcx_final_mix(
    hidden_states: torch.Tensor,
    block_output: torch.Tensor,
    injection: torch.Tensor,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    module = _PARTIAL_MODULES[name]
    reduced = module._reduce_partial(block_output)
    hidden, block, _ = module._combine_and_mix_reduced(
        hidden_states, reduced, injection
    )
    return hidden.contiguous(), block.contiguous()


direct_register_custom_op(
    op_name="qwen38_sm70_hcx_final_mix",
    op_func=_hcx_final_mix,
    fake_impl=lambda hidden_states, block_output, injection, name: (
        torch.empty_like(hidden_states),
        block_output.new_empty((hidden_states.shape[0], block_output.shape[1])),
    ),
)


direct_register_custom_op(
    op_name="qwen38_sm70_hcx_combine",
    op_func=_hcx_combine,
    fake_impl=lambda hidden_states, block_output, injection, name: torch.empty_like(
        hidden_states
    ),
)


direct_register_custom_op(
    op_name="qwen38_sm70_hcx_combine_and_mix",
    op_func=_hcx_combine_and_mix,
    fake_impl=_hcx_combine_and_mix_fake,
)

__all__ = [
    "GatedResidual",
    "GroupedGemmaRMSNorm",
    "HyperConnectionConfig",
]
