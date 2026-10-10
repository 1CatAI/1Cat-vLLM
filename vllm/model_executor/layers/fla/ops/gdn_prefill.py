# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN prefill providers, preserving layout, casts and output-buffer contracts."""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fla.ops import (
    chunk_gated_delta_rule as fla_chunk_gated_delta_rule,
)
from vllm.model_executor.layers.fla.ops import gdn_diagnostics as diagnostics
from vllm.model_executor.layers.fla.ops.gdn_profiling import GdnPrefillProfiler
from vllm.model_executor.layers.fla.ops.gdn_selector import GdnExecutionPlan
from vllm.model_executor.layers.fla.ops.gdn_stages import normalize_qk
from vllm.model_executor.layers.fla.ops.sm70.gdn_prefill import (
    flashqla_sm70_chunk_gated_delta_rule,
)

logger = init_logger(__name__)


def fi_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
):
    from flashinfer.gdn_prefill import (
        chunk_gated_delta_rule as chunk_gated_delta_rule_fi,
    )

    if use_qk_l2norm_in_kernel:
        q, k = normalize_qk(q, k)

    # use flashinfer implementation
    q = q.squeeze(0).contiguous()
    k = k.squeeze(0).contiguous()
    v = v.squeeze(0).contiguous()

    g = g.squeeze(0).contiguous()
    beta = beta.squeeze(0).contiguous()
    fi_state = initial_state.to(torch.float32)
    fi_g = g.to(torch.float32)
    fi_beta = beta.to(torch.float32)
    result = chunk_gated_delta_rule_fi(
        q=q,
        k=k,
        v=v,
        g=torch.exp(fi_g),
        beta=fi_beta,
        initial_state=fi_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    # FlashInfer returns (output, state) when output_final_state=True,
    # or just output when output_final_state=False.
    # Unsqueeze back to 4D (1, L, H, D) to match fla output format
    if output_final_state:
        output, final_state = result
        return output.unsqueeze(0), final_state
    else:
        return result.unsqueeze(0), None


class GdnPrefill:
    """Bound computation only: no model/layer callbacks and no legacy inputs."""

    def __init__(
        self,
        plan: GdnExecutionPlan,
        profiler: GdnPrefillProfiler,
        chunk_kernels=None,
        native_policy=None,
    ):
        self.native_policy = native_policy
        self.chunk_kernels = chunk_kernels
        self.execution_plan = plan
        self.profiler = profiler
        self.gdn_prefill_backend = plan.prefill.backend
        self._forward_method = {
            "flashinfer": self.forward_cuda,
            "flashqla_sm70": self.forward_flashqla_sm70,
            "cutedsl": self.forward_cutedsl,
            "triton": self.forward_native,
        }[self.gdn_prefill_backend]
        self._call_prefill = self._forward_method

    def execute_prefill(
        self,
        q,
        k,
        v,
        g,
        beta,
        *,
        ssm_state,
        state_indices,
        has_initial_state,
        cu_seqlens,
        chunk_indices,
        chunk_offsets,
        use_qk_l2norm_in_kernel,
        gate_is_exp,
        core_attn_out,
        layer_name,
        num_tokens,
    ):
        """Gather or index state, execute the selected provider, then commit.

        The caller owns request ordering and output-buffer admission. Indexed
        original FlashQLA writes the pool in place; other providers return a
        final state which is cast and scattered in the original order.
        """
        assert state_indices is not None
        assert has_initial_state is not None
        use_indexed_original_prefill = (
            self.gdn_prefill_backend == "flashqla_sm70"
            and self.execution_plan.original_prefill
            and self.execution_plan.indexed_prefill
            and state_indices.ndim == 1
        )
        profile_start = self.profiler.start()
        if use_indexed_original_prefill:
            initial_state = ssm_state
        else:
            initial_state = ssm_state[state_indices].contiguous()  # type: ignore[index]
            initial_state[~has_initial_state, ...] = 0  # type: ignore[operator]
        self.profiler.end(
            layer_name,
            "state_gather",
            profile_start,
            tokens=num_tokens,
            details=(
                f"state_shape={tuple(initial_state.shape)} "
                f"indices_contig={state_indices.is_contiguous()} "
                f"indexed={use_indexed_original_prefill}"
            ),
        )
        if use_indexed_original_prefill:
            diagnostics.capture_state_slice(
                "prefill_initial_state",
                layer_name,
                ssm_state,
                state_indices,
                int(has_initial_state.shape[0]),
            )
        else:
            diagnostics.capture_tensor(
                "prefill_initial_state",
                layer_name,
                initial_state,
                "state",
            )
        profile_start = self.profiler.start()
        chunk_kwargs = {
            "q": q,
            "k": k,
            "v": v,
            "g": g,
            "beta": beta,
            "initial_state": initial_state,
            "output_final_state": not use_indexed_original_prefill,
            "cu_seqlens": cu_seqlens,
            "chunk_indices": chunk_indices,
            "chunk_offsets": chunk_offsets,
            "use_qk_l2norm_in_kernel": use_qk_l2norm_in_kernel,
            "gate_is_exp": gate_is_exp,
            "core_attn_out": core_attn_out,
        }
        if self.gdn_prefill_backend == "flashqla_sm70":
            chunk_kwargs.update(
                {
                    "state_indices": (
                        state_indices if use_indexed_original_prefill else None
                    ),
                    "has_initial_state": (
                        has_initial_state if use_indexed_original_prefill else None
                    ),
                    "inplace_final_state": use_indexed_original_prefill,
                }
            )
        (
            core_attn_out_non_spec,
            last_recurrent_state,
        ) = self._call_prefill(**chunk_kwargs)
        self.profiler.end(
            layer_name,
            "core_call",
            profile_start,
            tokens=num_tokens,
            details=f"backend={self.gdn_prefill_backend}",
        )
        diagnostics.capture_tensor(
            "prefill_core_out",
            layer_name,
            core_attn_out_non_spec,
            "core",
        )
        if last_recurrent_state is not None:
            diagnostics.capture_tensor(
                "prefill_last_recurrent_state",
                layer_name,
                last_recurrent_state,
                "state",
            )
        # Init cache
        profile_start = self.profiler.start()
        if not use_indexed_original_prefill:
            assert last_recurrent_state is not None
            ssm_state[state_indices] = last_recurrent_state.to(ssm_state.dtype)
        self.profiler.end(
            layer_name,
            "state_writeback",
            profile_start,
            tokens=num_tokens,
            details=(
                f"state_dtype={ssm_state.dtype} indexed={use_indexed_original_prefill}"
            ),
        )
        diagnostics.capture_state_slice(
            "prefill_post_ssm_state",
            layer_name,
            ssm_state,
            state_indices,
            int(has_initial_state.shape[0]),
        )
        return core_attn_out_non_spec, last_recurrent_state

    def forward_cuda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
        gate_is_exp: bool = False,
    ):
        if gate_is_exp:
            g = torch.log(g)
        o, final_state = fi_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        if core_attn_out is not None:
            o_flat = o.squeeze(0).reshape(-1)
            co_flat = core_attn_out.reshape(-1)
            co_flat[: o_flat.numel()].copy_(o_flat)
        return o, final_state

    def forward_flashqla_sm70(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        state_indices: torch.Tensor | None = None,
        has_initial_state: torch.Tensor | None = None,
        inplace_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
        gate_is_exp: bool = False,
    ):
        if (
            q.dtype != torch.float16
            or k.dtype != torch.float16
            or v.dtype != torch.float16
        ):
            logger.warning_once(
                "FlashQLA-SM70 GDN prefill only runs on fp16 q/k/v tensors "
                "for SM70/V100 production use; got q=%s k=%s v=%s. Falling "
                "using the native GDN path for this call.",
                q.dtype,
                k.dtype,
                v.dtype,
            )
            return self.forward_native(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                chunk_offsets=chunk_offsets,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                core_attn_out=core_attn_out,
                gate_is_exp=gate_is_exp,
            )
        o, final_state = flashqla_sm70_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            state_indices=state_indices,
            has_initial_state=has_initial_state,
            inplace_final_state=inplace_final_state,
            use_original_tilelang=self.execution_plan.original_prefill,
            profiler=self.profiler,
            native_policy=self.native_policy,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            core_attn_out=core_attn_out,
            gate_is_exp=gate_is_exp,
        )
        return o, final_state

    def forward_native(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
        gate_is_exp: bool = False,
    ):
        if gate_is_exp:
            g = torch.log(g)
        return fla_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            core_attn_out=core_attn_out,
            kernels=self.chunk_kernels,
        )

    def forward_cutedsl(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
        gate_is_exp: bool = False,
    ):
        from vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl import (
            chunk_gated_delta_rule_cutedsl,
        )

        if use_qk_l2norm_in_kernel:
            q, k = normalize_qk(q, k)
        if gate_is_exp:
            g = torch.log(g)

        assert cu_seqlens is not None
        assert chunk_indices is not None
        assert chunk_offsets is not None

        o, final_state = chunk_gated_delta_rule_cutedsl(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            core_attn_out=core_attn_out,
        )
        if not output_final_state:
            final_state = None
        return o, final_state
