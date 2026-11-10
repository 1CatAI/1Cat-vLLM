# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ordered GDN warmup over bound operators and explicit weight/layout inputs.

No model object or callback crosses this boundary. Dummy tensors belong to
one stage, and shape deduplication uses the caller's engine-owned set. The
existing allocator flush remains after all four ordered stages.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from vllm.config.gdn_schedule import GdnScheduleConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.fla.ops import gdn_diagnostics as diagnostics
from vllm.model_executor.layers.fla.ops.gdn_preparation import prepare_prefill
from vllm.model_executor.layers.fla.ops.gdn_selector import GdnExecutionPlan
from vllm.model_executor.layers.fla.ops.gdn_stages import (
    GdnHeadContract,
    mixed_qkv_recurrence,
)
from vllm.model_executor.layers.fla.ops.sm70.gdn_decode import (
    FlashQlaDecodeAdmission,
    flashqla_decode,
    mixed_qkv_decode_layout,
)
from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
from vllm.model_executor.warmup.plan import run_warmup_tasks, warmup_unconditional
from vllm.utils.torch_utils import _encode_layer_name

# Preserve the diagnostic logger used by the former layer-local warmup.
logger = init_logger("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn")


class PrefillOperator(Protocol):
    gdn_prefill_backend: str
    execution_plan: GdnExecutionPlan

    def __call__(self, **kwargs: Any) -> Any: ...


@dataclass
class GdnWarmup:
    heads: GdnHeadContract
    prefill: PrefillOperator
    decode_admission: FlashQlaDecodeAdmission
    native_policy: Any
    schedule: GdnScheduleConfig | None
    conv_weight: torch.Tensor
    conv_bias: torch.Tensor | None
    activation: str
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    state_dtype: torch.dtype
    device: torch.device
    dtype: torch.dtype
    input_width: int
    qkv_dim: int
    qkv_row_stride: int
    spec_cache_stride: int
    conv_dim_first: bool
    prefix: str
    trace: bool
    decode_warmup: bool
    prefill_token_counts: tuple[int, ...] = (FLA_CHUNK_SIZE, max(1, FLA_CHUNK_SIZE - 1))

    # Preserve the recurrence metadata from the last attempted convolution,
    # including early failure at T=64 (normally the last attempt uses T=63).
    _last_conv_tokens: int = field(default=FLA_CHUNK_SIZE, init=False)

    def run(self, warmed: set[tuple]) -> None:
        key = (
            self.device.type,
            self.device.index,
            self.dtype,
            self.state_dtype,
            self.prefill.gdn_prefill_backend,
            self.prefill.execution_plan.original_prefill,
            self.prefill.execution_plan.direct_prefill_output,
            self.prefill.execution_plan.indexed_prefill,
            self.heads.local_k_heads,
            self.heads.local_v_heads,
            self.heads.head_k_dim,
            self.heads.head_v_dim,
            self.qkv_dim,
            self.qkv_row_stride,
            self.prefill_token_counts,
            self.conv_dim_first,
        )
        if key in warmed:
            if self.trace:
                logger.info(
                    "SM70 profile trace: GDN prefill/decode warmup skip "
                    "layer=%s reason=shape_already_warmed",
                    self.prefix,
                )
            return
        warmed.add(key)
        run_warmup_tasks(
            (
                warmup_unconditional("gdn.convolution", self.convolution),
                warmup_unconditional("gdn.preparation", self.preparation),
                warmup_unconditional("gdn.prefill", self.recurrence_prefill),
                warmup_unconditional("gdn.decode", self.recurrence_decode),
            )
        )
        torch.accelerator.empty_cache()

    def _can_use_flashqla_decode(
        self, mixed_qkv, state_indices, tokens, *, layer_name, stage
    ):
        reason = self.decode_admission.rejection(mixed_qkv, state_indices, tokens)
        diagnostics.log_decode_route(
            layer_name=layer_name,
            stage=stage,
            decision="take" if reason is None else "skip",
            reason=reason or "ok",
            mixed_qkv=mixed_qkv,
            state_indices=state_indices,
            num_decode_tokens=tokens,
        )
        return reason is None

    def _forward_core_decode_flashqla(self, **kwargs):
        return flashqla_decode(
            self.heads,
            self.A_log,
            self.dt_bias,
            native_policy=self.native_policy,
            **kwargs,
        )

    def convolution(self):
        device, dtype = self.device, self.dtype
        qkv_dim, qkv_row_stride = self.qkv_dim, self.qkv_row_stride
        prefill_token_counts = self.prefill_token_counts
        # All kernels use BT = chunk_size, so a single pass with T = chunk_size
        # is sufficient to populate every autotuner cache. Also run the
        # conv1d prefill kernel once; otherwise Qwen3.5/Next can still JIT
        # _causal_conv1d_fwd_kernel on the first real request after the JIT
        # monitor has been activated.
        conv_weights = self.conv_weight.view(
            self.conv_weight.size(0), self.conv_weight.size(2)
        )
        if self.conv_dim_first:
            dummy_conv_state = torch.zeros(
                1,
                conv_weights.shape[0],
                conv_weights.shape[1] - 1,
                device=device,
                dtype=dtype,
            )
        else:
            dummy_conv_state = torch.zeros(
                1,
                conv_weights.shape[1] - 1,
                conv_weights.shape[0],
                device=device,
                dtype=dtype,
            ).transpose(-1, -2)
        spec_cache_stride = self.spec_cache_stride
        dummy_cache_indices_variants = [
            torch.zeros(1, device=device, dtype=torch.int32)
        ]
        if spec_cache_stride > 1:
            dummy_cache_base = torch.zeros(
                spec_cache_stride, device=device, dtype=torch.int32
            )
            dummy_cache_indices_variants.append(
                torch.as_strided(
                    dummy_cache_base,
                    size=(1,),
                    stride=(spec_cache_stride,),
                )
            )
        dummy_has_initial_state = torch.ones(1, device=device, dtype=torch.bool)
        try:
            for prefill_tokens in prefill_token_counts:
                dummy_conv_inputs = [
                    torch.randn(
                        prefill_tokens,
                        qkv_dim,
                        device=device,
                        dtype=dtype,
                    ).transpose(0, 1)
                ]
                if qkv_row_stride > qkv_dim:
                    dummy_qkv = torch.randn(
                        prefill_tokens,
                        qkv_row_stride,
                        device=device,
                        dtype=dtype,
                    )
                    dummy_conv_inputs.append(dummy_qkv[:, :qkv_dim].transpose(0, 1))
                cu_seqlens = torch.tensor(
                    [0, prefill_tokens], device=device, dtype=torch.int32
                )
                self._last_conv_tokens = prefill_tokens
                for dummy_conv_in in dummy_conv_inputs:
                    for dummy_cache_indices in dummy_cache_indices_variants:
                        causal_conv1d_fn(
                            dummy_conv_in,
                            conv_weights,
                            self.conv_bias,
                            activation=self.activation,
                            conv_states=dummy_conv_state,
                            cache_indices=dummy_cache_indices,
                            has_initial_state=dummy_has_initial_state,
                            query_start_loc=cu_seqlens,
                        )
                        dummy_conv_state.zero_()
        except Exception:
            logger.warning(
                "GDN causal-conv prefill warmup failed for layer %s. "
                "First inference may JIT _causal_conv1d_fwd_kernel.",
                self.prefix,
                exc_info=True,
            )
        else:
            logger.debug(
                "GDN causal-conv prefill warmup completed for layer %s",
                self.prefix,
            )

    def preparation(self):
        device, dtype = self.device, self.dtype
        num_v_heads = self.heads.local_v_heads
        qkv_dim, qkv_row_stride = self.qkv_dim, self.qkv_row_stride
        prefill_token_counts = self.prefill_token_counts
        # Mirror the real prefill path here: build q/k/v/g/beta via
        # fused_post_conv_prep and then run chunk_gated_delta_rule with
        # in-kernel L2 norm disabled.
        for prefill_tokens in prefill_token_counts:
            compact_a = torch.randn(
                prefill_tokens, num_v_heads, device=device, dtype=dtype
            )
            compact_b = torch.randn(
                prefill_tokens, num_v_heads, device=device, dtype=dtype
            )
            gate_inputs = [(compact_a, compact_b)]
            gate_row_stride = 2 * num_v_heads
            if gate_row_stride > num_v_heads:
                dummy_ba = torch.randn(
                    prefill_tokens,
                    gate_row_stride,
                    device=device,
                    dtype=dtype,
                )
                # Split-projection Qwen keeps b as a view into [b, a], while
                # a is contiguous when compile-graph slicing uses index_select.
                gate_inputs.append((compact_a, dummy_ba[:, :num_v_heads]))
                # Keep the pure view case warmed as well for non-graph runs.
                gate_inputs.append(
                    (
                        dummy_ba[:, num_v_heads:gate_row_stride],
                        dummy_ba[:, :num_v_heads],
                    )
                )
            dummy_post_conv_inputs = [
                torch.randn(prefill_tokens, qkv_dim, device=device, dtype=dtype)
            ]
            if qkv_row_stride > qkv_dim:
                dummy_qkv_post = torch.randn(
                    prefill_tokens,
                    qkv_row_stride,
                    device=device,
                    dtype=dtype,
                )
                dummy_post_conv_inputs.append(dummy_qkv_post[:, :qkv_dim])
            for dummy_mixed_qkv in dummy_post_conv_inputs:
                for dummy_a, dummy_b in gate_inputs:
                    q, k, v, g, beta = prepare_prefill(
                        self.heads,
                        dummy_mixed_qkv,
                        dummy_a,
                        dummy_b,
                        self.A_log,
                        self.dt_bias,
                    )
                    del q, k, v, g, beta

    def recurrence_prefill(self):
        device, dtype = self.device, self.dtype
        num_k_heads, num_v_heads = self.heads.local_k_heads, self.heads.local_v_heads
        state_dtype = self.state_dtype
        cu_seqlens = torch.tensor(
            [0, self._last_conv_tokens], device=self.device, dtype=torch.int32
        )
        # The warmup is only meant to compile/autotune the GDN prefill
        # kernels. Keep the actual kernel inputs deterministic and bounded so
        # random profile tensors cannot trigger pathological TileLang runtime
        # waits before real inference starts.
        T = FLA_CHUNK_SIZE
        q = torch.full(
            (1, T, num_k_heads, self.heads.head_k_dim),
            1.0e-3,
            device=device,
            dtype=dtype,
        )
        k = torch.full(
            (1, T, num_k_heads, self.heads.head_k_dim),
            1.0e-3,
            device=device,
            dtype=dtype,
        )
        v = torch.full(
            (1, T, num_v_heads, self.heads.head_v_dim),
            1.0e-3,
            device=device,
            dtype=dtype,
        )
        g = torch.zeros(
            1,
            T,
            num_v_heads,
            device=device,
            dtype=torch.float32,
        )
        beta = torch.full(
            (1, T, num_v_heads),
            1.0e-2,
            device=device,
            dtype=torch.float32,
        )
        state = torch.zeros(
            1,
            num_v_heads,
            self.heads.head_v_dim,
            self.heads.head_k_dim,
            device=device,
            dtype=state_dtype,
        )
        dummy_prefill_core_attn_out = None
        if (
            self.prefill.gdn_prefill_backend == "flashqla_sm70"
            and self.prefill.execution_plan.original_prefill
            and self.prefill.execution_plan.direct_prefill_output
        ):
            dummy_prefill_core_attn_out = torch.empty(
                T,
                num_v_heads,
                self.heads.head_v_dim,
                device=device,
                dtype=dtype,
            )
        # CuteDSL kernels require metadata
        chunk_indices = None
        chunk_offsets = None
        if self.prefill.gdn_prefill_backend == "cutedsl":
            from vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl import (
                prepare_metadata_cutedsl,
            )

            chunk_indices, chunk_offsets = prepare_metadata_cutedsl(cu_seqlens, T)

        trace_prefill_warmup = (
            self.trace
            and self.prefill.gdn_prefill_backend == "flashqla_sm70"
            and self.prefill.execution_plan.original_prefill
        )
        prefill_warmup_start = time.perf_counter()
        if trace_prefill_warmup:
            logger.info(
                "SM70 profile trace: GDN prefill warmup enter layer=%s "
                "T=%d q_shape=%s v_shape=%s direct_output=%s",
                self.prefix,
                T,
                tuple(q.shape),
                tuple(v.shape),
                dummy_prefill_core_attn_out is not None,
            )
        try:
            self.prefill(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                chunk_offsets=chunk_offsets,
                use_qk_l2norm_in_kernel=False,
                core_attn_out=dummy_prefill_core_attn_out,
            )
        except Exception:
            logger.warning(
                "GDN prefill kernel warmup (T=%d) failed for "
                "layer %s. First inference may OOM due to "
                "autotuner.",
                T,
                self.prefix,
                exc_info=True,
            )
        else:
            if trace_prefill_warmup:
                torch.accelerator.synchronize()
                logger.info(
                    "SM70 profile trace: GDN prefill warmup exit layer=%s "
                    "T=%d elapsed_ms=%.3f",
                    self.prefix,
                    T,
                    (time.perf_counter() - prefill_warmup_start) * 1000.0,
                )
            logger.debug(
                "GDN prefill kernel warmup (T=%d) completed for layer %s",
                T,
                self.prefix,
            )
            if (
                self.prefill.gdn_prefill_backend == "flashqla_sm70"
                and self.prefill.execution_plan.original_prefill
                and self.prefill.execution_plan.indexed_prefill
            ):
                indexed_state = torch.zeros(
                    1,
                    num_v_heads,
                    self.heads.head_v_dim,
                    self.heads.head_k_dim,
                    device=device,
                    dtype=state_dtype,
                )
                indexed_out = torch.empty(
                    T,
                    num_v_heads,
                    self.heads.head_v_dim,
                    device=device,
                    dtype=dtype,
                )
                indexed_state_indices = torch.zeros(1, device=device, dtype=torch.long)
                indexed_has_initial_state = torch.ones(
                    1, device=device, dtype=torch.bool
                )
                try:
                    self.prefill(
                        q=q,
                        k=k,
                        v=v,
                        g=g,
                        beta=beta,
                        initial_state=indexed_state,
                        output_final_state=False,
                        cu_seqlens=cu_seqlens,
                        chunk_indices=chunk_indices,
                        chunk_offsets=chunk_offsets,
                        state_indices=indexed_state_indices,
                        has_initial_state=indexed_has_initial_state,
                        inplace_final_state=True,
                        use_qk_l2norm_in_kernel=False,
                        core_attn_out=(
                            indexed_out
                            if self.prefill.execution_plan.direct_prefill_output
                            else None
                        ),
                    )
                except Exception:
                    logger.warning(
                        "GDN indexed/direct FlashQLA prefill warmup (T=%d) "
                        "failed for layer %s. First real prefill may JIT the "
                        "indexed original TileLang variant.",
                        T,
                        self.prefix,
                        exc_info=True,
                    )
                else:
                    logger.debug(
                        "GDN indexed/direct FlashQLA prefill warmup (T=%d) "
                        "completed for layer %s",
                        T,
                        self.prefix,
                    )
                finally:
                    del (
                        indexed_state,
                        indexed_out,
                        indexed_state_indices,
                        indexed_has_initial_state,
                    )

    def recurrence_decode(self):
        device, dtype = self.device, self.dtype
        num_v_heads = self.heads.local_v_heads
        state_dtype = self.state_dtype
        decode_qkv_dim = self.qkv_dim
        for decode_tokens in (1, 2):
            dummy_decode_views = [
                torch.zeros(
                    decode_tokens,
                    decode_qkv_dim,
                    device=device,
                    dtype=dtype,
                )
            ]
            if self.input_width > decode_qkv_dim:
                dummy_decode_qkvz = torch.zeros(
                    decode_tokens,
                    self.input_width,
                    device=device,
                    dtype=dtype,
                )
                dummy_decode_views.append(dummy_decode_qkvz[:, :decode_qkv_dim])
            for dummy_decode_mixed_qkv in dummy_decode_views:
                dummy_decode_a = torch.zeros(
                    decode_tokens, num_v_heads, device=device, dtype=dtype
                )
                dummy_decode_b = torch.zeros(
                    decode_tokens, num_v_heads, device=device, dtype=dtype
                )
                dummy_decode_state = torch.zeros(
                    1,
                    num_v_heads,
                    self.heads.head_v_dim,
                    self.heads.head_k_dim,
                    device=device,
                    dtype=state_dtype,
                )
                dummy_decode_out = torch.empty(
                    decode_tokens,
                    1,
                    num_v_heads,
                    self.heads.head_v_dim,
                    device=device,
                    dtype=dtype,
                )
                dummy_decode_indices = torch.zeros(
                    decode_tokens, device=device, dtype=torch.int32
                )
                dummy_decode_query_start_loc = torch.arange(
                    decode_tokens + 1, device=device, dtype=torch.int32
                )
                trace_decode_warmup = self.trace
                mixed_decode_warmup_start = time.perf_counter()
                if trace_decode_warmup:
                    logger.info(
                        "SM70 profile trace: GDN mixed-QKV decode warmup enter "
                        "layer=%s tokens=%d mixed_shape=%s mixed_stride=%s "
                        "layout=%s",
                        self.prefix,
                        decode_tokens,
                        tuple(dummy_decode_mixed_qkv.shape),
                        tuple(dummy_decode_mixed_qkv.stride()),
                        mixed_qkv_decode_layout(dummy_decode_mixed_qkv),
                    )
                try:
                    mixed_qkv_recurrence(
                        self.heads,
                        A_log=self.A_log,
                        a=dummy_decode_a,
                        b=dummy_decode_b,
                        dt_bias=self.dt_bias,
                        mixed_qkv=dummy_decode_mixed_qkv,
                        initial_state=dummy_decode_state,
                        out=dummy_decode_out,
                        cu_seqlens=dummy_decode_query_start_loc,
                        state_indices=dummy_decode_indices,
                        schedule=self.schedule,
                    )
                except Exception:
                    logger.warning(
                        "GDN mixed-QKV decode warmup failed for layer %s. "
                        "First inference may JIT "
                        "fused_sigmoid_gating_delta_rule_update_kernel.",
                        self.prefix,
                        exc_info=True,
                    )
                else:
                    if trace_decode_warmup:
                        torch.accelerator.synchronize()
                        logger.info(
                            "SM70 profile trace: GDN mixed-QKV decode warmup "
                            "exit layer=%s elapsed_ms=%.3f",
                            self.prefix,
                            (time.perf_counter() - mixed_decode_warmup_start) * 1000.0,
                        )
                    logger.debug(
                        "GDN mixed-QKV decode warmup completed for layer %s",
                        self.prefix,
                    )
                if self._can_use_flashqla_decode(
                    dummy_decode_mixed_qkv,
                    dummy_decode_indices,
                    decode_tokens,
                    layer_name=_encode_layer_name(self.prefix),
                    stage="warmup",
                ):
                    if not self.decode_warmup:
                        if trace_decode_warmup:
                            logger.info(
                                "SM70 profile trace: GDN FlashQLA decode "
                                "warmup skip layer=%s reason=disabled",
                                self.prefix,
                            )
                    else:
                        flashqla_decode_warmup_start = time.perf_counter()
                        if trace_decode_warmup:
                            logger.info(
                                "SM70 profile trace: GDN FlashQLA decode "
                                "warmup enter layer=%s tokens=%d layout=%s",
                                self.prefix,
                                decode_tokens,
                                mixed_qkv_decode_layout(dummy_decode_mixed_qkv),
                            )
                        try:
                            self._forward_core_decode_flashqla(
                                mixed_qkv=dummy_decode_mixed_qkv,
                                a=dummy_decode_a,
                                b=dummy_decode_b,
                                ssm_state=dummy_decode_state,
                                state_indices=dummy_decode_indices,
                                num_decode_tokens=decode_tokens,
                                cu_seqlens=dummy_decode_query_start_loc,
                                core_attn_out=dummy_decode_out.squeeze(1),
                            )
                        except Exception:
                            logger.warning(
                                "GDN FlashQLA decode warmup failed for layer %s. "
                                "First inference may JIT flash_qla_sm70_gdn.",
                                self.prefix,
                                exc_info=True,
                            )
                        else:
                            if trace_decode_warmup:
                                torch.accelerator.synchronize()
                                logger.info(
                                    "SM70 profile trace: GDN FlashQLA decode "
                                    "warmup exit layer=%s elapsed_ms=%.3f",
                                    self.prefix,
                                    (time.perf_counter() - flashqla_decode_warmup_start)
                                    * 1000.0,
                                )
                            logger.debug(
                                "GDN FlashQLA decode warmup completed for layer %s",
                                self.prefix,
                            )
                del (
                    dummy_decode_a,
                    dummy_decode_b,
                    dummy_decode_state,
                    dummy_decode_out,
                    dummy_decode_indices,
                    dummy_decode_query_start_loc,
                )
            del dummy_decode_views
