# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Snapshot-free, indexed FP32 GDN verification and state publication.

Factors only live between the verifier and post-sampling state publication.
The running state is always materialized before returning control to the
scheduler, so preemption, prefix reuse, prefill, and plain decode never inherit
an uncommitted factor bank. Publication batches all layers into one launch.
"""

from dataclasses import replace

import torch

from vllm import envs
from vllm.config.sm70_dflash2 import capture_sm70_dflash2_config
from vllm.platforms import current_platform

_NATIVE_OPS = (
    "gdn_wy_verify_sm70_out",
    "gdn_wy_commit_sm70",
    "gdn_wy_commit_group_sm70",
    "gdn_wy_finish_group_sm70",
    "gdn_wy_commit_group_v2_sm70",
    "gdn_wy_finish_group_v2_sm70",
)

for _name in _NATIVE_OPS:
    if hasattr(torch.ops._C, _name):
        torch.library.register_fake(f"_C::{_name}")(lambda *args, **kwargs: None)


def admits_wy_cache(config, layer) -> bool:
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

    spec = config.speculative_config
    policy = capture_sm70_dflash2_config(config)
    return bool(
        current_platform.is_cuda()
        and current_platform.is_device_capability(70)
        and policy is not None
        and policy.qualified
        and policy.fused_gdn_verify
        and spec is not None
        and spec.method == "dflash"
        and spec.num_speculative_tokens == 7
        and not spec.ngram_assist
        and config.lora_config is None
        and config.parallel_config.pipeline_parallel_size == 1
        and not config.parallel_config.enable_dbo
        and config.cache_config.mamba_cache_mode == "align"
        and not envs.VLLM_MAMBA_ALIGN_CPU_POSTPROCESS
        and layer.model_config.dtype == torch.float16
        and layer.get_state_dtype()[1] == torch.float32
        and layer.get_state_dtype()[0] == torch.float16
        and not is_conv_state_dim_first()
        and layer.tp_size == 4
        and layer.hidden_size == 5120
        and layer.num_k_heads == 16
        and layer.num_v_heads == 48
        and layer.head_k_dim == layer.head_v_dim == 128
        and layer.conv_kernel_size == 4
        and not layer.gqa_interleaved_layout
        and all(hasattr(torch.ops._C, name) for name in _NATIVE_OPS)
    )


class WYStateCache:
    def __init__(self, capacity: int, device):
        self.u = torch.empty(capacity, 8, 12, 128, dtype=torch.float32, device=device)
        self.k = torch.empty(capacity, 8, 4, 128, dtype=torch.float32, device=device)
        self.G = torch.empty(capacity, 8, 12, dtype=torch.float32, device=device)
        self.pending = torch.full((capacity,), -1, dtype=torch.int32, device=device)
        # Graph padding may expose up to eight empty metadata rows per request.
        self.no_previous = torch.zeros(capacity * 8, dtype=torch.int32, device=device)

    @property
    def size_bytes(self):
        return sum(
            t.numel() * t.element_size()
            for t in (self.u, self.k, self.G, self.pending, self.no_previous)
        )

    @staticmethod
    def cache_spec(spec):
        return replace(spec, num_speculative_blocks=0, gdn_wy=True)

    def verify(self, layer, mixed, a, b, out, state, cu, indices, masks, nseq):
        # Opaque graph metadata can pad sequence rows to the token count.
        # Live requests never exceed the scheduler's request capacity.
        nseq = min(nseq, self.pending.numel())
        assert masks is not None and nseq <= self.no_previous.numel()
        q, k, v = mixed.split([512, 512, 1536], dim=1)
        torch.ops._C.gdn_wy_verify_sm70_out(
            out.reshape(-1, 1536),
            state,
            self.u,
            self.k,
            self.G,
            q,
            k,
            v,
            a,
            b,
            layer.A_log,
            layer.dt_bias,
            indices[:nseq],
            masks,
            self.pending,
            cu[: nseq + 1],
            self.no_previous[:nseq],
            self.u,
            self.k,
            self.G,
            128**-0.5,
            3,
        )


class WYCommitGroup:
    """Bind persistent layer/workspace pointers after KV allocation."""

    def __init__(self, kv_config, forward_context, group_ids, device):
        self.states, self.pending, self.conv = [], [], []
        desc, conv_desc = [], []
        for group_index, gid in enumerate(group_ids):
            group = kv_config.kv_cache_groups[gid]
            if not group.kv_cache_spec.gdn_wy:
                continue
            for name in group.layer_names:
                layer = forward_context[name]
                state = layer.kv_cache[1]
                cache = layer.gdn_wy_cache
                assert cache is not None
                self.states.append(state)
                self.pending.append(cache.pending)
                conv = layer.kv_cache[0]
                self.conv.append(conv)
                conv_desc.append([conv.data_ptr(), conv.stride(0), conv.size(2)])
                desc.append(
                    [
                        state.data_ptr(),
                        state.stride(0),
                        cache.u.data_ptr(),
                        cache.k.data_ptr(),
                        cache.G.data_ptr(),
                        cache.pending.data_ptr(),
                        group_index,
                        state.size(0),
                        cache.pending.numel(),
                    ]
                )
        self.descriptors = torch.tensor(desc, dtype=torch.int64, device=device)
        self.conv_descriptors = torch.tensor(
            conv_desc, dtype=torch.int64, device=device
        )

    def commit(self, ctx, num_reqs, accepted):
        if not self.states or not num_reqs:
            return
        torch.ops._C.gdn_wy_commit_group_sm70(
            self.states,
            self.pending,
            self.descriptors,
            ctx.block_table_ptrs,
            ctx.block_table_stride_req,
            accepted[:num_reqs],
            ctx.num_scheduled_tokens_buf.gpu[:num_reqs],
            ctx.num_computed_tokens_buf.gpu[:num_reqs],
            ctx.num_draft_tokens_buf.gpu[:num_reqs],
            ctx.block_size,
            False,
        )

    def finish(self, ctx, num_reqs, accepted):
        if not self.states or not num_reqs:
            return
        torch.ops._C.gdn_wy_finish_group_sm70(
            self.pending,
            self.conv,
            self.descriptors,
            self.conv_descriptors,
            ctx.block_table_ptrs,
            ctx.block_table_stride_req,
            accepted[:num_reqs],
            ctx.num_scheduled_tokens_buf.gpu[:num_reqs],
            ctx.num_computed_tokens_buf.gpu[:num_reqs],
            ctx.num_draft_tokens_buf.gpu[:num_reqs],
            ctx.block_size,
            ctx.num_accepted_tokens_out,
            ctx.spec_state_slot_selectors_out,
        )

    def commit_v2(self, ctx, mapping, accepted, computed):
        torch.ops._C.gdn_wy_commit_group_v2_sm70(
            self.states,
            self.pending,
            self.descriptors,
            ctx.block_table_ptrs,
            ctx.block_table_stride_req,
            accepted,
            computed,
            mapping,
            ctx.block_size,
        )

    def finish_v2(self, ctx, mapping, accepted, computed):
        torch.ops._C.gdn_wy_finish_group_v2_sm70(
            self.pending,
            self.conv,
            self.descriptors,
            self.conv_descriptors,
            ctx.block_table_ptrs,
            ctx.block_table_stride_req,
            ctx.num_accepted_tokens_out,
            computed,
            mapping,
            ctx.block_size,
            accepted,
        )
