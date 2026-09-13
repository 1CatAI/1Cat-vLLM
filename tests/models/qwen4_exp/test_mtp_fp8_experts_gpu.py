# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.mtp_fp8_checkpoint import (
    pad_mtp_fp8_checkpoint_matrix,
)
from vllm.models.qwen4_exp.nvidia.mtp_fp8_experts import (
    MTPCheckpointFp8SM70MoEMethod,
    MTPFp8SM70MoEMethod,
    quantize_expert_rows,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="Requires SM70 native operators",
)


@pytest.fixture
def should_do_global_cleanup_after_test():
    return False


@pytest.mark.parametrize("num_tokens", [1, 2, 8, 64])
def test_padded_mtp_fp8_moe_matches_reconstructed_weights_and_graph(num_tokens):
    with torch.device("cuda"):
        torch.manual_seed(11)
        e, h, i, topk = 16, 2560, 160, 4
        layer = torch.nn.Module()
        layer.moe_config = SimpleNamespace(
            has_bias=False, is_act_and_mul=True, experts_per_token=topk
        )
        layer.local_num_experts = e
        layer.global_num_experts = e
        layer.expert_map = None
        layer.apply_router_weight_on_input = False
        method = MTPFp8SM70MoEMethod(layer)
        method.create_weights(layer, e, h, i, torch.float16)
        layer.w13_weight.data.normal_(std=0.02)
        layer.w2_weight.data.normal_(std=0.02)
        refs = []
        for weights in [layer.w13_weight, layer.w2_weight]:
            ref = []
            for weight in weights:
                q, s = quantize_expert_rows(weight)
                ref.append((q.float() * s).half())
            refs.append(torch.stack(ref))
        method.process_weights_after_loading(layer)
        assert not hasattr(layer, "w13_weight") and not hasattr(layer, "w2_weight")
        for m in [num_tokens]:
            x = torch.randn(m, h, dtype=torch.float16)
            ids = torch.stack([torch.randperm(e)[:topk] for _ in range(m)]).int()
            weights = torch.softmax(torch.randn(m, topk), dim=-1)
            ref = torch.zeros_like(x)
            for slot in range(topk):
                w13 = refs[0][ids[:, slot].long()]
                w2 = refs[1][ids[:, slot].long()]
                gate, up = torch.bmm(w13, x.unsqueeze(-1)).squeeze(-1).chunk(2, dim=-1)
                act = torch.nn.functional.silu(gate) * up
                out = torch.bmm(w2, act.unsqueeze(-1)).squeeze(-1)
                ref += (out.float() * weights[:, slot, None]).half()
            actual = method.apply(layer, x, weights, ids, None, None).clone()
            torch.testing.assert_close(actual, ref, atol=0.003, rtol=0.03)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = method.apply(layer, x, weights, ids, None, None)
            graph.replay()
            torch.accelerator.synchronize()
            torch.testing.assert_close(captured, actual, atol=0, rtol=0)


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("num_tokens", [1, 2, 8, 64])
def test_checkpoint_block_scales_preserve_tp_offsets_and_graph(rank, num_tokens):
    with torch.device("cuda"):
        torch.manual_seed(41)
        e, h, logical, topk = 4, 2560, 160, 2
        layer = torch.nn.Module()
        layer.moe_config = SimpleNamespace(
            has_bias=False, is_act_and_mul=True, experts_per_token=topk
        )
        layer.local_num_experts = layer.global_num_experts = e
        layer.expert_map = None
        layer.apply_router_weight_on_input = False
        method = MTPCheckpointFp8SM70MoEMethod(layer)
        _, padded = method.maybe_roundup_sizes(
            h, logical, torch.float16, SimpleNamespace(tp_size=4)
        )
        method.create_weights(layer, e, h, padded, torch.float16)
        raw_refs: list[list[torch.Tensor]] = [[], [], []]
        for expert in range(e):
            for index in range(3):
                down = index == 2
                shape = (h, 640) if down else (640, h)
                q = torch.randint(-16, 17, shape).to(torch.float8_e4m3fn)
                scale = (
                    (torch.rand(tuple(dim // 128 for dim in shape)) * 0.002 + 0.001)
                    .to(torch.bfloat16)
                    .float()
                )
                axis = 1 if down else 0
                ref = (
                    q.float()
                    * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
                ).half()
                raw_refs[index].append(ref.narrow(axis, rank * logical, logical))
                weight, scales = pad_mtp_fp8_checkpoint_matrix(
                    q, scale, down_projection=down, tp_size=4
                )
                local_w = weight.narrow(axis, rank * padded, padded)
                local_s = scales.narrow(axis, rank * padded // 128, padded // 128)
                if down:
                    layer.w2_weight.data[expert].copy_(local_w)
                    layer.w2_weight_scale_inv.data[expert].copy_(local_s)
                else:
                    layer.w13_weight.data[
                        expert, index * padded : (index + 1) * padded
                    ].copy_(local_w)
                    layer.w13_weight_scale_inv.data[
                        expert, index * padded // 128 : (index + 1) * padded // 128
                    ].copy_(local_s)
        refs = [torch.stack(weights) for weights in raw_refs]
        method.process_weights_after_loading(layer)
        assert not hasattr(layer, "w13_weight") and not hasattr(layer, "w2_weight")
        x = torch.randn(num_tokens, h, dtype=torch.float16)
        ids = torch.stack([torch.randperm(e)[:topk] for _ in range(num_tokens)]).int()
        weights = torch.softmax(torch.randn(num_tokens, topk), dim=-1)
        reference = torch.zeros_like(x)
        for slot in range(topk):
            chosen = ids[:, slot].long()
            gate = torch.bmm(refs[0][chosen], x.unsqueeze(-1)).squeeze(-1)
            up = torch.bmm(refs[1][chosen], x.unsqueeze(-1)).squeeze(-1)
            hidden = torch.nn.functional.silu(gate) * up
            out = torch.bmm(refs[2][chosen], hidden.unsqueeze(-1)).squeeze(-1)
            reference += (out.float() * weights[:, slot, None]).half()
        actual = method.apply(layer, x, weights, ids, None, None).clone()
        torch.testing.assert_close(actual, reference, atol=0.003, rtol=0.03)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = method.apply(layer, x, weights, ids, None, None)
        graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(captured, actual, atol=0, rtol=0)
