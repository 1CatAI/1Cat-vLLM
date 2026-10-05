# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Canonical draft-store scope and alias checks without allocating model weights."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from vllm.models.qwen4_exp.nvidia import sm70_mtp_structural as structural


@pytest.mark.parametrize("canonical", [False, True])
def test_canonical_probe_uses_one_byte_store_and_keeps_target_scope(
    monkeypatch, canonical
):
    from vllm.model_executor.layers.fused_moe import layer as layer_module
    from vllm.model_executor.layers.fused_moe import (
        unquantized_fused_moe_method as method_module,
    )
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.models.qwen4_exp.nvidia import sm70_mtp_int8

    class Method:
        is_monolithic = False

        def apply(self, *args):
            raise AssertionError("Canonical dispatch must not use FP16 fallback")

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.quant_method = Method()
            self.activation = MoEActivation.SILU
            self.apply_router_weight_on_input = False
            self.expert_map = None
            self.w13_bias = self.w2_bias = None

    monkeypatch.setattr(layer_module, "FusedMoE", Layer)
    monkeypatch.setattr(method_module, "UnquantizedFusedMoEMethod", Method)
    monkeypatch.setattr(
        torch.ops._C, "sm70_mtp_moe_int8_block32_chain_out", object(), raising=False
    )
    with FakeTensorMode() as mode:

        def fake(shape, dtype):
            return FakeTensor(
                mode,
                torch.empty(shape, dtype=dtype, device="meta"),
                torch.device("cuda"),
            )

        def prepare(weight, *, block32):
            assert block32
            n, k = weight.shape
            return fake((n // 32, k // 16, 32, 16), torch.uint8), fake(
                (k // 32, n), torch.float16
            )

        monkeypatch.setattr(sm70_mtp_int8, "prepare_int8_expert_weight", prepare)
        draft = nn.Module()
        draft.layer = Layer()
        draft.layer.w13_weight = nn.Parameter(
            fake((512, 320, 2560), torch.float16), requires_grad=False
        )
        draft.layer.w2_weight = nn.Parameter(
            fake((512, 2560, 160), torch.float16), requires_grad=False
        )
        original = draft.layer.w13_weight, draft.layer.w2_weight
        assert (
            structural.prepare_draft_expert_qpn8_probe(
                draft, integer=True, block32=True, canonical=canonical
            )
            == 1
        )
        if canonical:
            for role, weight in (
                ("13", draft.layer.w13_weight),
                ("2", draft.layer.w2_weight),
            ):
                codes = getattr(draft.layer, "_sm70_mtp_qpn8_codes" + role)
                assert weight.dtype == torch.int8
                assert weight.untyped_storage()._cdata == codes.untyped_storage()._cdata
            assert all(p.dtype == torch.int8 for p in draft.parameters())
            for rows in (4, 20):
                x = fake((rows, 2560), torch.float16)
                out = structural._draft_expert_apply(
                    draft.layer.quant_method,
                    draft.layer,
                    x,
                    fake((rows, 10), torch.float32),
                    fake((rows, 10), torch.int32),
                    None,
                    None,
                )
                assert out.shape == x.shape
        else:
            assert draft.layer.w13_weight is original[0]
            assert draft.layer.w2_weight is original[1]


def test_canonical_probe_rejects_incompatible_format_and_existing_clone():
    with pytest.raises(ValueError, match="block32 INT8"):
        structural.prepare_draft_expert_qpn8_probe(nn.Module(), canonical=True)
    with pytest.raises(ValueError, match="precede"):
        structural.prepare_draft_expert_qpn8_probe(
            SimpleNamespace(_sm70_decode_graph_model=object()),
            integer=True,
            block32=True,
            canonical=True,
        )
