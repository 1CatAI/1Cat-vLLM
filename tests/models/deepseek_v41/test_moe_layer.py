# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L-MOE: the DeepseekV41MoE block (router + FusedMoE experts + shared expert + FP32 combine) on synthetic
weights, single rank: sm70 impl vs torch impl, decode GEMV vs cuBLAS path, spill vs resident, gate loading."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.models.deepseek_v41.sm70 import moe as v41_moe
from vllm.models.deepseek_v41.sm70.moe_method import make_spill_plan
from vllm.utils.torch_utils import set_default_torch_dtype

from .test_moe_method import HIDDEN, INTER, SyntheticCheckpoint

pytestmark = pytest.mark.sm70


class _V41TestQuantConfig(QuantizationConfig):
    """Stand-in for L-CORE's DeepseekV41FP8Config: FusedMoE -> make_v41_moe_method, linears unquantized."""

    def get_name(self):
        return "ds41_moe_test"

    def get_supported_act_dtypes(self):
        return [torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config):
        return cls()

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, FusedMoE):
            return v41_moe.make_v41_moe_method(layer.moe_config)
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        return None


def _hf_config() -> SimpleNamespace:
    return SimpleNamespace(hidden_size=HIDDEN, moe_intermediate_size=INTER, swiglu_limit=10.0,
                           routed_scaling_factor=1.5, scoring_func="sqrtsoftplus", topk_method="noaux_tc",
                           norm_topk_prob=True, n_shared_experts=1)


def build_block(n_experts: int, top_k: int, ckpt: SyntheticCheckpoint, *, spill=None, layer_id: int = 3,
                prefix: str = "model.layers.3.ffn") -> v41_moe.DeepseekV41MoE:
    vllm_config = VllmConfig()
    object.__setattr__(vllm_config, "model_config", SimpleNamespace(hf_config=_hf_config(), dtype=torch.float16))
    object.__setattr__(vllm_config, "quant_config", _V41TestQuantConfig())
    vllm_config.compilation_config.static_forward_context = {}
    with set_current_vllm_config(vllm_config), set_default_torch_dtype(torch.float16), torch.device("cuda"):
        block = v41_moe.DeepseekV41MoE(vllm_config, prefix, layer_id, n_routed_experts=n_experts, top_k=top_k,
                                       spill=spill)
    g = torch.Generator(device="cuda").manual_seed(ckpt.seed)
    gate_bf16 = (torch.randn(n_experts, HIDDEN, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    gate_bf16[0, :4] = torch.tensor([2.4e-8, -3e-7, 0.3, -0.25], device="cuda", dtype=torch.bfloat16)
    block.gate.weight.weight_loader(block.gate.weight, gate_bf16)
    block.gate.e_score_correction_bias.data.copy_(10.8 + 0.035 * torch.randn(n_experts, device="cuda",
                                                                              generator=g))
    block._test_gate_bf16 = gate_bf16  # noqa: SLF001 (test bookkeeping)
    for e in range(n_experts):
        for shard, (codes, scales) in ckpt.expert(e).items():
            base = "w13" if shard in ("w1", "w3") else "w2"
            for suffix, tensor in (("weight", codes), ("weight_scale", scales)):
                param = getattr(block.experts, f"{base}_{suffix}")
                param.weight_loader(param, tensor, f"experts.{base}_{suffix}", shard_id=shard, expert_id=e,
                                    return_success=True)
    sh_w1 = (torch.randn(INTER, HIDDEN, device="cuda", generator=g) * 0.01).half()
    sh_w3 = (torch.randn(INTER, HIDDEN, device="cuda", generator=g) * 0.01).half()
    sh_w2 = (torch.randn(HIDDEN, INTER, device="cuda", generator=g) * 0.01).half()
    gu = block.shared_experts.gate_up_proj
    gu.weight_loader(gu.weight, sh_w1, 0)
    gu.weight_loader(gu.weight, sh_w3, 1)
    dp = block.shared_experts.down_proj
    dp.weight_loader(dp.weight, sh_w2)
    block.experts.quant_method.process_weights_after_loading(block.experts)
    return block


@pytest.fixture
def moe_env(dist_init, monkeypatch):
    def _set(**env: str) -> None:
        for key in ("VLLM_DS41_MOE_IMPL", "VLLM_DS41_MOE_BACKEND", "VLLM_DS41_MOE_DECODE_GEMV"):
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
    return _set


@pytest.fixture(scope="module")
def ckpt() -> SyntheticCheckpoint:
    return SyntheticCheckpoint(seed=23)


def _x(num_tokens: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(num_tokens, HIDDEN, device="cuda", generator=g) * 0.7).half()


def _rel_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()


def test_gate_weight_exact_and_logits(moe_env, ckpt):
    moe_env()
    block = build_block(128, 3, ckpt)
    assert block.gate.weight_exp > 10
    torch.testing.assert_close(block.gate.true_weight_fp32(), block._test_gate_bf16.float(), rtol=0, atol=0)
    for num_tokens in (1, 8, 33):
        x = _x(num_tokens, seed=num_tokens)
        ref = x.float() @ block._test_gate_bf16.float().t()
        assert _rel_rms(block.gate_logits(x), ref) < 1e-6


def test_gate_loader_rejects_inexact():
    gate = v41_moe.DeepseekV41Gate(4, 8).cuda()
    w = torch.zeros(4, 8, device="cuda")
    w[0, 0], w[0, 1] = 1.0, 1e-12  # 2^40 dynamic range cannot fit FP16 exactly
    with pytest.raises(ValueError, match="not exact in FP16"):
        gate.weight.weight_loader(gate.weight, w)


def test_router_kernel_matches_reference_semantics(moe_env, ckpt):
    moe_env()
    block = build_block(384, 6, ckpt)
    x = _x(512, seed=3)
    logits = block.gate_logits(x)
    w, ids = block.route(logits)
    scores = F.softplus(logits).sqrt()
    ref_ids = (scores + block.gate.e_score_correction_bias).topk(6, dim=-1)[1]
    assert torch.equal(ids.sort(-1)[0].long(), ref_ids.sort(-1)[0])
    ref_w = scores.gather(1, ids.long())
    ref_w = ref_w / (ref_w.sum(-1, keepdim=True) + 1e-20) * 1.5
    assert (w - ref_w).abs().max().item() < 1e-5 * ref_w.abs().max().item()


@pytest.mark.parametrize("num_tokens", [1, 2, 5, 8, 64, 300])
def test_sm70_matches_torch_impl(moe_env, ckpt, num_tokens):
    moe_env(VLLM_DS41_MOE_IMPL="torch")
    oracle = build_block(384, 6, ckpt)
    moe_env()
    fast = build_block(384, 6, ckpt, prefix="model.layers.4.ffn", layer_id=4)
    x = _x(num_tokens, seed=10 + num_tokens)
    out = fast(x)
    ref = oracle(x)
    assert out.dtype == torch.float32 and out.shape == (num_tokens, HIDDEN)
    _, ids_fast = fast.route(fast.gate_logits(x))
    _, ids_ref = oracle.route(oracle.gate_logits(x))
    assert torch.equal(ids_fast.sort(-1)[0], ids_ref.sort(-1)[0])
    assert _rel_rms(out, ref) < 3e-4


def test_decode_gemv_matches_cublas_path(moe_env, ckpt):
    moe_env(VLLM_DS41_MOE_DECODE_GEMV="0")
    slow = build_block(128, 3, ckpt)
    moe_env()
    fast = build_block(128, 3, ckpt, prefix="model.layers.4.ffn", layer_id=4)
    for num_tokens in (1, 3, 8):
        x = _x(num_tokens, seed=num_tokens)
        assert _rel_rms(fast(x), slow(x)) < 1e-5


def test_zero_padding_rows_finite(moe_env, ckpt):
    moe_env()
    block = build_block(128, 3, ckpt)
    x = _x(8, seed=1)
    x[3:] = 0
    out = block(x)
    assert torch.isfinite(out).all()


def test_spill_block_bitwise(moe_env, ckpt):
    moe_env()
    resident = build_block(384, 6, ckpt)
    spilled = build_block(384, 6, ckpt, spill=make_spill_plan(116, 384), prefix="model.layers.4.ffn",
                          layer_id=4)
    for num_tokens in (1, 4, 40):
        x = _x(num_tokens, seed=num_tokens + 77)
        torch.testing.assert_close(spilled(x), resident(x), rtol=0, atol=0)


def test_expert_params_mapping_shape():
    mapping = v41_moe.DeepseekV41MoE.expert_params_mapping(384)
    assert len(mapping) == 3 * 384
    assert mapping[0] == ("experts.w13_", "experts.0.w1.", 0, "w1")
    assert ("experts.w2_", "experts.383.w2.", 383, "w2") in mapping


def test_topk_check_refuses_graph_capture(moe_env, ckpt, monkeypatch):
    """The debug TP checksum syncs the host: under capture it must refuse loudly, never silently skip."""
    moe_env()
    block = build_block(384, 6, ckpt)
    _, ids = block.route(block.gate_logits(_x(2, 0)))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="CUDA-graph capture"):
        block._check_topk_consistent(ids)
