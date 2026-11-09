# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bailing V3 integration contracts without loading a full model."""

from types import SimpleNamespace as NS

import pytest
import torch
from transformers import PretrainedConfig

from vllm.model_executor.layers.mamba import mamba_utils
from vllm.model_executor.layers.quantization import fp8, mxfp4
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.models import bailing_moe_v3 as model

pytestmark = pytest.mark.skip_global_cleanup


def bare(cls):
    value = object.__new__(cls)
    torch.nn.Module.__init__(value)
    return value


@pytest.mark.parametrize("layout", ["SD", "DS"])
@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("spec", [0, 3])
def test_model_and_layer_cache_contract(monkeypatch, layout, tp, spec):
    monkeypatch.setattr(mamba_utils, "get_conv_state_layout", lambda: layout)
    config = NS(
        model_config=NS(
            dtype=torch.float16,
            hf_config=NS(num_attention_heads=8, head_dim=16, short_conv_kernel_size=4),
        ),
        parallel_config=NS(tensor_parallel_size=tp),
        cache_config=NS(mamba_cache_dtype="auto"),
        speculative_config=NS(num_speculative_tokens=spec) if spec else None,
    )
    layer = bare(model.BailingMoeV3KimiDeltaAttention)
    layer.tp_size, layer.num_heads, layer.head_dim = tp, 8, 16
    layer.conv_size, layer.num_speculative_tokens = 4, spec
    layer.model_config, layer.cache_config = config.model_config, config.cache_config
    cls = model.BailingMoeV3ForCausalLM
    conv = (384 // tp, 3 + spec)
    expected = (conv if layout == "DS" else conv[::-1], (8 // tp, 16, 16))
    assert cls.get_mamba_state_shape_from_config(config) == expected
    assert layer.get_state_shape() == expected
    assert layer.get_state_dtype() == (torch.float16, torch.float32)
    assert cls.get_mamba_state_dtype_from_config(config) == layer.get_state_dtype()
    assert len(cls.get_mamba_state_copy_func()) == 2


def test_mxfp4_config_and_dispatch_respect_exclusions(monkeypatch):
    cfg = fp8.Fp8Config.from_config(
        {"quant_method": "fp8", "activation_scheme": "dynamic", "store_dtype": "mxfp4"}
    )
    assert cfg.store_dtype == "mxfp4"
    layer = bare(fp8.RoutedExperts)
    layer.moe_config = object()
    selected, skipped = object(), object()
    calls = []

    def choose(config):
        calls.append(config)
        return selected

    monkeypatch.setattr(
        mxfp4,
        "make_deepseek_v4_mxfp4_moe_method",
        choose,
    )
    monkeypatch.setattr(fp8, "UnquantizedFusedMoEMethod", lambda _: skipped)
    assert cfg.get_quant_method(layer, "model.layers.1.mlp.experts") is selected
    assert calls == [layer.moe_config]
    cfg.ignored_layers = ["model.layers.1.mlp.experts"]
    assert cfg.get_quant_method(layer, "model.layers.1.mlp.experts") is skipped
    assert len(calls) == 1


def test_suffix_exclusions_are_segment_bounded_and_preserve_exact_default():
    prefix = "model.layers.0.self_attn.b_proj"
    assert not is_layer_skipped(prefix, ["b_proj"])
    assert is_layer_skipped(prefix, ["b_proj"], match_mode="suffix")
    assert not is_layer_skipped(prefix, ["proj"], match_mode="suffix")
    mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    with pytest.raises(ValueError, match="some but not all"):
        is_layer_skipped(
            "model.layers.0.self_attn.qkv_proj",
            ["q_proj"],
            mapping,
            match_mode="suffix",
        )


def test_ling_block_config_uses_supported_suffix_api(monkeypatch):
    cfg = fp8.Fp8Config(True, ignored_layers=["b_proj"], weight_block_size=[128, 128])
    hf = PretrainedConfig(
        quantization_config={
            "scale_fmt": "ue8m0",
            "routed_experts_quant_method": "mxfp4",
        }
    )
    model._configure_ling_fp8_quant_config(cfg, hf)
    assert cfg.store_dtype == "mxfp4" and cfg.is_scale_e8m0
    assert model._is_fp8_module_excluded(cfg, "model.layers.0.self_attn.b_proj")
    layer = bare(fp8.LinearBase)
    assert isinstance(
        cfg.get_quant_method(layer, "model.layers.0.self_attn.b_proj"),
        fp8.UnquantizedLinearMethod,
    )
    monkeypatch.setattr(model, "get_tensor_model_parallel_world_size", lambda: 4)
    assert model._get_block_fp8_mlp_padded_intermediate_size(cfg, 160, "mlp") == 512
    raw = torch.ones(320, 8)
    padded = model._pad_block_fp8_mlp_checkpoint_tensor(
        cfg, "mlp.gate_up_proj.weight", raw, 160, 512
    )
    assert padded.shape == (1024, 8)
    assert torch.equal(padded[:160], raw[:160])
    assert not padded[160:512].count_nonzero()
    assert torch.equal(padded[512:672], raw[160:])
    assert not padded[672:].count_nonzero()


def test_rope_and_mxfp4_weight_mapping_preserve_input():
    config = NS(
        rope_theta=10000,
        rope_parameters={"partial_rotary_factor": 0.5},
        rope_scaling={"type": "linear", "factor": 2.0},
    )
    assert model._build_rope_parameters(config) == {
        "rope_theta": 10000,
        "rope_type": "linear",
        "factor": 2.0,
    }
    assert config.rope_parameters["partial_rotary_factor"] == 0.5
    cfg = fp8.Fp8Config(store_dtype="mxfp4")
    name = "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv"
    value = torch.ones(1)
    assert list(model._maybe_remap_ling_mxfp4_weight_names([(name, value)], cfg)) == [
        (name.removesuffix("_inv"), value)
    ]


@pytest.mark.parametrize("safe", [False, True])
def test_forward_passes_safe_gate(monkeypatch, safe):
    layer = bare(model.BailingMoeV3KimiDeltaAttention)
    layer.separate_b_proj = False
    layer.projection_size_per_partition = 32
    layer.local_num_heads, layer.head_dim = 2, 16
    layer.safe_gate, layer.lower_bound, layer.prefix = safe, -5.0, "test"
    layer.A_log, layer.dt_bias = torch.zeros(2), torch.zeros(32)
    x = torch.randn(3, 32)
    layer.qkvb_proj = lambda x: (torch.cat((x, x, x, x[:, :2]), -1), None)
    layer.f_proj = layer.g_proj = layer.o_proj = lambda x: (x, None)
    layer.o_norm = lambda x, _: x
    seen = []

    def gate(g, a, dim, **kwargs):
        seen.append(kwargs)
        return g.reshape(3, 2, dim)

    monkeypatch.setattr(model, "fused_kda_gate", gate)
    monkeypatch.setattr(torch.ops.vllm, "bailing_v3_kda_attention", lambda *args: None)
    layer.forward(x, torch.arange(3), torch.empty_like(x))
    assert seen[0]["safe_gate"] is safe
    assert seen[0]["lower_bound"] == (-5.0 if safe else None)


@pytest.mark.parametrize("safe", [False, True])
def test_gpu_gate_formula(safe):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires owned SM70")
    g = torch.linspace(-15, 15, 128, device="cuda").reshape(2, 64).half()
    a = torch.tensor([-1.0, 0.0, 0.5, 1.0], device="cuda")
    bias = torch.linspace(-0.1, 0.1, 64, device="cuda")
    out = model.fused_kda_gate(g, a, 16, g_bias=bias, safe_gate=safe, lower_bound=-5.0)
    shifted = (g.double() + bias.double()).reshape(2, 4, 16)
    decay = a.double().exp()[None, :, None]
    expected = (
        -5 * torch.sigmoid(decay * shifted)
        if safe
        else -decay * torch.nn.functional.softplus(shifted)
    )
    torch.testing.assert_close(out.double(), expected, rtol=3e-5, atol=2e-6)


def test_weight_loader_maps_shards_and_skips_remote_pp(monkeypatch):
    instance = bare(model.BailingMoeV3ForCausalLM)
    instance.config = NS(num_hidden_layers=2)
    instance.quant_config = None
    instance.get_expert_mapping = lambda: []
    local = "model.layers.0.self_attn.qkvb_proj.weight"
    remote = "model.layers.1.self_attn.qkvb_proj.weight"
    params = {name: torch.nn.Parameter(torch.zeros(7, 4)) for name in (local, remote)}
    widths = (2, 2, 2, 1)

    def loader(param, value, shard):
        start = sum(widths[:shard])
        param.data[start : start + widths[shard]].copy_(value)

    for value in params.values():
        value.weight_loader = loader
    instance.named_parameters = lambda **kwargs: iter(params.items())
    monkeypatch.setattr(
        model, "is_pp_missing_parameter", lambda name, _: name == remote
    )
    weights = [
        (
            f"model.layers.{layer}.attention.{proj}.weight",
            torch.full((width, 4), i + 1.0),
        )
        for layer in (0, 1, 2)
        for i, (proj, width) in enumerate(
            zip(("q_proj", "k_proj", "v_proj", "b_proj"), widths)
        )
    ]
    assert instance.load_weights(weights) == {local}
    expected = torch.tensor([1, 1, 2, 2, 3, 3, 4])[:, None].expand(7, 4)
    assert torch.equal(params[local], expected)
    assert not params[remote].count_nonzero()


@pytest.mark.parametrize("layout", ["SD", "DS"])
def test_decode_updates_only_owned_states_and_replays(monkeypatch, layout):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires owned SM70")
    torch.manual_seed(846)
    monkeypatch.setattr(mamba_utils, "get_conv_state_layout", lambda: layout)
    layer = bare(model.BailingMoeV3KimiDeltaAttention)
    layer.prefix, layer.head_dim, layer.safe_gate, layer.lower_bound = (
        "test",
        16,
        True,
        -5.0,
    )
    heads, width, slots = 2, 32, 4
    weights = [torch.randn(width, 1, 4, device="cuda") * 0.1 for _ in range(3)]
    layer.q_conv1d, layer.k_conv1d, layer.v_conv1d = [
        NS(weight=w, bias=None) for w in weights
    ]
    shape = (3 * width, 3) if layout == "DS" else (3, 3 * width)
    conv = torch.randn(slots, *shape, device="cuda", dtype=torch.float16) * 0.1
    state = torch.randn(slots, heads, 16, 16, device="cuda") * 0.1
    initial_conv, initial_state = conv.clone(), state.clone()
    layer.kv_cache = (conv, state)
    indices = torch.tensor([0, 2], device="cuda", dtype=torch.int32)
    metadata = NS(
        has_initial_state=None,
        spec_query_start_loc=None,
        non_spec_query_start_loc=torch.tensor(
            [0, 1, 2], device="cuda", dtype=torch.int32
        ),
        spec_state_indices_tensor=None,
        non_spec_state_indices_tensor=indices,
        spec_sequence_masks=None,
        spec_token_indx=None,
        non_spec_token_indx=None,
        num_accepted_tokens=None,
        num_actual_tokens=2,
        num_prefills=0,
        num_decodes=2,
        num_spec_decodes=0,
    )
    monkeypatch.setattr(model, "GDNAttentionMetadata", NS)
    monkeypatch.setattr(
        model, "get_forward_context", lambda: NS(attn_metadata={"test": metadata})
    )
    inputs = [
        torch.randn(2, width, device="cuda", dtype=torch.float16) * 0.1
        for _ in range(3)
    ]
    gates = torch.full((1, 2, heads, 16), -0.5, device="cuda")
    beta = torch.full((1, 2, heads), 0.3, device="cuda")
    output = torch.empty(1, 2, heads, 16, device="cuda", dtype=torch.float16)

    def call():
        layer._forward(*inputs, gates, beta, output)

    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(3):
        for x in inputs:
            x.normal_(0, 0.1)
        conv.copy_(initial_conv)
        state.copy_(initial_state)
        # causal_conv1d_update overwrites its projection input in place.
        original_inputs = [x.clone() for x in inputs]
        graph.replay()
        old = initial_conv if layout == "DS" else initial_conv.transpose(-1, -2)
        projections = []
        for x, w, history in zip(original_inputs, weights, old.chunk(3, dim=-2)):
            window = torch.cat((history[indices].float(), x.float().unsqueeze(-1)), -1)
            projected = torch.nn.functional.silu((window * w[:, 0]).sum(-1)).half()
            projections.append(projected.reshape(2, heads, 16).double())
        q, k, v = projections
        q = q * (q.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
        k = k * (k.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
        h = initial_state[indices].double() * gates[0].double().exp().unsqueeze(-2)
        delta = (v - (h * k.unsqueeze(-2)).sum(-1)) * beta[0].double().unsqueeze(-1)
        expected_state = h + delta.unsqueeze(-1) * k.unsqueeze(-2)
        expected = (expected_state * q.unsqueeze(-2)).sum(-1) * 0.25
        torch.testing.assert_close(output[0].double(), expected, rtol=0.005, atol=2e-4)
        torch.testing.assert_close(
            state[indices].double(), expected_state, rtol=0.005, atol=2e-4
        )
        assert torch.equal(state[[1, 3]], initial_state[[1, 3]])
        assert torch.equal(conv[[1, 3]], initial_conv[[1, 3]])
