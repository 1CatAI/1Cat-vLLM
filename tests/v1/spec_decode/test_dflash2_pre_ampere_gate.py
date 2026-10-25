# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash2 range preservation follows the resolved runtime dtype.

The legacy helper's device fallback must still follow the worker's own card,
while constructors use the actual dtype inherited from the target.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models import qwen3_dflash2 as dflash2

pytestmark = pytest.mark.cpu_test

SM70 = (7, 0)
SM75 = (7, 5)
SM80 = (8, 0)
SM89 = (8, 9)

BF16_CONFIG = SimpleNamespace(dtype=torch.bfloat16)


@dataclass
class Node:
    """Per-index capabilities of a node and the index this worker runs on."""

    capabilities: dict[int, tuple[int, int]] = field(
        default_factory=lambda: {0: SM75, 1: SM70, 2: SM80}
    )
    index: int = 0


def _as_tuple(capability: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(capability, int):
        return divmod(capability, 10)
    return capability


@pytest.fixture
def node(monkeypatch) -> Node:
    state = Node()

    def fake_has_device_capability(capability, device_id=0):
        return state.capabilities[device_id] >= _as_tuple(capability)

    def fake_is_device_capability(capability, device_id=0):
        return state.capabilities[device_id] == _as_tuple(capability)

    monkeypatch.delenv("VLLM_SM70_DFLASH2_BF16_EMULATION", raising=False)
    monkeypatch.setattr(dflash2.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        dflash2.current_platform,
        "has_device_capability",
        fake_has_device_capability,
    )
    monkeypatch.setattr(
        dflash2.current_platform,
        "is_device_capability",
        fake_is_device_capability,
    )
    monkeypatch.setattr(
        dflash2.torch.accelerator,
        "current_device_index",
        lambda: state.index,
    )
    return state


def test_volta_worker_takes_the_emulation(node: Node) -> None:
    node.index = 1
    assert dflash2._use_sm70_bf16_emulation(BF16_CONFIG)


def test_turing_worker_takes_the_emulation(node: Node) -> None:
    """The regression: Turing has no BF16 arithmetic either, but the old
    exact-SM70 check sent it down the plain FP16 path."""
    node.index = 0
    assert dflash2._use_sm70_bf16_emulation(BF16_CONFIG)


def test_ampere_worker_keeps_native_bf16(node: Node) -> None:
    node.index = 2
    assert not dflash2._use_sm70_bf16_emulation(BF16_CONFIG)


def test_gate_does_not_answer_for_device_zero(node: Node) -> None:
    """Device 0 is Ampere here; the Turing worker must still emulate."""
    node.capabilities = {0: SM80, 1: SM75}
    node.index = 1
    assert dflash2._use_sm70_bf16_emulation(BF16_CONFIG)


def test_fp16_checkpoint_needs_no_emulation(node: Node) -> None:
    node.index = 0
    fp16_config = SimpleNamespace(dtype=torch.float16)
    assert not dflash2._use_sm70_bf16_emulation(fp16_config)


def test_switch_disables_the_emulation(monkeypatch, node: Node) -> None:
    monkeypatch.setenv("VLLM_SM70_DFLASH2_BF16_EMULATION", "0")
    node.index = 0
    assert not dflash2._use_sm70_bf16_emulation(BF16_CONFIG)


def test_non_cuda_platform_never_emulates(monkeypatch) -> None:
    monkeypatch.setattr(dflash2.current_platform, "is_cuda", lambda: False)
    assert not dflash2._use_sm70_bf16_emulation(BF16_CONFIG)


def test_flashinfer_topk_gate_follows_the_worker_device(
    monkeypatch, node: Node
) -> None:
    """The selector's FlashInfer top-k has no pre-SM80 kernel. Device 0 is
    Ampere here; the Turing worker must still fall back to torch.topk."""
    node.capabilities = {0: SM80, 1: SM75}
    node.index = 1
    monkeypatch.setattr(dflash2, "has_flashinfer", lambda: True)
    dflash2._flashinfer_topk.cache_clear()
    try:
        assert dflash2._flashinfer_topk() is None
    finally:
        dflash2._flashinfer_topk.cache_clear()


@pytest.mark.parametrize("capability", [SM70, SM75, SM80, SM89])
@pytest.mark.parametrize(
    "runtime_dtype", [torch.float16, torch.bfloat16, torch.float32]
)
@pytest.mark.parametrize("checkpoint_dtype", [torch.bfloat16, torch.float16])
def test_explicit_runtime_dtype_takes_precedence_over_device(
    node, capability, runtime_dtype, checkpoint_dtype
):
    node.capabilities = {0: capability}
    config = SimpleNamespace(dtype=checkpoint_dtype)
    assert dflash2._use_sm70_bf16_emulation(config, runtime_dtype) == (
        checkpoint_dtype == torch.bfloat16 and runtime_dtype == torch.float16
    )


@pytest.mark.parametrize("declared_dtype", [torch.bfloat16, "bf16", "bfloat16"])
def test_legacy_torch_dtype_field_preserves_range_on_ada(node, declared_dtype):
    node.capabilities = {0: SM89}
    config = SimpleNamespace(torch_dtype=declared_dtype)
    assert dflash2._use_sm70_bf16_emulation(config, torch.float16)


def test_explicit_runtime_dtype_respects_switch_and_platform(monkeypatch, node):
    node.capabilities = {0: SM89}
    monkeypatch.setenv("VLLM_SM70_DFLASH2_BF16_EMULATION", "0")
    assert not dflash2._use_sm70_bf16_emulation(BF16_CONFIG, torch.float16)
    monkeypatch.delenv("VLLM_SM70_DFLASH2_BF16_EMULATION")
    monkeypatch.setattr(dflash2.current_platform, "is_cuda", lambda: False)
    assert not dflash2._use_sm70_bf16_emulation(BF16_CONFIG, torch.float16)


@pytest.fixture
def small_draft(monkeypatch, node):
    """Replace distributed construction only; keep production dtype wrappers."""
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import RMSNorm

    config = SimpleNamespace(
        dtype=torch.bfloat16,
        hidden_size=8,
        vocab_size=16,
        rms_norm_eps=1e-5,
        dflash_config={
            "conv_kernel_size": 2,
            "conv_group_size": 2,
            "selector_rank": 2,
            "selector_top_k": 2,
        },
    )

    class LocalLinear(torch.nn.Linear):
        def __init__(self, input_size, output_size, *, params_dtype, **kwargs):
            super().__init__(input_size, output_size, bias=False, dtype=params_dtype)
            self.quant_method = dflash2.UnquantizedLinearMethod()

    def init_layer(self, vllm_config, **kwargs):
        torch.nn.Module.__init__(self)
        dtype = vllm_config.model_config.dtype
        self.self_attn = torch.nn.Module()
        self.self_attn.qkv_proj = LocalLinear(8, 24, params_dtype=dtype)
        self.self_attn.o_proj = LocalLinear(8, 8, params_dtype=dtype)
        self.mlp = torch.nn.Module()
        self.mlp.gate_up_proj = LocalLinear(8, 16, params_dtype=dtype)
        self.mlp.down_proj = LocalLinear(8, 8, params_dtype=dtype)
        self.input_layernorm = RMSNorm(8, eps=1e-5, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(8, eps=1e-5, dtype=dtype)

    def init_model(self, *, vllm_config, **kwargs):
        torch.nn.Module.__init__(self)
        self.config = config
        self.hidden_norm = RMSNorm(8, eps=1e-5, dtype=vllm_config.model_config.dtype)
        self.norm = RMSNorm(8, eps=1e-5, dtype=vllm_config.model_config.dtype)

    monkeypatch.setattr(dflash2.DFlashQwen3DecoderLayer, "__init__", init_layer)
    monkeypatch.setattr(dflash2.DFlashQwen3Model, "__init__", init_model)
    monkeypatch.setattr(dflash2, "ReplicatedLinear", LocalLinear)
    # TP4 also qualifies for the SM70-specific projection optimization. An
    # Ada worker must not gain that optimization when it gains range safety.
    monkeypatch.setattr(dflash2, "get_tensor_model_parallel_world_size", lambda: 4)
    engine_config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    with set_current_vllm_config(engine_config):
        yield config


@pytest.mark.parametrize("capability", [SM70, SM89])
@pytest.mark.parametrize("runtime_dtype", [torch.float16, torch.bfloat16])
def test_constructors_preserve_weight_paths_and_select_runtime_math(
    node, small_draft, capability, runtime_dtype
):
    node.capabilities = {0: capability}
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=runtime_dtype),
        speculative_config=SimpleNamespace(num_speculative_tokens=7),
    )
    layer = dflash2.DFlash2Qwen3DecoderLayer(
        vllm_config, config=small_draft, layer_idx=0
    )
    model = dflash2.DFlash2Qwen3Model(vllm_config=vllm_config)
    enabled = runtime_dtype == torch.float16
    assert layer.use_sm70_bf16_emulation == enabled
    assert model.use_sm70_bf16_emulation == enabled
    assert isinstance(layer.mlp, dflash2.DFlashSM70MLP) == enabled
    for norm in (
        layer.input_layernorm,
        layer.post_attention_layernorm,
        model.hidden_norm,
        model.norm,
    ):
        assert isinstance(norm, dflash2.DFlashSM70RMSNorm) == enabled
    assert getattr(layer.self_attn, "output_input_scale", 1.0) == (
        dflash2.DFLASH_SM70_WIDE_OUTPUT_SCALE if enabled else 1.0
    )
    assert not layer.self_attn._sm70_dflash2_qk_rope
    assert getattr(layer.self_attn.o_proj, "_sm70_dflash2_fp16_m8", False) == (
        enabled and capability == SM70
    )

    # BF16/FP16 transport must load identical checkpoint paths, with no new
    # wrapper prefix and no lost trained norm or convolution parameters.
    expected = {
        "self_attn.qkv_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_up_proj.weight",
        "mlp.down_proj.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "attention_conv.base_kernel",
        "attention_conv.kernel_projection.weight",
        "mlp_conv.base_kernel",
        "mlp_conv.kernel_projection.weight",
    }
    assert set(dict(layer.named_parameters())) == expected
    for module in (layer, model):
        weights = {
            name: torch.full_like(parameter, 0.5)
            for name, parameter in module.named_parameters()
        }
        module.load_state_dict(weights, strict=True)
        for parameter in module.parameters():
            assert parameter.dtype == runtime_dtype
            assert torch.all(parameter == 0.5)

    if enabled:
        # Reproduce the failure at the actual convolution: its FP32 result
        # exceeds FP16's range. Scaling the attention projection input keeps
        # conv transport finite, then the installed norm restores FP32 range.
        attention_output = torch.full((8, 8), 25000.0, dtype=runtime_dtype)
        coefficients = torch.ones((8, 2, 4), dtype=runtime_dtype)
        unscaled = layer.attention_conv.finish(
            attention_output.float(), coefficients.float()
        )
        assert unscaled.max() > torch.finfo(torch.float16).max
        assert not torch.isfinite(
            layer.attention_conv.finish(attention_output, coefficients)
        ).all()
        transported = layer.attention_conv.finish(
            attention_output / layer.self_attn.output_input_scale, coefficients
        )
        assert torch.isfinite(transported).all()
        residual = torch.full((8, 8), 70000.0, dtype=torch.float32)
        normalized, combined = layer.post_attention_layernorm(transported, residual)
        assert torch.isfinite(normalized).all() and torch.isfinite(combined).all()
        assert combined.dtype == torch.float32
        assert combined.max() > torch.finfo(torch.float16).max
        expected_residual = (transported.float() * 256 + residual).bfloat16().float()
        torch.testing.assert_close(combined, expected_residual, rtol=0, atol=0)
        assert model.hidden_norm.residual_scale == 1.0
