# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU integration gates, not GPU arithmetic or endpoint performance evidence."""

from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.models.qwen4_exp.nvidia import sm70_fp16_gemv as gemv
from vllm.models.qwen4_exp.nvidia import sm70_fp16_hc as hc


@pytest.fixture(autouse=True)
def restore_precision_and_env_cache():
    backend = torch.backends.cuda.matmul
    names = (
        "allow_fp16_reduced_precision_reduction",
        "allow_bf16_reduced_precision_reduction",
        "allow_fp16_accumulation",
    )
    before = [getattr(backend, name) for name in names]
    envs.disable_envs_cache()
    yield
    for name, value in zip(names, before):
        setattr(backend, name, value)
    envs.disable_envs_cache()


@pytest.fixture
def config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            architectures=("Qwen4ExpForCausalLM",),
            dtype=torch.float16,
            hf_text_config=SimpleNamespace(
                hidden_size=2560,
                num_hidden_layers=48,
                num_experts=512,
                num_experts_per_tok=10,
                moe_intermediate_size=640,
                hc_count=4,
                hc_lowrank=320,
                num_attention_heads=24,
                num_key_value_heads=2,
                indexer_head_dim=128,
                indexer_budget=2048,
                indexer_compress_ratio=4,
            ),
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=4, use_ubatching=False),
        speculative_config=None,
    )


@pytest.mark.parametrize("rank", range(4))
def test_down_pack_preserves_bits_and_tp_ownership(rank):
    raw = torch.randint(-(2**15), 2**15, (336, 10240), dtype=torch.int16)
    packed = hc._pack_hc_batch_weight(raw.view(torch.float16), "down", rank)
    assert packed.shape == (3, 640, 2, 32, 8)
    restored = packed.permute(0, 3, 1, 2, 4).contiguous().view(96, 10240)
    assert torch.equal(restored[:88].view(torch.int16), raw[rank * 80 : rank * 80 + 88])
    assert not restored[88:].view(torch.int16).count_nonzero()
    if rank == 3:
        assert torch.equal(restored[80:84].view(torch.int16), raw[320:324])


@pytest.mark.parametrize("rank", range(4))
def test_up_pack_preserves_all_branch_bits(rank):
    raw = torch.randint(-(2**15), 2**15, (10240, 320), dtype=torch.int16)
    packed = hc._pack_hc_batch_weight(raw.view(torch.float16), "up", rank)
    assert packed.shape == (80, 20, 2, 4, 8, 8)
    restored = packed.permute(3, 0, 4, 1, 2, 5).contiguous().view(4, 640, 320)
    assert torch.equal(
        restored.view(torch.int16),
        raw.view(4, 2560, 320)[:, rank * 640 : (rank + 1) * 640],
    )


@pytest.mark.parametrize(
    "shape,dtype,role,rank",
    [
        ((336, 10240), torch.float16, "down", -1),
        ((336, 10240), torch.float16, "down", 4),
        ((336, 10240), torch.float32, "down", 0),
        ((336, 10240), torch.float16, "up", 0),
        ((324, 10240), torch.float16, "down", 0),
        ((10240, 320), torch.float16, "other", 0),
    ],
)
def test_bad_packing_rejected(shape, dtype, role, rank):
    with pytest.raises(ValueError):
        hc._pack_hc_batch_weight(torch.empty(shape, dtype=dtype), role, rank)


def test_batch_contract_does_not_admit_mtp_or_ubatching(config, monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    assert gemv._batch_runtime_contract(config)
    config.speculative_config = SimpleNamespace(method="mtp", num_speculative_tokens=4)
    assert gemv._exact_runtime_contract(config)  # Existing M1/MTP contract.
    assert not gemv._batch_runtime_contract(config)
    config.speculative_config = None
    config.parallel_config.use_ubatching = True
    assert not gemv._batch_runtime_contract(config)
    config.parallel_config.use_ubatching = False
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    assert not gemv._batch_runtime_contract(config)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    config.parallel_config.tensor_parallel_size = 2
    assert gemv._batch_runtime_contract(config)


def test_full_precision_also_applies_to_control(config, monkeypatch):
    monkeypatch.setenv("VLLM_SM70_QWEN38_FP16_GEMV", "0")
    monkeypatch.setenv("VLLM_SM70_QWEN38_BATCH_FASTPATH", "0")
    monkeypatch.setattr(gemv.current_platform, "is_device_capability", lambda _: True)
    backend = torch.backends.cuda.matmul
    backend.allow_fp16_reduced_precision_reduction = True
    backend.allow_bf16_reduced_precision_reduction = True
    backend.allow_fp16_accumulation = True
    gemv.enable_qwen38_sm70_fp16_gemv(torch.nn.Module(), torch.float16, config)
    assert not backend.allow_fp16_reduced_precision_reduction
    assert not backend.allow_bf16_reduced_precision_reduction
    assert not backend.allow_fp16_accumulation


@pytest.mark.parametrize("enabled", (False, True))
def test_nomtp_policy_does_not_overwrite_explicit_mtp_policy(
    config, monkeypatch, enabled
):
    config.speculative_config = SimpleNamespace(method="mtp", num_speculative_tokens=4)
    monkeypatch.setenv("VLLM_SM70_QWEN38_FP16_GEMV", "0")
    monkeypatch.setattr(gemv.current_platform, "is_device_capability", lambda _: True)
    backend = torch.backends.cuda.matmul
    for name in (
        "allow_fp16_reduced_precision_reduction",
        "allow_bf16_reduced_precision_reduction",
        "allow_fp16_accumulation",
    ):
        monkeypatch.setattr(backend, name, enabled)
    gemv.enable_qwen38_sm70_fp16_gemv(torch.nn.Module(), torch.float16, config)
    assert backend.allow_fp16_reduced_precision_reduction == enabled
    assert backend.allow_bf16_reduced_precision_reduction == enabled
    assert backend.allow_fp16_accumulation == enabled


@pytest.mark.parametrize("enabled", [False, True])
def test_hc_loader_tags_only_when_admitted(config, monkeypatch, enabled):
    monkeypatch.setenv("VLLM_SM70_QWEN38_FUSED_HC_FP16", "1")
    monkeypatch.setenv("VLLM_SM70_QWEN38_BATCH_FASTPATH", str(int(enabled)))
    monkeypatch.setenv("VLLM_SM70_QWEN4_EXP_ONLINE_QPN8", "0")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    monkeypatch.setattr(hc.current_platform, "is_device_capability", lambda _: True)
    module = torch.nn.Module()
    module.use_combine, module.lora_rank = True, 320
    module.hc_count, module.hidden_size = 4, 2560
    for name in ("input_mix_weight_down_block_inject", "input_mix_weight_up"):
        layer = torch.nn.Module()
        layer.quant_method = UnquantizedLinearMethod()
        module.add_module(name, layer)
    hc.enable_qwen38_sm70_fp16_fused_hc(module, torch.float16, config)
    assert module._sm70_qwen38_fp16_fused_hc
    for role, layer in zip(("down", "up"), module.children()):
        assert hasattr(layer, "_sm70_qwen38_hc_batch_role") == enabled
        if enabled:
            assert layer._sm70_qwen38_hc_batch_role == role
            assert isinstance(layer.quant_method, gemv.Qwen38SM70FP16LinearMethod)
        else:
            assert type(layer.quant_method) is UnquantizedLinearMethod


@pytest.mark.parametrize("rows", [1, 2, 3, 8, 16, 17])
def test_cpu_rejected_and_fake_shapes(rows, monkeypatch):
    monkeypatch.setenv("VLLM_SM70_QWEN38_BATCH_FASTPATH", "1")
    x = torch.empty(rows, 10240, dtype=torch.float16)
    assert not hc._batch_runtime_ok(x, None, None)
    fake = hc._qwen38_sm70_fp16_fused_hc_fake(x, x, x, x, x)
    assert [tuple(t.shape) for t in fake] == [(rows, 2560), (rows, 4)]


def test_packed_hc_survives_fake_export():
    class HC(torch.nn.Module):
        def forward(self, x, down, up, packed_down, packed_up):
            return torch.ops.vllm.qwen38_sm70_fp16_fused_hc(
                x, down, up, packed_down, packed_up
            )

    args = tuple(
        torch.empty(shape, device="meta", dtype=torch.float16)
        for shape in (
            (2, 10240),
            (336, 10240),
            (10240, 320),
            (3, 640, 2, 32, 8),
            (80, 20, 2, 4, 8, 8),
        )
    )
    exported = torch.export.export(HC(), args)
    calls = [
        node
        for node in exported.graph.nodes
        if node.target == torch.ops.vllm.qwen38_sm70_fp16_fused_hc.default
    ]
    assert len(calls) == 1 and len(calls[0].args) == 5


def test_hc_fusion_never_bypasses_adapter_or_quantized_forward(monkeypatch):
    monkeypatch.setattr(hc, "use_sm70_decode_graph_semantics", lambda: True)
    calls = []

    def record(*args):
        calls.append(args)
        return args[0], args[0]

    monkeypatch.setattr(
        torch.ops.vllm,
        "qwen38_sm70_fp16_fused_hc",
        record,
    )

    def plain(shape):
        layer = object.__new__(LinearBase)
        torch.nn.Module.__init__(layer)
        layer.weight = torch.nn.Parameter(
            torch.empty(shape, dtype=torch.float16, device="meta")
        )
        layer.quant_method = UnquantizedLinearMethod()
        return layer

    down, up = plain((336, 10240)), plain((10240, 320))
    x = torch.empty(1, 10240, dtype=torch.float16, device="meta")
    assert hc.maybe_apply_qwen38_sm70_fp16_fused_hc(down, up, x, True) is not None
    assert len(calls) == 1
    wrapped = torch.nn.Module()
    wrapped.weight, wrapped.quant_method = down.weight, down.quant_method
    wrapped.base_layer = down
    assert hc.maybe_apply_qwen38_sm70_fp16_fused_hc(wrapped, up, x, True) is None
    down.quant_method = object()
    assert hc.maybe_apply_qwen38_sm70_fp16_fused_hc(down, up, x, True) is None
    assert len(calls) == 1


def test_admitted_hc_selects_fused_native_output(monkeypatch):
    from vllm.distributed import parallel_state

    calls = []
    communicator = SimpleNamespace(
        can_sm70_qwen38_hc_batch=lambda x: True,
        sm70_qwen38_hc_batch=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setattr(hc, "_batch_runtime_ok", lambda *args: True)
    monkeypatch.setattr(
        parallel_state,
        "get_tp_group",
        lambda: SimpleNamespace(
            device_communicator=SimpleNamespace(ca_comm=communicator)
        ),
    )
    x = torch.empty(2, 10240, dtype=torch.float16)
    block, injection = hc._qwen38_sm70_fp16_fused_hc(
        x, x, x, x, x, concurrent_batch=True
    )
    assert calls[0][1] == {
        "round_down_partials": False,
        "cooperative": False,
        "full_unroll": False,
        "fused_chain": True,
    }
    assert calls[0][0][6] is block and calls[0][0][7] is injection
    assert block.shape == (2, 2560) and injection.shape == (2, 4)


@pytest.mark.parametrize(
    "role,shape",
    [
        ("mlp.gate", (512, 2560)),
        ("linear_attn.out_proj", (2560, 1536)),
    ],
)
@pytest.mark.parametrize("policy", ("normal", "disabled", "mtp", "ubatch", "invariant"))
def test_dense_loader_permission_respects_batch_contract(
    config, monkeypatch, role, shape, policy
):
    class Dense(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(shape, device="meta"))
            self.prefix = "model.layers.0." + role
            self.quant_method = UnquantizedLinearMethod()

    monkeypatch.setattr(gemv, "LinearBase", Dense)
    monkeypatch.setattr(gemv.current_platform, "is_device_capability", lambda _: True)
    monkeypatch.setenv("VLLM_SM70_QWEN38_FP16_GEMV", "1")
    monkeypatch.setenv("VLLM_SM70_QWEN4_EXP_ONLINE_QPN8", "0")
    monkeypatch.setenv(
        "VLLM_SM70_QWEN38_BATCH_FASTPATH", str(int(policy != "disabled"))
    )
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", str(int(policy == "invariant")))
    if policy == "mtp":
        config.speculative_config = SimpleNamespace(
            method="mtp", num_speculative_tokens=4
        )
    if policy == "ubatch":
        config.parallel_config.use_ubatching = True
    layer = Dense()
    gemv.enable_qwen38_sm70_fp16_gemv(layer, torch.float16, config)
    assert layer._sm70_qwen38_dense_batch == (policy == "normal")
