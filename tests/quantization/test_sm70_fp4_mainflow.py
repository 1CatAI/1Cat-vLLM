# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare real adapters to frozen main at the public native-call boundary."""

import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from vllm import envs
from vllm.config.kernel import KernelConfig
from vllm.config.sm70_moe import (
    MXFP4_ALIASES,
    NVFP4_ALIASES,
    Sm70MxFp4MoEConfig,
    Sm70NvFp4MoEConfig,
)
from vllm.model_executor.layers.fused_moe.sm70 import fp4_codec, fp4_stages
from vllm.model_executor.layers.quantization import mxfp4_sm70_moe, nvfp4_sm70_moe
from vllm.model_executor.layers.quantization.utils.sm70_layer_workspaces import (
    LayerWorkspaceView,
)


@pytest.fixture
def should_do_global_cleanup_after_test():
    return False


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    envs.disable_envs_cache()
    for name in os.environ:
        if name.startswith("VLLM_SM70_"):
            monkeypatch.delenv(name)
    yield
    envs.disable_envs_cache()


class CudaInput(torch.Tensor):
    @property
    def is_cuda(self):
        return True


NV_ROUTES: dict[str, dict[str, bool | int]] = {
    "dense": {},
    "qpn_m1": {"qpn_m1": True},
    "qpn_batch": {"qpn_batch": True},
    "qpn_dynamic": {"qpn_batch": True, "qpn_dynamic": True},
    "mtp": {"qpn_mtp5": True},
    "mtp_batch_overlap": {"qpn_mtp5": True, "qpn_batch": True, "qpn_dynamic": True},
    "fused_m1": {
        "qpn_m1": True,
        "fused_swiglu_prefill": True,
        "w2_direct_reduce": True,
    },
    "fused_batch": {"qpn_batch": True, "fused_batch_w13": True, "fused_batch_w2": True},
    "raw_batch": {"qpn_batch": True, "raw_scale": True},
    "raw_fused": {
        "qpn_batch": True,
        "raw_scale": True,
        "fused_batch_w13": True,
        "fused_batch_w2": True,
    },
    "indexed": {"grouped_prefill": True, "indexed_prefill": True},
    "indexed_fused": {
        "grouped_prefill": True,
        "indexed_prefill": True,
        "fused_swiglu_prefill": True,
    },
    "indexed_split": {
        "grouped_prefill": True,
        "indexed_prefill": True,
        "fused_swiglu_prefill": True,
        "fast_prefill": True,
    },
    "grouped_mtp": {"grouped_mtp5": True, "qpn_mtp5": True},
    "clamp": {},
}
MX_ROUTES: dict[str, dict[str, bool | int]] = {
    "dense": {},
    "direct": {"direct_top6": True},
    "direct_order": {"direct_top6": True, "direct_order": True},
    "qpn": {"direct_top6": True, "direct_order": True, "qpn_m1": True},
    "active": {"active_experts": True, "active_expert_max_tokens": 8},
    "grouped": {
        "active_experts": True,
        "active_expert_max_tokens": 8,
        "grouped_verifier": True,
    },
    "grouped_rows": {
        "active_experts": True,
        "active_expert_max_tokens": 8,
        "grouped_verifier": True,
        "grouped_expert_rows": True,
    },
    "conflict": {
        "active_experts": True,
        "active_expert_max_tokens": 8,
        "single_token_permute": True,
    },
}
CASES = [
    (family, route, m)
    for family, routes, widths in (
        ("nvfp4", NV_ROUTES, (0, 1, 2, 3, 4, 5, 8, 16, 17, 127, 128)),
        ("mxfp4", MX_ROUTES, (0, 1, 2, 5, 8, 9)),
    )
    for route in routes
    for m in widths
]


@pytest.mark.parametrize("family,route,m", CASES)
def test_frozen_apply_call_order(monkeypatch, family, route, m):
    module = nvfp4_sm70_moe if family == "nvfp4" else mxfp4_sm70_moe
    aliases = NVFP4_ALIASES if family == "nvfp4" else MXFP4_ALIASES
    cls = (
        module.ModelOptNvFp4SM70MoEMethod
        if family == "nvfp4"
        else module.Mxfp4SM70MoEMethod
    )
    config_cls = Sm70NvFp4MoEConfig if family == "nvfp4" else Sm70MxFp4MoEConfig
    values: dict[str, bool | int] = dict.fromkeys(aliases, False)
    values.update((NV_ROUTES if family == "nvfp4" else MX_ROUTES)[route])
    for field, name in aliases.items():
        monkeypatch.setenv(name, str(int(values[field])))
    policy = config_cls()
    policy.resolve()
    trace, names = [], {}

    def tensor(name, shape, dtype=torch.float16, sparse=False):
        t = (
            torch.zeros(1, dtype=dtype).expand(shape)
            if sparse
            else torch.zeros(shape, dtype=dtype)
        )
        names[t.untyped_storage().data_ptr()] = name
        return t

    def describe(value):
        if isinstance(value, torch.Tensor):
            return (
                names.get(value.untyped_storage().data_ptr(), "temporary"),
                tuple(value.shape),
                tuple(value.stride()),
                value.storage_offset(),
                str(value.dtype),
            )
        return value

    def record(name):
        def call(*args, **kwargs):
            trace.append((name, tuple(map(describe, args)), kwargs))

        return call

    class Native:
        def __getattr__(self, name):
            if name.startswith("has_"):
                return lambda: True
            return record(name)

    native = Native()
    monkeypatch.setattr(module, "sm70_ops", native)
    monkeypatch.setattr(module, "is_exact_sm70_cuda", lambda *a, **kw: True)
    monkeypatch.setattr(fp4_codec, "ops", native)
    monkeypatch.setattr(fp4_stages, "ops", native)
    monkeypatch.setattr(
        torch.ops._moe_C,
        "moe_permute_sort_workspace_size",
        lambda *a: 256,
        raising=False,
    )
    for owner, op in (
        (torch.ops._C, "silu_and_mul"),
        (torch.ops._C, "silu_and_mul_interleaved"),
        (torch.ops._C, "silu_and_mul_with_clamp"),
        (torch.ops._moe_C, "moe_unpermute"),
        (torch.ops._moe_C, "moe_permute_with_scratch"),
        (torch.ops._moe_C, "moe_permute_metadata_with_scratch"),
    ):
        monkeypatch.setattr(owner, op, record(op), raising=False)
    for helper in (
        "_prepare_single_token_slots",
        "_single_token_weighted_reduce",
        "_mtp_weighted_reduce",
        "_prepare_compact_expert_groups",
        "_prepare_compact_slot_groups",
        "_compact_mxfp4_active_experts",
    ):
        if hasattr(module, helper):
            monkeypatch.setattr(module, helper, record(helper))
        if hasattr(fp4_stages, helper):
            monkeypatch.setattr(fp4_stages, helper, record(helper))

    h, intermediate, e, k = (
        (2560, 160, 512, 10) if family == "nvfp4" else (4096, 512, 256, 6)
    )
    layer = torch.nn.Module()
    layer.moe_config = NS(tp_size=4)
    layer.sm70_moe_policy = policy
    layer.local_num_experts = layer.global_num_experts = e
    layer.expert_map = None
    layer.swiglu_limit = 7.0 if route == "clamp" or family == "mxfp4" else None
    layer.apply_router_weight_on_input = False
    for field, value in dict(
        hidden_size=h,
        intermediate_size=intermediate,
        num_experts=e,
        top_k=k,
        group_size=16 if family == "nvfp4" else 32,
        w13_k_dim=h,
        w13_n_dim=2 * intermediate,
        w2_k_dim=intermediate,
        w2_n_dim=h,
    ).items():
        setattr(layer, f"sm70_{family}_{field}", value)
    for stage, kd, nd in (("w13", h, 2 * intermediate), ("w2", intermediate, h)):
        for suffix, shape, dtype in (
            ("tm_weight", (e, kd, nd // 8), torch.int32),
            (
                "tm_scales",
                (e, kd // (16 if family == "nvfp4" else 32), nd),
                torch.float16,
            ),
            ("strided_ptrs_w", (e,), torch.int64),
            ("strided_ptrs_s", (e,), torch.int64),
            ("raw_scale_codes", (e,), torch.uint8),
            ("raw_global_scales", (e,), torch.float32),
        ):
            setattr(
                layer,
                stage + "_" + suffix,
                tensor(stage + "_" + suffix, shape, dtype, sparse=True),
            )
    for prefix in ("w13_head", "w13_tail"):
        for suffix in ("strided_ptrs_w", "strided_ptrs_s"):
            setattr(
                layer,
                prefix + "_" + suffix,
                tensor(prefix + "_" + suffix, (e,), torch.int64, sparse=True),
            )
    for f in (
        "indexed_prefill",
        "fused_swiglu_prefill",
        "fast_prefill",
        "raw_scale",
        "w2_direct_reduce",
    ):
        setattr(layer, "sm70_nvfp4_qwen38_" + f, values.get(f, False))
    layer.sm70_nvfp4_qwen38_fused_swiglu_decode = values.get(
        "fused_swiglu_prefill", False
    )
    layer.sm70_nvfp4_grouped_mtp5 = values.get("grouped_mtp5", False)
    layer.sm70_mxfp4_qpn_m1_available = True
    for f in ("rows", "experts", "sizes", "total"):
        setattr(layer, "_nvfp4_grouped_" + f, tensor("grouped_" + f, (1,), torch.int32))
    layer.sm70_fp4_codec = fp4_codec.Fp4MoECodec(
        family,
        LayerWorkspaceView(layer, ""),
        LayerWorkspaceView(layer, "sm70_" + family + "_"),
        values.get("raw_scale", False),
        layer.swiglu_limit,
    )
    method = object.__new__(cls)
    method.sm70_moe_policy = policy
    method._allocate_graph_safe_decode_buffers(layer)
    buffers = (
        method._get_buffers(layer, m, False)
        if family == "nvfp4"
        else method._get_buffers(layer, m)
    )
    for name, t in buffers.items():
        if isinstance(t, torch.Tensor):
            names[t.untyped_storage().data_ptr()] = "buffer." + name
    monkeypatch.setattr(method, "_get_buffers", lambda *a: buffers)
    x = tensor("x", (m, h)).as_subclass(CudaInput)
    ids = tensor("ids", (m, k), torch.int32)
    weights = tensor("weights", (m, k), torch.float32)
    source = json.loads(
        (Path(__file__).parent / "fixtures/sm70_fp4_stage_legacy.json").read_text()
    )[family]
    legacy = dict(vars(module), envs=envs, os=os)
    for code in source.values():
        exec(code, legacy)
    trace.clear()
    legacy["apply"](method, layer, x, weights, ids, None, None)
    before = list(trace)
    trace.clear()

    def no_configuration_reads():
        raise AssertionError("prepared execution attempted to resolve policy")

    capture = "capture_" + family + "_moe_config"
    monkeypatch.setattr(module, capture, no_configuration_reads)
    method.apply(layer, x, weights, ids, None, None)
    assert trace == before


@pytest.mark.parametrize(
    "config_cls,aliases",
    [(Sm70NvFp4MoEConfig, NVFP4_ALIASES), (Sm70MxFp4MoEConfig, MXFP4_ALIASES)],
)
def test_fp4_config_capture_and_explicit_precedence(monkeypatch, config_cls, aliases):
    monkeypatch.setenv(aliases["qpn_m1"], "1")
    first = config_cls(qpn_m1=False)
    first.resolve()
    assert not first.qpn_m1 and first.sources["qpn_m1"] == "configuration"
    second = config_cls()
    second.resolve()
    assert second.qpn_m1
    monkeypatch.setenv(aliases["qpn_m1"], "0")
    second.resolve()
    assert second.qpn_m1
    third = config_cls()
    third.resolve()
    assert not third.qpn_m1


def test_fp4_hash_ignores_other_formats_and_diagnostics():
    c = KernelConfig()
    baseline = c.compute_hash()
    c.sm70_moe.nvfp4.resolve()
    used = c.compute_hash()
    assert used != baseline
    c.sm70_moe.nvfp4.route_debug = not c.sm70_moe.nvfp4.route_debug
    c.sm70_moe.mxfp4.qpn_m1 = False
    assert c.compute_hash() == used
    c.sm70_moe.nvfp4.qpn_m1 = not c.sm70_moe.nvfp4.qpn_m1
    assert c.compute_hash() != used


@pytest.mark.parametrize("family", ["nvfp4", "mxfp4"])
def test_workspace_persistent_views_overflow_and_weight_rebinding(monkeypatch, family):
    from vllm.model_executor.layers.fused_moe.sm70.fp4_workspace import (
        MxFp4MoEWorkspace,
        NvFp4MoEWorkspace,
    )

    monkeypatch.setattr(
        torch.ops._moe_C,
        "moe_permute_sort_workspace_size",
        lambda *a: 64,
        raising=False,
    )
    layer = torch.nn.Module()
    layer.w13_tm_weight = torch.zeros(1)
    layer.local_num_experts = layer.global_num_experts = 4
    for key, value in dict(
        hidden_size=64, intermediate_size=32, num_experts=4, top_k=6, w13_n_dim=64
    ).items():
        setattr(layer, f"sm70_{family}_{key}", value)
    owner = NvFp4MoEWorkspace if family == "nvfp4" else MxFp4MoEWorkspace
    owner.allocate(layer)

    def get(m):
        return owner.get(layer, m, False) if family == "nvfp4" else owner.get(layer, m)

    one, eight = get(1), get(8)
    assert (
        one["gate_up"].untyped_storage().data_ptr()
        == eight["gate_up"].untyped_storage().data_ptr()
    )
    assert (
        one["output"].untyped_storage().data_ptr()
        == eight["output"].untyped_storage().data_ptr()
    )
    outside = get(19 if family == "nvfp4" else 9)
    assert (
        outside["output"].untyped_storage().data_ptr()
        != one["output"].untyped_storage().data_ptr()
    )
    # AOT reload/attribute replacement is observed by the codec's borrowed view.
    view = LayerWorkspaceView(layer, "")
    replacement = torch.ones(2)
    layer.w13_tm_weight = replacement
    assert view.w13_tm_weight is replacement
    if family == "mxfp4":
        slot_offsets = one["slot_expert_offsets"].clone()
        eight["compact_expert_offsets"].fill_(123)
        assert torch.equal(get(1)["slot_expert_offsets"], slot_offsets)


def test_explicit_typed_mxfp4_missing_operator_fails_closed(monkeypatch):
    monkeypatch.setattr(mxfp4_sm70_moe, "_mxfp4_qpn_m1_op_available", lambda: False)
    policy = Sm70MxFp4MoEConfig(qpn_m1=True)
    policy.resolve()
    with pytest.raises(RuntimeError, match="explicitly enabled"):
        mxfp4_sm70_moe._mxfp4_qpn_m1_extension_enabled(policy)


def test_two_engine_captures_do_not_share_policy(monkeypatch):
    import vllm.config
    from vllm.config.sm70_moe import capture_nvfp4_moe_config

    first, second = KernelConfig(), KernelConfig()
    first.sm70_moe.nvfp4.qpn_m1 = True
    second.sm70_moe.nvfp4.qpn_m1 = False
    monkeypatch.setattr(
        vllm.config, "get_current_vllm_config_or_none", lambda: NS(kernel_config=first)
    )
    a = capture_nvfp4_moe_config()
    monkeypatch.setattr(
        vllm.config, "get_current_vllm_config_or_none", lambda: NS(kernel_config=second)
    )
    b = capture_nvfp4_moe_config()
    assert a is not b and a.qpn_m1 and not b.qpn_m1


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_shared_gguf_skinny_reduction_preserves_rounding(dtype):
    from vllm.model_executor.layers.fused_moe.sm70.reduction import weighted_reduce_rows

    generator = torch.Generator().manual_seed(54633)
    rows = torch.randn(3, 6, 64, generator=generator).to(dtype)
    weights = torch.randn(3, 6, generator=generator)
    reference = (rows.float() * weights[..., None].float()).sum(1).to(dtype)
    assert torch.equal(weighted_reduce_rows(rows, weights, dtype), reference)


def test_native_owned_switch_cannot_silently_conflict(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_NVFP4_QWEN38_MOE_FAST_PREFILL", "0")
    with pytest.raises(ValueError, match="native ABI"):
        Sm70NvFp4MoEConfig(fast_prefill=True).resolve()
    allowed = Sm70NvFp4MoEConfig(fast_prefill=False)
    allowed.resolve()
    assert not allowed.fast_prefill
