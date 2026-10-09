# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the extracted stages to frozen main, including argument/view order."""

import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm import envs
from vllm.config.kernel import KernelConfig
from vllm.config.sm70_moe import Sm70MoEConfig, Sm70MoEFormatConfig
from vllm.model_executor.layers.fused_moe.sm70 import single_token, weight_codec
from vllm.model_executor.layers.quantization.sm70_moe_router import (
    select_single_token_plan,
)


@pytest.fixture
def should_do_global_cleanup_after_test():
    return False


@pytest.fixture(autouse=True)
def clean_aliases(monkeypatch):
    import os

    envs.disable_envs_cache()
    for name in os.environ:
        if name.startswith("VLLM_SM70_"):
            monkeypatch.delenv(name)
    yield
    envs.disable_envs_cache()


@pytest.mark.parametrize("family", ["awq", "fp8"])
def test_policy_precedence_and_snapshot(monkeypatch, family):
    monkeypatch.setenv("VLLM_SM70_MOE_SINGLE_TOKEN_INDEXED_STAGE_FASTPATH", "1")
    monkeypatch.setenv("VLLM_SM70_MOE_SINGLE_TOKEN_COMPACT_W13_FASTPATH", "1")
    a = Sm70MoEFormatConfig()
    a.resolve(family)
    assert a.single_token_w13 == ("compact", "indexed", "dense")
    monkeypatch.setenv("VLLM_SM70_MOE_SINGLE_TOKEN_COMPACT_W13_FASTPATH", "0")
    a.resolve(family)
    assert a.single_token_w13[0] == "compact"
    b = Sm70MoEFormatConfig(single_token_w13=("dense",), single_token_w2="dense")
    b.resolve(family)
    assert b.single_token_w13 == ("dense",) and b.single_token_w2 == "dense"
    assert b.sources["single_token_w13"] == "configuration"


def test_only_used_family_changes_hash():
    cfg = KernelConfig()
    baseline = cfg.compute_hash()
    cfg.sm70_moe.fp8.single_token_reduce = "weighted"
    assert cfg.compute_hash() == baseline
    cfg.sm70_moe.awq.resolve("awq")
    used = cfg.compute_hash()
    assert used != baseline
    cfg.sm70_moe.fp8.single_token_reduce = "unpermute"
    assert cfg.compute_hash() == used
    cfg.sm70_moe.awq.sources["batched"] = "different attribution"
    assert cfg.compute_hash() == used
    cfg.sm70_moe.awq.single_token_reduce = "unpermute"
    assert cfg.compute_hash() != used
    assert not Sm70MoEConfig().resolved


CASES = [
    (family, *case)
    for family in ("awq", "fp8")
    for case in itertools.product([False, True], repeat=5)
    if family == "awq" or not case[-1]
]


@pytest.mark.parametrize("family,compact,indexed13,indexed2,weighted,strict", CASES)
def test_frozen_single_token_calls(
    monkeypatch, family, compact, indexed13, indexed2, weighted, strict
):
    trace = []
    tensors = {}

    def tensor(name, shape, dtype=torch.float16):
        result = torch.zeros(shape, dtype=dtype)
        tensors[result.untyped_storage().data_ptr()] = name
        return result

    def describe(value):
        if isinstance(value, torch.Tensor):
            return (
                tensors[value.untyped_storage().data_ptr()],
                tuple(value.shape),
                tuple(value.stride()),
                value.storage_offset(),
            )
        return value

    def record(name):
        def call(*args):
            trace.append((name, tuple(map(describe, args))))
            # Stage markers verify that the output is the returned caller buffer.
            if isinstance(args[0], torch.Tensor):
                args[0].fill_(len(trace))

        return call

    class Native:
        def __init__(self):
            self.moe_unpermute = record("reduce")

        def __getattr__(self, name):
            return record(name)

    native = Native()
    monkeypatch.setattr(weight_codec, "ops", native)
    monkeypatch.setattr(torch.ops._C, "silu_and_mul", record("silu"), raising=False)
    monkeypatch.setattr(
        torch.ops._moe_C, "moe_unpermute", record("reduce"), raising=False
    )
    layer = SimpleNamespace(
        sm70_num_experts=4,
        sm70_w13_k_dim=8,
        sm70_w13_n_dim=16,
        sm70_w2_k_dim=8,
        sm70_w2_n_dim=8,
        sm70_hidden_logical_size=6,
    )
    for name in (
        "w13_strided_ptrs_w",
        "w13_strided_ptrs_s",
        "w2_strided_ptrs_w",
        "w2_strided_ptrs_s",
    ):
        setattr(layer, name, tensor(name, (4, 8), torch.uint8))
    buffers = {
        name: tensor(name, shape, dtype)
        for name, shape, dtype in (
            ("gate_up", (2, 16), torch.float16),
            ("intermediate", (2, 8), torch.float16),
            ("permuted_input", (2, 8), torch.float16),
            ("sorted_output", (2, 8), torch.float16),
            ("expert_offsets", (5,), torch.int32),
            ("expert_offsets64", (5,), torch.int64),
            ("inv_permuted_idx", (1, 2), torch.int32),
            ("sorted_expert_ids", (2,), torch.int32),
            ("compact_w13_ptrs_w", (2, 8), torch.uint8),
            ("compact_w13_ptrs_s", (2, 8), torch.uint8),
            ("output", (1, 6), torch.float16),
        )
    }
    x = tensor("x", (1, 6))
    ids = tensor("ids", (1, 2), torch.int32)
    weights = tensor("weights", (1, 2), torch.float32)

    def observe(layer, value, label):
        trace.append(("observe", label))
        return value

    def activation(layer, out, gate):
        torch.ops._C.silu_and_mul(out, gate)

    namespace = dict(
        torch=torch,
        sm70_ops=native,
        _log_runtime_route_once=lambda *args: None,
        _single_token_compact_w13_enabled=lambda: compact,
        _single_token_indexed_w13_enabled=lambda: indexed13,
        _single_token_indexed_w2_enabled=lambda: indexed2,
        _single_token_weighted_reduce_enabled=lambda: weighted,
        _dump_awq_moe_buffer=observe,
        _silu_and_mul_w13=activation,
    )
    frozen = json.loads(
        (Path(__file__).parent / "fixtures/sm70_single_token_legacy.json").read_text()
    )
    exec(
        compile(frozen["functions"][family], "<frozen-main-single-token>", "exec"),
        namespace,
    )
    expected = namespace["reference"](
        SimpleNamespace(group_size=32),
        layer,
        x,
        weights,
        ids,
        buffers,
        2,
        buffers["output"],
        strict,
        False,
    )
    expected_trace = list(trace)
    trace.clear()
    plan = select_single_token_plan(
        compact_w13=compact,
        indexed_w13=indexed13,
        indexed_w2=indexed2,
        weighted_reduce=weighted,
        strict=strict,
    )
    codec = weight_codec.Sm70MoEWeightCodec(
        family.upper(), SimpleNamespace(info_once=lambda *args: None)
    )
    actual = single_token.execute_single_token(
        codec,
        plan,
        layer,
        x,
        weights,
        ids,
        buffers,
        32,
        activation=activation if family == "awq" else None,
        observe=observe if family == "awq" else None,
        trim_output=family == "awq",
    )
    assert trace == expected_trace
    assert actual is expected is buffers["output"]


@pytest.mark.parametrize("family", ["awq", "fp8"])
def test_missing_compact_operator_keeps_indexed_fallback(monkeypatch, family):
    from vllm.model_executor.layers.quantization import awq_sm70_moe, fp8_sm70_moe

    module = awq_sm70_moe if family == "awq" else fp8_sm70_moe
    policy = Sm70MoEFormatConfig(
        single_token_w13=("compact", "indexed", "dense"),
        single_token_w2="indexed",
    )
    policy.resolve(family)
    monkeypatch.setattr(
        torch.ops,
        "_C",
        SimpleNamespace(
            **{
                f"{family}_moe_single_token_indexed_dense_w13_sm70_out": object(),
                f"{family}_moe_single_token_indexed_dense_stage_sm70_out": object(),
            }
        ),
    )
    plan = select_single_token_plan(
        compact_w13=module._single_token_compact_w13_enabled(policy),
        indexed_w13=module._single_token_indexed_w13_enabled(policy),
        indexed_w2=module._single_token_indexed_w2_enabled(policy),
        weighted_reduce=module._single_token_weighted_reduce_enabled(policy),
    )
    assert plan.w13 == plan.w2 == "indexed"
    assert not plan.weighted_reduce


def test_diagnostics_and_engine_isolation(monkeypatch):
    import os

    first = KernelConfig()
    first.sm70_moe.awq.resolve("awq")
    before = first.compute_hash()
    first.sm70_moe.awq.diagnostics.dump_buffers = True
    first.sm70_moe.awq.diagnostics.compare_dir = "a-different-diagnostic-directory"
    assert first.compute_hash() == before
    monkeypatch.setenv("VLLM_SM70_AWQ_MOE_BATCHED_GEMM", "0")
    environment = dict(os.environ)
    second = KernelConfig()
    second.sm70_moe.awq.resolve("awq")
    first.sm70_moe.awq.resolve("awq")
    assert first.sm70_moe.awq.batched and not second.sm70_moe.awq.batched
    assert first.compute_hash() != second.compute_hash()
    assert dict(os.environ) == environment
