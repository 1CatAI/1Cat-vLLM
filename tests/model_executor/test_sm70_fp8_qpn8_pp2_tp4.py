# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm import envs
from vllm.model_executor.layers.quantization import fp8


def _layer(
    suffix: str,
    tp_size: int,
    k_dim: int,
    n_dim: int,
    *,
    output_partition_sizes: list[int] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        prefix=f"arbitrary.engine.graph.{suffix}",
        tp_size=tp_size,
        input_size_per_partition=k_dim,
        output_size_per_partition=n_dim,
        output_partition_sizes=output_partition_sizes,
        weight_block_size=[128, 128],
        weight=torch.empty((n_dim, k_dim), device="meta"),
    )


def test_pp2_tp4_qpn8_is_default_off_with_explicit_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_SM70_FP8_QPN8", raising=False)
    monkeypatch.delenv("VLLM_SM70_FP8_QPN8_PP2_TP4", raising=False)
    envs.disable_envs_cache()
    try:
        assert not fp8._sm70_fp8_qpn8_pp2_tp4_enabled()

        monkeypatch.setenv("VLLM_SM70_FP8_QPN8_PP2_TP4", "1")
        envs.disable_envs_cache()
        assert fp8._sm70_fp8_qpn8_pp2_tp4_enabled()

        monkeypatch.setenv("VLLM_SM70_FP8_QPN8", "0")
        envs.disable_envs_cache()
        assert not fp8._sm70_fp8_qpn8_pp2_tp4_enabled()

        monkeypatch.delenv("VLLM_SM70_FP8_QPN8")
        monkeypatch.setenv("VLLM_SM70_FP8_QPN8_PP2_TP4", "0")
        envs.disable_envs_cache()
        assert not fp8._sm70_fp8_qpn8_pp2_tp4_enabled()
    finally:
        envs.disable_envs_cache()


def test_qpn8_extension_load_requires_explicit_route(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_SM70_FP8_QPN8_LIBRARY", "/tmp/qpn8-test.so")
    monkeypatch.delenv("VLLM_SM70_FP8_QPN8", raising=False)
    monkeypatch.delenv("VLLM_SM70_FP8_QPN8_PP2_TP4", raising=False)
    with patch.object(torch.ops, "load_library") as load_library:
        fp8.sm70_ops._maybe_load_fp8_qpn8_library()
        load_library.assert_not_called()

        monkeypatch.setenv("VLLM_SM70_FP8_QPN8_PP2_TP4", "1")
        fp8.sm70_ops._maybe_load_fp8_qpn8_library()
        load_library.assert_called_once_with("/tmp/qpn8-test.so")

        load_library.reset_mock()
        monkeypatch.setenv("VLLM_SM70_FP8_QPN8", "0")
        fp8.sm70_ops._maybe_load_fp8_qpn8_library()
        load_library.assert_not_called()


@pytest.mark.parametrize(
    ("layer", "gated_silu", "expected"),
    [
        (_layer("fused_wqa_wkv", 1, 4096, 1536), False, (32, 2, False)),
        (_layer("wq_b", 4, 1024, 8192), False, (8, 2, False)),
        (_layer("wo_b", 4, 2048, 4096), False, (16, 2, False)),
        (_layer("down_proj", 4, 512, 4096), False, (16, 2, False)),
    ],
)
def test_pp2_tp4_qpn8_exact_operator_contracts(
    layer: SimpleNamespace,
    gated_silu: bool,
    expected: tuple[int, int, bool],
) -> None:
    assert fp8._sm70_fp8_qpn8_pp2_tp4_config(layer, gated_silu=gated_silu) == expected


def test_pp2_tp4_qpn8_rejects_wrong_tensor_and_concurrency_roles() -> None:
    wrong_tp = _layer("wq_b", 8, 1024, 8192)
    assert fp8._sm70_fp8_qpn8_pp2_tp4_config(wrong_tp, gated_silu=False) is None

    concurrent_indexer = _layer("wq_b", 1, 1024, 8192)
    assert (
        fp8._sm70_fp8_qpn8_pp2_tp4_config(concurrent_indexer, gated_silu=False) is None
    )

    excluded_gate = _layer(
        "gate_up_proj",
        4,
        4096,
        1024,
        output_partition_sizes=[512, 512],
    )
    assert fp8._sm70_fp8_qpn8_pp2_tp4_config(excluded_gate, gated_silu=False) is None
    assert fp8._sm70_fp8_qpn8_pp2_tp4_config(excluded_gate, gated_silu=True) is None

    wrong_layout = _layer("wo_b", 4, 2048, 4096)
    wrong_layout.weight_block_size = [64, 128]
    assert fp8._sm70_fp8_qpn8_pp2_tp4_config(wrong_layout, gated_silu=False) is None


def test_pp2_tp4_qpn8_shared_gate_requires_explicit_opt_in(monkeypatch) -> None:
    layer = _layer(
        "gate_up_proj",
        4,
        4096,
        1024,
        output_partition_sizes=[512, 512],
    )
    layer.prefix = "arbitrary.layers.7.mlp.shared_experts.gate_up_proj"

    monkeypatch.delenv("VLLM_SM70_FP8_QPN8_PP2_TP4_SHARED_GATE", raising=False)
    envs.disable_envs_cache()
    try:
        assert fp8._is_sm70_fp8_qpn8_pp2_tp4_shared_gate_contract(layer)
        assert fp8._sm70_fp8_qpn8_pp2_tp4_config(layer, gated_silu=False) is None

        monkeypatch.setenv("VLLM_SM70_FP8_QPN8_PP2_TP4_SHARED_GATE", "1")
        envs.disable_envs_cache()
        assert fp8._sm70_fp8_qpn8_pp2_tp4_config(layer, gated_silu=False) == (
            32,
            2,
            False,
        )
        assert fp8._sm70_fp8_qpn8_pp2_tp4_config(layer, gated_silu=True) is None

        layer.prefix = "arbitrary.layers.7.mlp.gate_up_proj"
        assert not fp8._is_sm70_fp8_qpn8_pp2_tp4_shared_gate_contract(layer)
        assert fp8._sm70_fp8_qpn8_pp2_tp4_config(layer, gated_silu=False) is None

        layer.prefix = "arbitrary.layers.7.mlp.shared_experts.gate_up_proj"
        monkeypatch.setenv("VLLM_SM70_FP8_QPN8_PP2_TP4_SHARED_GATE", "0")
        envs.disable_envs_cache()
        assert fp8._sm70_fp8_qpn8_pp2_tp4_config(layer, gated_silu=False) is None
    finally:
        envs.disable_envs_cache()


def test_pp2_tp4_qpn8_runtime_and_workspace_contract() -> None:
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=2,
            tensor_parallel_size=4,
            enable_dbo=False,
            ubatch_size=0,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
        speculative_config=None,
    )
    with patch.object(fp8, "get_current_vllm_config", return_value=config):
        assert fp8._is_sm70_fp8_qpn8_pp2_tp4_runtime_contract()
        config.speculative_config = object()
        assert not fp8._is_sm70_fp8_qpn8_pp2_tp4_runtime_contract()
        config.speculative_config = None
        config.parallel_config.pipeline_parallel_size = 1
        assert not fp8._is_sm70_fp8_qpn8_pp2_tp4_runtime_contract()
        config.parallel_config.pipeline_parallel_size = 2
        config.scheduler_config.max_num_seqs = 2
        assert not fp8._is_sm70_fp8_qpn8_pp2_tp4_runtime_contract()
        config.scheduler_config.max_num_seqs = 1
        config.parallel_config.enable_dbo = True
        assert not fp8._is_sm70_fp8_qpn8_pp2_tp4_runtime_contract()
        config.parallel_config.enable_dbo = False
        config.parallel_config.ubatch_size = 2
        assert not fp8._is_sm70_fp8_qpn8_pp2_tp4_runtime_contract()

    assert fp8._SM70_FP8_QPN8_PP2_TP4_WORKSPACE_ELEMENTS * 2 == 16 * 1024 * 1024


def test_pp2_tp4_qpn8_grouped_dispatches_caller_groups() -> None:
    layer = _layer("wo_a", 4, 4096, 2048)
    layer.is_bmm = True
    layer.bmm_batch_size = 2
    assert fp8._sm70_fp8_qpn8_pp2_tp4_bmm_config(layer) == (32, 2, False)

    layer.sm70_fp8_turbomind = True
    layer.sm70_fp8_qpn8 = True
    layer.sm70_fp8_qpn8_bmm = True
    layer.sm70_fp8_bmm_groups = 2
    layer.sm70_fp8_bmm_output_size = 1024
    layer.sm70_fp8_qpn8_split_k = 32
    layer.sm70_fp8_qpn8_nacc = 2
    layer.sm70_fp8_qpn8_prefetch = False
    layer.sm70_fp8_prefill_exact_dense_workspace_ptr = 123
    layer.weight = torch.empty((2, 4096, 1024), device="meta")
    layer.weight_scale_inv = torch.empty((2, 256, 32), device="meta")

    calls: list[tuple[tuple[int, ...], int, int, bool, bool]] = []

    def fake_dispatch(
        out: torch.Tensor,
        workspace_ptr: int,
        input_: torch.Tensor,
        codes: torch.Tensor,
        scales: torch.Tensor,
        split_k: int,
        nacc: int,
        prefetch: bool,
        gated_silu: bool,
    ) -> None:
        del codes, scales
        calls.append(
            (tuple(input_.shape), workspace_ptr, split_k, prefetch, gated_silu)
        )
        out.fill_(len(calls))
        assert nacc == 2

    x = torch.zeros((1, 2, 4096), dtype=torch.float16)
    with patch.object(fp8.sm70_ops, "fp8_qpn8_dispatch_sm70_out", fake_dispatch):
        out = fp8.Fp8LinearMethod.apply(None, layer, x)

    assert calls == [
        ((1, 4096), 123, 32, False, False),
        ((1, 4096), 123, 32, False, False),
    ]
    assert out.shape == (1, 2, 1024)
    torch.testing.assert_close(out[:, 0], torch.ones_like(out[:, 0]))
    torch.testing.assert_close(out[:, 1], torch.full_like(out[:, 1], 2))


def test_pp2_tp4_qpn8_explicit_opt_in_prepares_matching_layer(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_SM70_FP8_QPN8", raising=False)
    monkeypatch.setenv("VLLM_SM70_FP8_QPN8_PP2_TP4", "1")
    envs.disable_envs_cache()
    layer = _layer("fused_wqa_wkv", 1, 4096, 1536)
    layer.orig_dtype = torch.float16
    layer.is_bmm = False
    layer.weight_scale_inv = torch.empty((12, 32), device="meta")
    method = fp8.Fp8LinearMethod.__new__(fp8.Fp8LinearMethod)
    method.use_marlin = False
    method.use_sm70_fp8_turbomind = True
    method.weight_block_size = [128, 128]
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=2,
            tensor_parallel_size=4,
            enable_dbo=False,
            ubatch_size=0,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
        speculative_config=None,
    )

    try:
        with (
            patch.object(fp8, "get_current_vllm_config", return_value=config),
            patch.object(fp8, "_missing_sm70_fp8_qpn8_ops", return_value=[]),
            patch.object(
                fp8,
                "_get_sm70_fp8_qpn8_pp2_tp4_workspace",
                return_value=torch.empty(1),
            ),
            patch.object(
                fp8,
                "process_fp8_weight_block_strategy",
                side_effect=lambda weight, scales: (weight, scales),
            ),
            patch.object(
                fp8.sm70_ops,
                "fp8_qpn8_prepare_sm70",
                return_value=(
                    torch.empty((4096, 1536), dtype=torch.uint8, device="meta"),
                    torch.empty((256, 48), dtype=torch.float16, device="meta"),
                ),
            ),
            patch.object(fp8, "replace_parameter"),
        ):
            method.process_weights_after_loading(layer)
    finally:
        envs.disable_envs_cache()

    assert layer.sm70_fp8_qpn8
    assert layer.sm70_fp8_qpn8_split_k == 32
    assert layer.sm70_fp8_qpn8_nacc == 2
    assert not layer.sm70_fp8_qpn8_prefetch


def test_pp2_tp4_qpn8_shared_gate_retains_external_activation(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_SM70_FP8_QPN8_PP2_TP4", "1")
    monkeypatch.setenv("VLLM_SM70_FP8_QPN8_PP2_TP4_SHARED_GATE", "1")
    monkeypatch.delenv("VLLM_SM70_FP8_QPN8", raising=False)
    envs.disable_envs_cache()
    layer = _layer(
        "gate_up_proj",
        4,
        4096,
        1024,
        output_partition_sizes=[512, 512],
    )
    layer.prefix = "arbitrary.layers.7.mlp.shared_experts.gate_up_proj"
    layer.orig_dtype = torch.float16
    layer.is_bmm = False
    layer.weight_scale_inv = torch.empty((8, 32), device="meta")
    method = fp8.Fp8LinearMethod.__new__(fp8.Fp8LinearMethod)
    method.use_marlin = False
    method.use_sm70_fp8_turbomind = True
    method.weight_block_size = [128, 128]
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=2,
            tensor_parallel_size=4,
            enable_dbo=False,
            ubatch_size=0,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
        speculative_config=None,
    )

    try:
        with (
            patch.object(fp8, "get_current_vllm_config", return_value=config),
            patch.object(fp8, "_missing_sm70_fp8_qpn8_ops", return_value=[]),
            patch.object(
                fp8,
                "_get_sm70_fp8_qpn8_pp2_tp4_workspace",
                return_value=torch.empty(1),
            ),
            patch.object(
                fp8,
                "process_fp8_weight_block_strategy",
                side_effect=lambda weight, scales: (weight, scales),
            ),
            patch.object(
                fp8.sm70_ops,
                "fp8_qpn8_prepare_sm70",
                return_value=(
                    torch.empty((4096, 1024), dtype=torch.uint8, device="meta"),
                    torch.empty((256, 32), dtype=torch.float16, device="meta"),
                ),
            ),
            patch.object(fp8, "replace_parameter"),
        ):
            method.process_weights_after_loading(layer)
    finally:
        envs.disable_envs_cache()

    assert layer.sm70_fp8_qpn8
    assert layer.sm70_fp8_qpn8_split_k == 32
    assert layer.sm70_fp8_qpn8_nacc == 2
    assert not layer.sm70_fp8_qpn8_prefetch
    assert not getattr(layer, "sm70_fp8_gated_silu", False)
