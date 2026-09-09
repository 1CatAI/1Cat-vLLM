# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP4 residual tests: launch the GPU cases with torchrun --nproc-per-node=4."""

import argparse
import os
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.minimax_h3.config import H3Config, H3InputError
from vllm.model_executor.models.minimax_h3.transformer import MiniMaxH3DiTBlock


@pytest.mark.parametrize("backend", ["FLASH_ATTN_V100", "FLASHINFER_SM70"])
def test_residual_parallel_accepts_native_sm70_backends(backend):
    config = H3Config(
        transformer_path="int8.safetensors",
        attention_backend=backend,
        residual_sequence_parallel=True,
    )
    assert config.residual_sequence_parallel


@pytest.mark.parametrize(
    "change",
    [
        {"tensor_parallel_size": 1},
        {"tensor_parallel_size": 2},
        {"attention_backend": "TORCH_SDPA"},
        {"partition": "ref2va"},
        {"transformer_path": None},
        {"lora_path": "adapter.safetensors"},
    ],
)
def test_residual_parallel_rejects_unvalidated_deployments(change):
    config = H3Config(
        transformer_path="int8.safetensors",
        attention_backend="FLASHINFER_SM70",
        residual_sequence_parallel=True,
    )
    with pytest.raises(H3InputError, match="residual sequence parallelism"):
        replace(config, **change)


@pytest.mark.parametrize("mode", ["generate", "serve"])
def test_residual_parallel_is_explicit_in_both_entrypoints(mode):
    from vllm.entrypoints.cli.video import VideoSubcommand

    parser = argparse.ArgumentParser()
    VideoSubcommand().subparser_init(parser.add_subparsers())
    assert not parser.parse_args(["video", mode]).residual_sequence_parallel
    assert parser.parse_args(
        ["video", mode, "--residual-sequence-parallel"]
    ).residual_sequence_parallel


@pytest.mark.parametrize(
    "rows,total,dtype,extra",
    [
        (4, 17, torch.float32, {}),
        (16, 16, torch.float32, {}),
        (4, 16, torch.float16, {}),
        (4, 16, torch.float32, {"num_requests": 2}),
        (4, 16, torch.float32, {"sp_seq_lens": [4] * 4}),
    ],
)
def test_residual_parallel_rejects_invalid_rows_before_collectives(
    rows, total, dtype, extra
):
    block = SimpleNamespace(
        residual_group=SimpleNamespace(world_size=4, rank_in_group=0)
    )
    with pytest.raises(ValueError, match="local FP32 rows"):
        MiniMaxH3DiTBlock.forward(
            block,
            torch.zeros(rows, 8, dtype=dtype),
            t_emb=torch.zeros(1, 8),
            combined_indices=torch.zeros(total, dtype=torch.long),
            rope_table=torch.zeros(total, 8),
            cu_seqlens=torch.tensor([0, total]),
            max_seqlen=total,
            packed_total=total,
            **extra,
        )


@pytest.fixture
def tp4_group():
    if os.environ.get("WORLD_SIZE") != "4":
        pytest.skip("requires torchrun --nproc-per-node=4 on a leased GPU group")
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(local_rank)
    torch.set_num_threads(2)
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=4))
    ):
        init_distributed_environment(4, rank, "env://", local_rank, "nccl")
        initialize_model_parallel(4)
        try:
            yield get_tp_group()
        finally:
            cleanup_dist_env_and_memory()


@torch.inference_mode()
def test_gpu_residual_parallel_matches_replicated_blocks(tp4_group):
    # The repository's autouse fixture tears down distributed state after each
    # test. Keep all collective shapes inside the same torchrun lifetime.
    for valid in (32, 33, 131):
        for backend in ("FLASH_ATTN_V100", "FLASHINFER_SM70"):
            _check_replicated_blocks(tp4_group, valid, backend)


def _check_replicated_blocks(group, valid, backend):
    from vllm.model_executor.models.minimax_h3.attention import attention_backend
    from vllm.model_executor.models.minimax_h3.transformer import (
        MiniMaxH3DiTArchConfig,
    )

    arch = MiniMaxH3DiTArchConfig(
        hidden_size=512,
        num_attention_heads=8,
        ffn_hidden_size=1024,
        adaln_curve_grid=2,
        adaln_out_features=18 * 512,
    )
    token = attention_backend.set(backend)
    try:
        baseline = MiniMaxH3DiTBlock(arch, None, prefix="blocks.0").cuda().eval()
        candidate = (
            MiniMaxH3DiTBlock(
                arch, None, prefix="blocks.0", residual_sequence_parallel=True
            )
            .cuda()
            .eval()
        )
    finally:
        attention_backend.reset(token)
    # Rank-dependent projection weights exercise real TP partial sums.
    torch.manual_seed(1234 + group.rank_in_group)
    for name, parameter in baseline.named_parameters():
        if "norm" in name:
            parameter.fill_(1)
        else:
            parameter.normal_(0, 0.03)
    candidate.load_state_dict(baseline.state_dict())
    torch.manual_seed(42)
    total = (valid + 3) // 4 * 4
    x = torch.randn(total, 512, device="cuda", dtype=torch.float32)
    x[::5, 0] = 70000  # Residuals must not pass through an FP16 collective.
    kwargs = dict(
        t_emb=torch.randn(1, 8, device="cuda"),
        combined_indices=torch.arange(total, device="cuda") % 3,
        rope_table=torch.randn(total, 96, device="cuda"),
        cu_seqlens=torch.tensor([0, valid], device="cuda", dtype=torch.int32),
        max_seqlen=valid,
        packed_total=total,
    )
    rows = total // 4
    actual = x.narrow(0, group.rank_in_group * rows, rows).clone()
    expected = x.clone()
    for _ in range(2):
        expected = baseline(expected, **kwargs)
        actual = candidate(actual, **kwargs)
    actual = group.all_gather(actual, dim=0)
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    assert actual.abs().max() > torch.finfo(torch.float16).max
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=3e-4)
    # Modifying padding must not change any valid attention output.
    if valid < total:
        changed = x.clone()
        changed[valid:] *= 10
        altered = candidate(
            changed.narrow(0, group.rank_in_group * rows, rows), **kwargs
        )
        original = candidate(x.narrow(0, group.rank_in_group * rows, rows), **kwargs)
        torch.testing.assert_close(
            group.all_gather(altered, dim=0)[:valid],
            group.all_gather(original, dim=0)[:valid],
            rtol=0,
            atol=0,
        )
