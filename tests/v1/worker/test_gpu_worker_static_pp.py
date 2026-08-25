# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import torch

import vllm.envs as envs
from vllm.config import CUDAGraphMode
from vllm.sequence import IntermediateTensors
from vllm.v1.worker import gpu_worker
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker


def _worker() -> SimpleNamespace:
    model_config = SimpleNamespace(
        dtype=torch.float16,
    )
    return SimpleNamespace(
        model_runner=SimpleNamespace(
            model=SimpleNamespace(
                make_empty_intermediate_tensors=lambda batch_size, dtype, device: (
                    IntermediateTensors(
                        {
                            "hidden_states": torch.empty(
                                (batch_size, 4, 4096), dtype=dtype, device=device
                            )
                        }
                    )
                )
            )
        ),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                pipeline_parallel_size=2,
                tensor_parallel_size=4,
                enable_dbo=False,
                ubatch_size=0,
            ),
            scheduler_config=SimpleNamespace(max_num_seqs=1),
            speculative_config=None,
            compilation_config=SimpleNamespace(
                cudagraph_mode=CUDAGraphMode.FULL,
                pass_config=SimpleNamespace(enable_sp=False),
            ),
            model_config=model_config,
        ),
    )


def test_static_pp_admission_uses_engine_and_tensor_geometry(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_SM70_PP_STATIC_HIDDEN_TRANSFER", "1")
    envs.disable_envs_cache()
    worker = _worker()
    try:
        with (
            patch.object(gpu_worker.current_platform, "is_cuda", return_value=True),
            patch.object(
                gpu_worker.current_platform,
                "is_device_capability",
                return_value=True,
            ),
        ):
            assert Worker._use_sm70_static_pp_hidden_transfer(worker, 1)
            assert not Worker._use_sm70_static_pp_hidden_transfer(worker, 2)

            wrong_schema = _worker()
            wrong_schema.model_runner.model.make_empty_intermediate_tensors = (
                lambda batch_size, dtype, device: IntermediateTensors(
                    {
                        "hidden_states": torch.empty(
                            (batch_size, 4096), dtype=dtype, device=device
                        )
                    }
                )
            )
            assert not Worker._use_sm70_static_pp_hidden_transfer(wrong_schema, 1)
    finally:
        envs.disable_envs_cache()


def test_static_pp_admission_rejects_unsafe_concurrency(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_SM70_PP_STATIC_HIDDEN_TRANSFER", "1")
    envs.disable_envs_cache()
    try:
        for field, value in (
            ("max_num_seqs", 2),
            ("pipeline_parallel_size", 1),
            ("tensor_parallel_size", 8),
            ("enable_dbo", True),
            ("ubatch_size", 2),
        ):
            worker = _worker()
            target = (
                worker.vllm_config.scheduler_config
                if field == "max_num_seqs"
                else worker.vllm_config.parallel_config
            )
            setattr(target, field, value)
            with (
                patch.object(gpu_worker.current_platform, "is_cuda", return_value=True),
                patch.object(
                    gpu_worker.current_platform,
                    "is_device_capability",
                    return_value=True,
                ),
            ):
                assert not Worker._use_sm70_static_pp_hidden_transfer(worker, 1)
    finally:
        envs.disable_envs_cache()


def test_static_pp_schema_is_exact() -> None:
    hidden_states = SimpleNamespace(
        shape=(1, 4, 4096),
        dtype=torch.float16,
        is_cuda=True,
        is_contiguous=lambda: True,
    )
    assert Worker._is_static_pp_hidden_tensor_dict(
        {"hidden_states": hidden_states},
        1,  # type: ignore[dict-item]
    )
    hidden_states.shape = (1, 4096)
    assert not Worker._is_static_pp_hidden_tensor_dict(
        {"hidden_states": hidden_states},
        1,  # type: ignore[dict-item]
    )


def test_static_pp_receive_buffer_skips_self_copy() -> None:
    hidden_states = torch.ones((4, 8), dtype=torch.float16)
    runner = SimpleNamespace(
        intermediate_tensors=IntermediateTensors({"hidden_states": hidden_states}),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=4)
        ),
    )
    source = IntermediateTensors({"hidden_states": hidden_states})
    version = hidden_states._version

    with patch(
        "vllm.v1.worker.gpu_model_runner.is_residual_scattered_for_sp",
        return_value=False,
    ):
        result = GPUModelRunner.sync_and_gather_intermediate_tensors(
            runner, 1, source, True
        )

    assert hidden_states._version == version
    assert result["hidden_states"].data_ptr() == hidden_states.data_ptr()
