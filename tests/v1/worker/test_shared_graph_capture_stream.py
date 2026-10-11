# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm._sm70.runtime import bind_native_runtime
from vllm.config import KernelConfig, set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import parallel_state as ps
from vllm.model_executor.layers.quantization.gguf_transcode import transcode_affine
from vllm.runtime_resources import release_runtime_resources
from vllm.v1.worker.gpu import cudagraph_utils as cg
from vllm.v1.worker.gpu.spec_decode.eagle import speculator as eagle

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires SM70",
)


@pytest.mark.parametrize("external", [False, True])
def test_eagle_prefill_decode_and_multistep_share_capture_context(
    external, monkeypatch
):
    class Manager:
        def __init__(self, config, device, mode, *args, **kwargs):
            self.pool = torch.cuda.graph_pool_handle()
            self.cudagraph_mode = mode
            self.capture_context = None
            self.device = device

        get_capture_context = cg.CudaGraphManager.get_capture_context

    monkeypatch.setattr(eagle, "PrefillEagleCudaGraphManager", Manager)
    monkeypatch.setattr(eagle, "DecodeEagleCudaGraphManager", Manager)
    speculator = eagle.EagleSpeculator.__new__(eagle.EagleSpeculator)
    speculator.device = torch.device("cuda")
    speculator.num_speculative_steps = 4
    speculator.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE
        ),
        kernel_config=SimpleNamespace(capture_all_draft_steps=True),
    )
    context = ps.GraphCaptureContext(torch.cuda.Stream()) if external else None
    speculator.init_cudagraph_manager(
        CUDAGraphMode.FULL_AND_PIECEWISE, capture_context=context
    )
    managers = (
        speculator.prefill_cudagraph_manager,
        speculator.decode_cudagraph_manager,
        speculator.multistep_cudagraph_manager,
    )
    actual = managers[0].get_capture_context()
    assert all(m.capture_context is actual for m in managers)
    assert all(m.pool == managers[0].pool for m in managers)
    if external:
        assert actual is context


@pytest.mark.parametrize("tokens", [5, 20])
def test_serial_graphs_share_library_scratch_without_sharing_output_pools(
    tokens, monkeypatch
):
    import vllm._C  # noqa: F401

    group = ps.GroupCoordinator.__new__(ps.GroupCoordinator)
    group.device_communicator = None
    monkeypatch.setattr(ps, "get_tp_group", lambda: group)
    monkeypatch.setattr(ps, "get_pp_group", lambda: group)
    monkeypatch.setattr(cg, "is_global_first_rank", lambda: False)
    monkeypatch.setattr(
        cg, "graph_policy", lambda: SimpleNamespace(compile_graph=False)
    )
    monkeypatch.setattr(
        cg, "sm70_decode_graph_compilation", lambda enabled: nullcontext()
    )
    monkeypatch.setattr(
        cg,
        "get_offloader",
        lambda: SimpleNamespace(
            sync_prev_onload=lambda: None, join_after_forward=lambda: None
        ),
    )
    n, k = 512, 2560
    raw = np.random.default_rng(1167).integers(
        0, 256, (n * k // 256, 144), dtype=np.uint8
    )
    raw[:, :2] = np.frombuffer(np.float16(0.001337).tobytes(), dtype=np.uint8)
    raw[:, 2:4] = np.frombuffer(np.float16(0.000739).tobytes(), dtype=np.uint8)
    canonical = transcode_affine(raw.reshape(n, -1), 12)
    codes, scales, mins = (
        torch.from_numpy(a).cuda()
        for a in (canonical.codes, canonical.scales, canonical.mins)
    )
    weight, stats, meta = torch.ops._C.gguf_affine_sm70_prepare(
        codes, scales, mins, 4, 32
    )
    kl, sl = meta.tolist()
    x = torch.randn(tokens, k, device="cuda", dtype=torch.float16) * 0.125
    dense = torch.randn(640, k, device="cuda", dtype=torch.float16) * 0.01
    keep_alive = []
    measurements = []
    configs = []
    for shared in (False, True):
        config = SimpleNamespace(
            kernel_config=KernelConfig(sm70_turbomind_workspace_bytes=8 * 1024**2)
        )
        configs.append(config)
        with set_current_vllm_config(config):
            owner = bind_native_runtime()
        with owner.activate():
            reference = torch.empty(tokens, n, device="cuda", dtype=torch.float16)
            torch.ops._C.gguf_affine_gemm_sm70_out(
                reference, x, weight, stats, 4, kl, sl, 32
            )
            base_streams = torch.ops._C.sm70_gemm_workspace_storage()[0]
        desc = cg.BatchExecutionDescriptor(CUDAGraphMode.FULL, tokens, 1, tokens)
        managers: list[cg.CudaGraphManager] = []
        outputs: list[dict[str, torch.Tensor]] = []
        torch.accelerator.synchronize()
        allocated_before = torch.accelerator.memory_allocated()
        for index in range(4):
            manager = cg.CudaGraphManager.__new__(cg.CudaGraphManager)
            manager.device = x.device
            manager.pool = torch.cuda.graph_pool_handle()
            manager.capture_context = (
                managers[0].get_capture_context() if shared and managers else None
            )
            manager.graphs = {}
            manager._capture_descs = {CUDAGraphMode.FULL: [desc]}
            output: dict[str, torch.Tensor] = {}

            def forward(mode, output=output, index=index):
                output["dense"] = x @ dense.T + index
                output["quant"] = torch.empty(
                    tokens, n, device="cuda", dtype=torch.float16
                )
                torch.ops._C.gguf_affine_gemm_sm70_out(
                    output["quant"], x, weight, stats, 4, kl, sl, 32
                )

            with owner.activate():
                manager.capture(lambda batch, forward=forward: (forward, None))
            managers.append(manager)
            outputs.append(output)
        torch.accelerator.synchronize()
        with owner.activate():
            streams, partials, native_bytes = torch.ops._C.sm70_gemm_workspace_storage()
        assert streams - base_streams == (1 if shared else 4)
        assert len({m.pool for m in managers}) == 4
        for seed in (7, 11, 19):
            torch.manual_seed(seed)
            x.copy_(torch.randn_like(x) * 0.125)
            with owner.activate():
                reference = torch.empty(tokens, n, device="cuda", dtype=torch.float16)
                torch.ops._C.gguf_affine_gemm_sm70_out(
                    reference, x, weight, stats, 4, kl, sl, 32
                )
            for index, (manager, output) in enumerate(zip(managers, outputs)):
                manager.graphs[desc].replay()
                torch.accelerator.synchronize()
                torch.testing.assert_close(output["quant"], reference, rtol=0, atol=0)
                torch.testing.assert_close(
                    output["dense"], x @ dense.T + index, rtol=0, atol=0
                )
        row = dict(
            tokens=tokens,
            shared=shared,
            streams=streams,
            base_streams=base_streams,
            native_bytes=native_bytes,
            torch_delta_bytes=torch.accelerator.memory_allocated() - allocated_before,
            changed_input_graph_exact=True,
        )
        measurements.append(row)
        print(json.dumps(row), flush=True)
        keep_alive.append((managers, outputs))
    assert measurements[0]["native_bytes"] - measurements[1]["native_bytes"] == 3 * (
        10 * 1024**2 + 4
    )
    keep_alive.clear()
    torch.accelerator.synchronize()
    for config in configs:
        release_runtime_resources(config)
