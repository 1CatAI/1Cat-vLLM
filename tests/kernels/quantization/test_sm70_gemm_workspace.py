# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm._sm70.runtime import bind_native_runtime
from vllm.config import KernelConfig, set_current_vllm_config
from vllm.model_executor.layers.quantization.gguf_lut_transcode import transcode_lut4
from vllm.model_executor.layers.quantization.gguf_transcode import transcode_affine
from vllm.runtime_resources import release_runtime_resources
from vllm.transformers_utils.gguf_tensor_reader import quant_size

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize(
    "kind,n,k", [(12, 512, 2560), (14, 3328, 2560), (23, 320, 2560)]
)
def test_bounded_workspace_matches_fp32_and_changed_graphs(kind, n, k, tmp_path):
    import vllm._C  # noqa: F401
    import vllm._moe_C  # noqa: F401

    block, size = quant_size(kind)
    raw = np.random.default_rng(1167 + kind).integers(
        0, 256, (n * k // block, size), dtype=np.uint8
    )
    scale_offset = size - 2 if kind == 14 else 0
    raw[:, scale_offset : scale_offset + 2] = np.frombuffer(
        np.float16(0.001337).tobytes(), dtype=np.uint8
    )
    if kind == 12:
        raw[:, 2:4] = np.frombuffer(np.float16(0.000739).tobytes(), dtype=np.uint8)
    source = raw.reshape(n, -1)
    canonical = (
        transcode_lut4(source, kind) if kind == 23 else transcode_affine(source, kind)
    )
    codes, scales = (
        torch.from_numpy(array).cuda() for array in (canonical.codes, canonical.scales)
    )
    if kind == 23:
        weight, stats, meta = torch.ops._C.gguf_lut4_sm70_prepare(
            codes, scales, canonical.lut_id, canonical.group_size
        )
    else:
        weight, stats, meta = torch.ops._C.gguf_affine_sm70_prepare(
            codes,
            scales,
            torch.from_numpy(canonical.mins).cuda(),
            canonical.bits,
            canonical.group_size,
        )
    kl, sl = meta.tolist()
    decoded = torch.from_numpy(canonical.dequantize()).half().cuda()
    configs = [
        SimpleNamespace(
            kernel_config=KernelConfig(sm70_turbomind_workspace_bytes=budget)
        )
        for budget in (32 * 1024**2, 8 * 1024**2)
    ]
    owners = []
    for config in configs:
        with set_current_vllm_config(config):
            owners.append(bind_native_runtime())

    def run(output, x):
        if kind == 23:
            torch.ops._C.gguf_lut4_gemm_sm70_out(
                output, x, weight, stats, canonical.lut_id, kl, sl, canonical.group_size
            )
        else:
            torch.ops._C.gguf_affine_gemm_sm70_out(
                output, x, weight, stats, canonical.bits, kl, sl, canonical.group_size
            )

    for m in (1, 5, 20, 512):
        torch.manual_seed(1167 + m)
        x = (torch.randn(m, k, device="cuda") * 0.125).half()
        outputs = [
            torch.empty(m, n, device="cuda", dtype=torch.float16) for _ in owners
        ]
        original_x = x.clone()
        reference = x.float() @ decoded.float().T
        for owner, output in zip(owners, outputs):
            with owner.activate():
                cache_path = str(tmp_path / f"control-{kind}-{m}.cache")
                if owner is owners[1]:
                    torch.ops._C.sm70_gemm_import_cache(x, cache_path)
                run(output, x)
                torch.testing.assert_close(
                    output.float(), reference, rtol=0.003, atol=0.003
                )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run(output, x)
                x_changed = x * 0.5
                x.copy_(x_changed)
                graph.replay()
                captured = output.clone()
                run(output, x)
                torch.testing.assert_close(captured, output, rtol=0, atol=0)
                x.copy_(original_x)
                run(output, x)
                graph.replay()
                start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                for _ in range(40):
                    graph.replay()
                stop.record()
                stop.synchronize()
                print(
                    json.dumps(
                        {
                            "kind": kind,
                            "m": m,
                            "partials_bytes": configs[
                                owners.index(owner)
                            ].kernel_config.sm70_turbomind_workspace_bytes,
                            "graph_us": start.elapsed_time(stop) * 1000 / 40,
                            "relative_l2": float(
                                (output.float() - reference).norm() / reference.norm()
                            ),
                            "changed_input_graph_exact": True,
                        }
                    ),
                    flush=True,
                )
                del graph
                if owner is owners[0]:
                    torch.ops._C.sm70_gemm_export_cache(x, cache_path)
        error = (outputs[1].float() - outputs[0].float()).norm()
        assert error <= reference.norm() * 0.003 + 1e-6
    for owner, budget in zip(owners, (32 * 1024**2, 8 * 1024**2)):
        with owner.activate():
            count, actual_budget, total = torch.ops._C.sm70_gemm_workspace_storage()
            assert count >= 1 and actual_budget == budget
            assert total == count * (budget + 2 * 1024**2 + 4)
            print(
                json.dumps(
                    {
                        "streams": count,
                        "partials_bytes": budget,
                        "workspace_bytes": total,
                    }
                ),
                flush=True,
            )
            with pytest.raises(RuntimeError, match="cannot change"):
                torch.ops._C.sm70_gemm_configure_workspace(4 * 1024**2)
    for config in configs:
        release_runtime_resources(config)


@pytest.mark.parametrize(
    "kind,n,k,shared",
    [
        (8, 4096, 2560, False),
        (14, 4096, 2560, False),
        (14, 320, 2560, True),
        (8, 2560, 160, True),
    ],
)
def test_dense_scratch_bounds_restore_and_graph(kind, n, k, shared):
    import vllm._C  # noqa: F401
    import vllm._moe_C  # noqa: F401

    from vllm.model_executor.layers.quantization.gguf_dense_hmma import workspace
    from vllm.model_executor.layers.quantization.gguf_dense_hmma_formats import (
        decode,
        pack_device,
    )

    block, size = quant_size(kind)
    raw = np.random.default_rng(9683 + n).integers(
        0, 256, (n * k // block, size), dtype=np.uint8
    )
    at = size - 2 if kind == 14 else 0
    raw[:, at : at + 2] = np.frombuffer(np.float16(0.001337).tobytes(), dtype=np.uint8)
    raw = raw.reshape(n, -1)
    canonical = transcode_affine(raw, kind)
    control_w, control_s, meta = torch.ops._C.gguf_affine_sm70_prepare(
        torch.from_numpy(canonical.codes).cuda(),
        torch.from_numpy(canonical.scales).cuda(),
        torch.from_numpy(canonical.mins).cuda(),
        canonical.bits,
        canonical.group_size,
    )
    fmt, q, scale, minimum, group = decode(raw, kind)
    payload = pack_device(fmt, q, scale, minimum, group, torch.device("cuda"))
    scratch = workspace(torch.device("cuda"), shared)
    assert scratch["weight"].numel() == (1 if shared else 10) * 1024**2
    assert scratch["stats"].numel() == (1 if shared else 5) * 1024**2
    weight = (
        scratch["weight"][: control_w.numel() * 4].view(torch.int32).view_as(control_w)
    )
    stats = (
        scratch["stats"][: control_s.nbytes].view(control_s.dtype).view_as(control_s)
    )
    torch.ops._C.gguf_dense_restore_canonical_sm70_out(
        weight, stats, *payload, fmt, k, n
    )
    torch.testing.assert_close(weight, control_w, rtol=0, atol=0)
    torch.testing.assert_close(stats, control_s, rtol=0, atol=0)
    config = SimpleNamespace(
        kernel_config=KernelConfig(sm70_turbomind_workspace_bytes=8 * 1024**2)
    )
    with set_current_vllm_config(config):
        owner = bind_native_runtime()
    kl, sl = meta.tolist()
    with owner.activate():
        for m in (5, 20, 32, 10, 512):
            x = (torch.randn(m, k, device="cuda") * 0.125).half()
            output = torch.empty(m, n, device="cuda", dtype=torch.float16)
            control = torch.empty_like(output)

            def run(output=output, x=x):
                torch.ops._C.gguf_dense_restore_canonical_sm70_out(
                    weight, stats, *payload, fmt, k, n
                )
                torch.ops._C.gguf_affine_gemm_sm70_out(
                    output, x, weight, stats, canonical.bits, kl, sl, group
                )

            run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            x.normal_(std=0.125)
            graph.replay()
            torch.ops._C.gguf_affine_gemm_sm70_out(
                control, x, control_w, control_s, canonical.bits, kl, sl, group
            )
            torch.testing.assert_close(output, control, rtol=0, atol=0)
            if m <= 32:
                split = 5 if k == 2560 and n <= 320 else 1
                torch.ops._C.gguf_dense_segments_sm70_out(
                    x,
                    [payload[0]],
                    [payload[1]],
                    [payload[2]],
                    [output],
                    [fmt],
                    [n],
                    k,
                    split,
                    8,
                    scratch["partial"],
                    scratch["counters"],
                    None,
                )
                reference = (
                    x.float()
                    @ torch.from_numpy(canonical.dequantize()).half().cuda().float().T
                )
                assert (output.float() - reference).norm() < reference.norm() * 0.003
            del graph
    release_runtime_resources(config)
