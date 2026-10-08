# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch
from test_gguf_lattice_transcode import source

from vllm import _custom_ops  # noqa: F401
from vllm.model_executor.layers.quantization.gguf_lattice_staging import stage_lattice
from vllm.model_executor.layers.quantization.gguf_lattice_transcode import (
    transcode_lattice,
)
from vllm.model_executor.layers.quantization.gguf_raw import RawGGUFProjection

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("kind", [18, 21, 22])
@pytest.mark.parametrize("k", [256, 2560])
def test_stage_matches_packaged_converter_and_changed_input_graph(kind, k):
    n, experts = 64, 2
    group = 16 if kind == 22 else 32
    raw_rows, prepared = [], []
    for expert in range(experts):
        rows = source(kind, n=n, k=k, scale=0.001953125, seed=940 + expert)
        # Include sign changes, zero and coefficients that round to subnormals.
        block_bytes = {18: 98, 21: 110, 22: 82}[kind]
        blocks = rows.reshape(-1, block_bytes)
        scales = np.resize(
            np.array([0.0, -0.0, 2**-24, -0.001953125, 0.001953125], np.float16),
            len(blocks),
        )
        blocks[:, :2] = scales.view(np.uint8).reshape(-1, 2)
        projection = transcode_lattice(rows, kind)
        codes, metadata = projection.mma884_storage()
        metadata = metadata.view(np.int32 if kind == 22 else np.int64)
        w, s, ld = torch.ops._C.gguf_lattice_sm70_prepare(
            torch.from_numpy(codes).cuda(),
            torch.from_numpy(metadata).cuda(),
            kind,
            group,
        )
        assert ld.tolist() == [k * 32, n]
        prepared.append((w, s))
        raw_rows.append(RawGGUFProjection.from_rows(rows, kind).data)
    raw = torch.from_numpy(np.stack(raw_rows)).cuda()
    codes = torch.empty((experts, k, n // 16), dtype=torch.int32, device="cuda")
    stats = torch.empty(
        (experts, k // group, n),
        dtype=torch.int32 if kind == 22 else torch.int64,
        device="cuda",
    )
    stage_lattice(raw, codes, stats, kind)
    for expert, (w, s) in enumerate(prepared):
        torch.testing.assert_close(codes[expert], w, rtol=0, atol=0)
        torch.testing.assert_close(stats[expert], s, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        stage_lattice(raw, codes, stats, kind)
    # Same staging addresses must accept another layer's bank on every replay.
    raw.copy_(raw.flip(0))
    graph.replay()
    for expert, (w, s) in enumerate(reversed(prepared)):
        torch.testing.assert_close(codes[expert], w, rtol=0, atol=0)
        torch.testing.assert_close(stats[expert], s, rtol=0, atol=0)
