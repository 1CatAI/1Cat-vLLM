# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gguf
import numpy as np
import pytest
import torch

from vllm.models.qwen4_exp.nvidia.gguf_embedding import pinned_iq4nl_rows
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [16, 80, 320])
def test_pinned_iq4nl_official_values_and_changed_graph_ids(rows):
    generator = np.random.default_rng(730)
    raw = generator.integers(0, 256, (512, 5, 18), dtype=np.uint8)
    scales = generator.uniform(0, 0.02, (512, 5)).astype(np.float16)
    scales[0] = np.array([0, -0.0, 2**-24, -(2**-24), 2**-14], dtype=np.float16)
    scales[1] *= -1
    raw[:, :, :2] = scales.view(np.uint8).reshape(512, 5, 2)
    raw = raw.reshape(512, 90)
    host = torch.empty(raw.shape, dtype=torch.uint8, pin_memory=True)
    host.copy_(torch.from_numpy(raw))
    mapped = get_accelerator_view_from_cpu_tensor(host)
    book = torch.tensor(gguf.quants.IQ4_NL.kvalues, dtype=torch.float32, device="cuda")
    ids = torch.arange(rows, dtype=torch.int64, device="cuda")
    out = torch.empty((rows, 160), dtype=torch.float16, device="cuda")
    official = torch.from_numpy(
        gguf.quants.dequantize(raw, gguf.GGMLQuantizationType.IQ4_NL).astype(np.float16)
    ).cuda()

    def run():
        pinned_iq4nl_rows(mapped.data_ptr(), ids, book, out, 160)

    run()
    torch.testing.assert_close(out, official[ids], rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    ids.copy_(ids.flip(0) + 32)
    graph.replay()
    torch.testing.assert_close(out, official[ids], rtol=0, atol=0)
