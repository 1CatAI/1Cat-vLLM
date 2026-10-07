# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch
import vllm._C  # noqa: F401

from vllm.model_executor.layers.quantization.gguf_raw import RawGGUFProjection
from vllm.transformers_utils.gguf_tensor_reader import quant_size

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("kind", [18, 21, 22])
@pytest.mark.parametrize("m", [1, 5, 20])
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_bank_aware_reader_matches_retained_dot_and_changed_graph(
    kind, m, quantized, index_dtype
):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    experts, n, k, top_k = 4, 64, 768, 2
    _, size = quant_size(kind)
    rng = np.random.default_rng(2048 + kind)
    original = []
    for _ in range(2):
        data = rng.integers(0, 256, (experts * n, k // 256, size), dtype=np.uint8)
        data[..., :2] = (
            rng.uniform(0.001, 0.005, (experts * n, k // 256, 1))
            .astype("<f2")
            .view(np.uint8)
        )
        data = data.reshape(experts * n, -1)
        raw = RawGGUFProjection.from_rows(data, kind)
        original.append(torch.from_numpy(raw.data.reshape(experts, n, -1)).cuda())
    torch.manual_seed(2048 + m)
    x = (torch.randn(m, k, device="cuda") * 0.125).half()
    ids = (torch.arange(m * top_k).reshape(m, top_k) % experts).to(
        device="cuda", dtype=index_dtype
    )
    q8 = torch.empty(m, k // 32, 36, dtype=torch.uint8, device="cuda")
    shape = (m, top_k, n // 32, 36) if quantized else (m, top_k, 2, n)
    dtype = torch.uint8 if quantized else torch.float16
    count = int(np.prod(shape))
    guard = torch.full((count + 64,), 7, dtype=dtype, device="cuda")
    out = guard[32:-32].view(shape)
    reference = torch.empty_like(out)

    def run():
        torch.ops._C.gguf_quantize_q8_1_sm70_out(q8, x)
        torch.ops._C.gguf_dp4a_gate_up_sm70_out(
            out, q8, ids, *original, kind, quantized, 16, True
        )
        torch.ops._C.gguf_dp4a_gate_up_sm70_out(
            reference, q8, ids, *original, kind, quantized
        )

    def check():
        torch.testing.assert_close(out, reference, rtol=0, atol=0)
        assert torch.all(guard[:32] == 7) and torch.all(guard[-32:] == 7)

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for shift in [1, 3]:
        x.copy_(torch.randn_like(x) * 0.125)
        ids.copy_((ids + shift) % experts)
        graph.replay()
        check()
