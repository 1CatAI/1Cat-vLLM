# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(70),
    reason="Requires SM70",
)


@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("page", [1024, 2048])
@torch.inference_mode()
def test_direct_bmhd_preserves_window_and_overwrites_output(batch, page):
    from flash_attn_v100 import flash_attn_prefill_paged
    from flash_attn_v100.flash_attn_interface import flash_attn_v100_cuda as native

    assert hasattr(native, "dflash2_paged_bmhd_fwd")
    torch.manual_seed(123)
    q = torch.randn(batch, 8, 8, 128, device="cuda", dtype=torch.float16)
    pages = (8192 + 17 * (batch - 1) + page - 1) // page
    kc = torch.randn(pages * batch, page, 2, 128, device="cuda", dtype=q.dtype)
    vc = torch.randn_like(kc)
    table = torch.stack(
        [torch.randperm(pages, device="cuda") + pages * b for b in range(batch)]
    ).int()
    lengths = torch.full((batch,), 1024, device="cuda", dtype=torch.int32)
    output = torch.empty_like(q)
    flash_attn_prefill_paged(
        q,
        kc,
        vc,
        table,
        lengths,
        out=output,
        causal=False,
        window_size=(2047, 2047),
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flash_attn_prefill_paged(
            q,
            kc,
            vc,
            table,
            lengths,
            out=output,
            causal=False,
            window_size=(2047, 2047),
        )
    for length in (0, 8, 1024, 2047, 2048, 2055, 8192):
        lengths.copy_(torch.arange(batch, device="cuda") * 17 + length)
        if length == 0:
            lengths.zero_()
        legacy = (
            native.prefill_paged_fwd(
                q.permute(0, 2, 1, 3).contiguous(),
                kc,
                vc,
                None,
                table,
                lengths,
                128**-0.5,
                "auto",
                1.0,
                1.0,
                False,
                2047,
                2047,
                None,
                0,
            )
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        output.fill_(float("nan"))
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(output, legacy)
        assert torch.isfinite(output).all()
