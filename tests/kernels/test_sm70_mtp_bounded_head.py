# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm._sm70.policy import NativeBindings
from vllm.model_executor.models import qwen3_5_mtp_bounded_head as bounded


@pytest.fixture(autouse=True)
def fp32_gemm_reduction_policy():
    matmul = torch.backends.cuda.matmul
    reduced = matmul.allow_fp16_reduced_precision_reduction
    accumulate = getattr(matmul, "allow_fp16_accumulation", None)
    matmul.allow_fp16_reduced_precision_reduction = False
    if accumulate is not None:
        matmul.allow_fp16_accumulation = False
    yield
    matmul.allow_fp16_reduced_precision_reduction = reduced
    if accumulate is not None:
        matmul.allow_fp16_accumulation = accumulate


def require_native():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    for name in ("sm70_fp16_masked_head_out", "sm70_fp16_candidate_scores_out"):
        if not hasattr(torch.ops._C, name):
            pytest.skip(f"requires {name}")


def test_bounded_target_cpu_fallback():
    head = torch.nn.Module()
    head.weight = torch.nn.Parameter(torch.zeros(1))
    config = SimpleNamespace(speculative_config=None)
    assert bounded.prepare_mtp_bounded_target_head(head, config) is None


def test_bounded_target_model_post_load_hook(monkeypatch):
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLMBase

    original = torch.nn.Module()
    prepared = torch.nn.Module()
    config = SimpleNamespace(
        kernel_config=SimpleNamespace(sm70_mtp_bounded_target_head=True)
    )
    model = SimpleNamespace(lm_head=original, vllm_config=config)
    calls = []

    def prepare(head, current_config):
        calls.append((head, current_config))
        return prepared

    monkeypatch.setattr(bounded, "prepare_mtp_bounded_target_head", prepare)
    Qwen3_5ForCausalLMBase.prepare_loaded_linear_weights(model)
    assert model._sm70_target_head is prepared
    assert calls == [(original, config)]
    config.kernel_config.sm70_mtp_bounded_target_head = False
    Qwen3_5ForCausalLMBase.prepare_loaded_linear_weights(model)
    assert model._sm70_target_head is None


def test_bounded_target_certificate_strictness():
    require_native()
    packet = torch.tensor(
        [
            [3, 100, 2, 3, 7, 2, 2, 200, 1, 1, 300, 0],
            [3, 100, 3, 2, 7, 2, 2, 200, 1, 1, 300, 0],
            [float("nan"), 100, 2, 3, 7, 2, 2, 200, 1, 1, 300, 0],
            [float("inf"), 100, 2, 3, 7, 2, 2, 200, 1, 1, 300, 0],
            [3, 100, float("inf"), 2, 7, 2, 2, 200, 1, 1, 300, 0],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    flags = torch.empty(5, device="cuda", dtype=torch.int32)
    pairs = torch.empty(5, 2, device="cuda", dtype=torch.float32)
    bounded._certify[(5,)](packet, flags, pairs, 4, num_warps=1)
    assert flags.tolist() == [1, 0, 0, 0, 0]
    assert pairs[0].tolist() == [3, 7]


@pytest.mark.parametrize("vocab", [65537, 124160])
def test_wide_dense_fallback_keeps_high_ids_and_all_tiles(vocab):
    require_native()
    from vllm.triton_utils import triton

    dense = torch.randn(3, vocab, device="cuda", dtype=torch.float16)
    flags = torch.zeros(3, device="cuda", dtype=torch.int32)
    pairs = torch.full((3, 2), 19.0, device="cuda", dtype=torch.float32)
    tiles = triton.cdiv(vocab, 1024)
    scratch = torch.empty(3, tiles, device="cuda", dtype=torch.int64)
    output = torch.empty_like(pairs)

    def run():
        bounded._masked_max_tiles[(3, tiles)](
            dense, flags, scratch, vocab, tiles, True, num_warps=4
        )
        bounded._masked_max_finish[(3,)](
            dense,
            flags,
            scratch,
            pairs,
            output,
            vocab,
            tiles,
            0,
            triton.next_power_of_2(tiles),
            True,
            num_warps=2,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for phase in ("high_id", "nan", "ties", "certified"):
        dense.normal_()
        dense[:, -1] = 100
        flags.zero_()
        if phase == "nan":
            dense[:, -32] = float("nan")
        elif phase == "ties":
            dense.zero_()
        elif phase == "certified":
            flags[1] = 1
        graph.replay()
        active = flags == 0
        ids = dense.argmax(-1)
        assert torch.equal(output[active, 1].long(), ids[active])
        torch.testing.assert_close(
            output[active, 0].half(),
            dense.gather(1, ids[:, None])[active, 0],
            rtol=0,
            atol=0,
            equal_nan=True,
        )
        assert torch.equal(output[~active], pairs[~active])


@pytest.mark.parametrize("rows", [1, 5, 8, 9, 17, 32, 33, 40, 64])
def test_masked_dense_head_graph_replay(rows):
    require_native()
    torch.manual_seed(49071)
    vocab = 124160 if rows in (1, 5, 8, 17) else 62080
    weight = torch.randn(vocab, 5120, device="cuda", dtype=torch.float16)
    x = torch.randn(rows, 5120, device="cuda", dtype=torch.float16)
    flags = torch.zeros(rows, device="cuda", dtype=torch.int32)
    scores = torch.empty(rows, vocab, device="cuda", dtype=torch.float16)
    operation = torch.ops._C.sm70_fp16_masked_head_out
    operation(scores, x, weight, flags)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation(scores, x, weight, flags)
    for cycle in range(16):
        x.normal_().mul_(0.1 if cycle < 8 else 2)
        flags.copy_((torch.arange(rows, device="cuda") + cycle) % 2)
        scores.fill_(111)
        graph.replay()
        reference = torch.nn.functional.linear(x, weight)
        active = flags == 0
        if vocab <= 65536 or rows > 8:
            assert torch.equal(
                scores[active].view(torch.int16), reference[active].view(torch.int16)
            )
        else:
            oracle = x[active].double() @ weight.double().T
            assert (
                scores[active].double() - oracle
            ).norm() <= 2**-11 * oracle.norm() + 2**-14
        assert bool((scores[~active] == 111).all())
    flags.fill_(1)
    scores.fill_(42)
    graph.replay()
    assert bool((scores == 42).all())


@pytest.mark.parametrize("vocab,ranks", [(62080, 4), (124160, 2)])
def test_bounded_target_dynamic_fallback(monkeypatch, vocab, ranks):
    require_native()
    torch.manual_seed(68450)
    weight = torch.randn(vocab, 5120, device="cuda", dtype=torch.float16) * 0.02
    codes, scales, coefficients = bounded._prepare_proposal(weight, NativeBindings())
    monkeypatch.setattr(
        bounded,
        "tensor_model_parallel_all_gather",
        lambda packet, dim: torch.cat([packet] * ranks, dim=dim),
    )
    x = weight[vocab - 33].expand(5, -1).contiguous()

    def run():
        return bounded._bounded_pair(x, weight, codes, scales, coefficients, 0)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pairs, flags, upper = run()
    for scenario in ("certified", "zero", "mixed", "nonfinite", "subnormal"):
        x.copy_(weight[vocab - 33].expand_as(x))
        if scenario == "zero":
            x.zero_()
        elif scenario == "mixed":
            x[1::2].zero_()
        elif scenario == "nonfinite":
            x[0, 15] = float("nan")
        elif scenario == "subnormal":
            x.mul_(0.00001)
        graph.replay()
        reference = torch.nn.functional.linear(x, weight)
        assert torch.equal(pairs[:, 1].long(), reference.argmax(-1))
        assert not bool((torch.isfinite(reference) & (reference > upper)).any())
        if scenario == "certified":
            assert bool((flags == 1).all())
        if scenario == "zero":
            assert bool((flags == 0).all())
