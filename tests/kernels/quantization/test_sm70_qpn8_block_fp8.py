# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-FP8 linears on SM70/SM75 through the native QPN8 operators."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.kernels.linear.scaled_mm.qpn8_blk import (
    QPN8Fp8BlockScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
)
from vllm.utils.torch_utils import current_stream

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((7, 0), (7, 5)),
    reason="requires an SM70 or SM75 GPU",
)


def _chunk_policy(enabled):
    from vllm.config.sm70_native import Sm70NativeConfig

    policy = Sm70NativeConfig(fp8_qpn8_m32_chunked=enabled)
    policy.resolve("fp8")
    return "sm70:1:" + "".join(f"{len(value)}:{value}" for value in policy.values)


def _native_block_policy():
    from vllm.config.sm70_native import Sm70NativeConfig

    policy = Sm70NativeConfig(fp8_qpn8_m32_chunked=True, fp8_qpn8_m32_native=True)
    policy.resolve("fp8")
    return "sm70:1:" + "".join(f"{len(value)}:{value}" for value in policy.values)


@pytest.mark.parametrize("rows", [9, 16, 17, 31, 32])
@pytest.mark.parametrize("gated", [False, True])
def test_block_dispatch_chunked_policy_and_graph(rows, gated):
    import vllm._C  # noqa: F401

    torch.manual_seed(297)
    k, n = 512, 256
    weight = (torch.randn(n, k, device="cuda") * 8).to(torch.float8_e4m3fn)
    scales = torch.rand(n // 128, k // 128, device="cuda") * 0.01 + 0.002
    codes, packed = torch.ops._C.fp8_qpn8_prepare_sm70(weight, scales)
    x = torch.empty(rows, k, device="cuda", dtype=torch.float16)
    out = x.new_empty(rows, n // 2 if gated else n)
    reference = torch.empty_like(out)
    workspace = x.new_full((k, n), float("nan"))
    token = _chunk_policy(True)
    small = (
        torch.ops._C.fp8_qpn8_gated_pair_sm70_out
        if gated
        else torch.ops._C.fp8_qpn8_gemm_sm70_out
    )

    def call():
        torch.ops._C.fp8_qpn8_dispatch_sm70_out(
            out,
            workspace.data_ptr(),
            x,
            codes,
            packed,
            8,
            2,
            False,
            gated,
            native_policy=token,
        )

    x.normal_(0, 0.1)
    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for amplitude in (0.1, 0.3, 0.0):
        x.normal_(0, amplitude)
        for start in range(0, rows, 8):
            small(
                reference[start : start + 8],
                x[start : start + 8],
                codes,
                packed,
                8,
                2,
                True,
                False,
            )
        for invoke in (call, graph.replay):
            out.fill_(float("nan"))
            invoke()
            assert torch.isfinite(out).all()
            # Chunking preserves the established small-row reduction order.
            assert torch.equal(out.view(torch.int16), reference.view(torch.int16))
            # Admission must avoid full-matrix reconstruction, including replay.
            assert workspace.isnan().all()


@pytest.mark.parametrize("rows", [17, 24, 31, 32, 33, 40])
@pytest.mark.parametrize("gated", [False, True])
def test_block_dispatch_native_policy_and_graph(rows, gated):
    import vllm._C  # noqa: F401

    torch.manual_seed(297)
    k, n = 512, 256
    weight = (torch.randn(n, k, device="cuda") * 8).to(torch.float8_e4m3fn)
    scales = torch.rand(n // 128, k // 128, device="cuda") * 0.01 + 0.002
    codes, packed = torch.ops._C.fp8_qpn8_prepare_sm70(weight, scales)
    x = torch.empty(rows, k, device="cuda", dtype=torch.float16)
    out = x.new_empty(rows, n // 2 if gated else n)
    reference = torch.empty_like(out)
    workspace = x.new_full((k, n), float("nan"))
    token = _native_block_policy()
    split = 8 if gated else 16
    small = (
        torch.ops._C.fp8_qpn8_gated_pair_sm70_out
        if gated
        else torch.ops._C.fp8_qpn8_gemm_sm70_out
    )

    def call():
        torch.ops._C.fp8_qpn8_dispatch_sm70_out(
            out,
            workspace.data_ptr(),
            x,
            codes,
            packed,
            split,
            2,
            False,
            gated,
            native_policy=token,
        )

    x.normal_(0, 0.1)
    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for amplitude in (0.1, 0.3, 0.0):
        x.normal_(0, amplitude)
        for start in range(0, rows, 8):
            small(
                reference[start : start + 8],
                x[start : start + 8],
                codes,
                packed,
                split,
                2,
                True,
                False,
            )
        for invoke in (call, graph.replay):
            out.fill_(float("nan"))
            invoke()
            assert torch.isfinite(out).all()
            # Chunking preserves the established small-row reduction order.
            assert torch.equal(out.view(torch.int16), reference.view(torch.int16))
            # Admission must avoid full-matrix reconstruction, including replay.
            assert workspace.isnan().all()


@pytest.mark.parametrize(
    "enabled,rows,split,chains,gated",
    [
        (None, 9, 8, 2, False),
        (False, 32, 8, 2, True),
        (True, 33, 8, 2, False),
        (True, 17, 3, 2, False),
        (True, 17, 12, 1, False),
        (True, 17, 32, 2, True),
    ],
)
def test_block_dispatch_retains_prefill_fallback(enabled, rows, split, chains, gated):
    import vllm._C  # noqa: F401

    layer, _ = _layer(256, 512)
    codes, packed = torch.ops._C.fp8_qpn8_prepare_sm70(
        layer.weight, layer.weight_scale_inv
    )
    x = torch.randn(rows, 512, device="cuda", dtype=torch.float16) * 0.1
    out = x.new_empty(rows, 128 if gated else 256)
    reference = torch.empty_like(out)
    workspace = x.new_full((512, 256), float("nan"))
    torch.ops._C.fp8_qpn8_prefill_sm70_out(
        reference, workspace.data_ptr(), x, codes, packed, gated
    )
    workspace.fill_(float("nan"))
    torch.ops._C.fp8_qpn8_dispatch_sm70_out(
        out,
        workspace.data_ptr(),
        x,
        codes,
        packed,
        split,
        chains,
        False,
        gated,
        native_policy=_chunk_policy(enabled),
    )
    assert torch.isfinite(workspace).all()
    torch.testing.assert_close(out, reference, rtol=0, atol=0)


def _config(n: int, k: int) -> FP8ScaledMMLinearLayerConfig:
    return FP8ScaledMMLinearLayerConfig(
        weight_quant_key=kFp8Static128BlockSym,
        activation_quant_key=kFp8Dynamic128Sym,
        weight_shape=(n, k),
        input_dtype=torch.float16,
        out_dtype=torch.float16,
    )


def _layer(n: int, k: int, scale_dtype=torch.float32):
    torch.manual_seed(n + k)
    raw = torch.randint(0, 256, (n, k), dtype=torch.uint8, device="cuda")
    # 0x7F/0xFF are e4m3 NaN codes; a checkpoint never stores them.
    raw[raw == 0x7F] = 0x7E
    raw[raw == 0xFF] = 0xFE
    weight = raw.view(torch.float8_e4m3fn)
    scales = (2.0 ** (torch.rand(n // 128, k // 128, device="cuda") * 3 - 12)).float()
    scales = scales.to(scale_dtype)
    full = scales.float().repeat_interleave(128, 0).repeat_interleave(128, 1)
    reference = weight.float() * full
    layer = torch.nn.Module()
    layer.prefix = f"test.{n}x{k}"
    layer.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
    layer.weight_scale_inv = torch.nn.Parameter(scales.clone(), requires_grad=False)
    return layer, reference


def test_admits_only_block_128_geometry():
    assert QPN8Fp8BlockScaledMMLinearKernel.can_implement(_config(1536, 4096))[0]
    assert not QPN8Fp8BlockScaledMMLinearKernel.can_implement(_config(1536, 4000))[0]
    assert not QPN8Fp8BlockScaledMMLinearKernel.can_implement(_config(1500, 4096))[0]


@pytest.mark.parametrize(
    ("n", "k"), [(128, 128), (768, 1408), (1536, 4096), (8192, 1024), (4096, 8192)]
)
@pytest.mark.parametrize("m", [0, 1, 6, 8, 9, 20, 64, 512])
@pytest.mark.parametrize("scale_dtype", [torch.float32, torch.float8_e8m0fnu])
def test_matches_dequantized_reference(
    default_vllm_config, n: int, k: int, m: int, scale_dtype
):
    layer, reference = _layer(n, k, scale_dtype)
    kernel = QPN8Fp8BlockScaledMMLinearKernel(_config(n, k))
    kernel.process_weights_after_loading(layer)
    # Raw checkpoint weights are released after packing.
    assert layer.weight.numel() == 0
    x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.1
    y = kernel.apply_weights(layer, x)
    expected = x.float() @ reference.t()
    assert y.shape == (m, n) and y.dtype == torch.float16
    if m:
        assert ((y.float() - expected).norm() / expected.norm()).item() < 1e-3


@pytest.mark.parametrize("dense_only", [False, True])
def test_concurrent_streams_keep_their_own_prefill_buffer(
    default_vllm_config, dense_only
):
    # DeepSeek-V4 runs the indexer's wq_b on an aux stream next to the main
    # wq_b; with one shared dense buffer the prefill of one overwrote the
    # other's dequantized weight.
    layer_a, reference_a = _layer(8192, 1024)
    layer_b, reference_b = _layer(4096, 1024)
    kernel_a = QPN8Fp8BlockScaledMMLinearKernel(_config(8192, 1024))
    kernel_b = QPN8Fp8BlockScaledMMLinearKernel(_config(4096, 1024))
    kernel_a.process_weights_after_loading(layer_a)
    kernel_b.process_weights_after_loading(layer_b)
    if dense_only:
        for layer in (layer_a, layer_b):
            for name in (
                "_sm70_block_fp8_turbomind_packed_weight",
                "_sm70_block_fp8_turbomind_packed_scales",
            ):
                if hasattr(layer, name):
                    delattr(layer, name)
    x = torch.randn(256, 1024, device="cuda", dtype=torch.float16) * 0.1
    aux = torch.cuda.Stream()
    # vLLM's stream, as the model code takes it: leaving the aux context below
    # restores it, while torch's default stream would stay recorded as vLLM's
    # current stream and break later graph captures in this process.
    main = current_stream()
    aux.wait_stream(main)
    for _ in range(20):
        with torch.cuda.stream(aux):
            y_b = kernel_b.apply_weights(layer_b, x)
        y_a = kernel_a.apply_weights(layer_a, x)
    main.wait_stream(aux)
    torch.accelerator.synchronize()
    for y, reference in ((y_a, reference_a), (y_b, reference_b)):
        expected = x.float() @ reference.t()
        assert ((y.float() - expected).norm() / expected.norm()).item() < 1e-3


@pytest.mark.parametrize("m", [9, 64])
def test_volta_medium_prefill_keeps_original_turbomind_output(default_vllm_config, m):
    if torch.cuda.get_device_capability()[0:2] not in ((7, 0), (7, 2)):
        pytest.skip("Volta retains the TurboMind fallback layout")

    layer, _ = _layer(1536, 4096)
    kernel = QPN8Fp8BlockScaledMMLinearKernel(_config(1536, 4096))
    kernel.process_weights_after_loading(layer)
    x = torch.randn(m, 4096, device="cuda", dtype=torch.float16)
    expected = torch.empty((m, 1536), device="cuda", dtype=torch.float16)
    # The kernel calls the original operator with its bound native policy.
    # TurboMind keeps one dispatch cache per policy, so the reference uses the
    # same binding instead of the process-wide legacy one.
    kernel.native_ops.fp8_gemm_sm70_out(
        expected,
        x,
        layer._sm70_block_fp8_turbomind_packed_weight,
        layer._sm70_block_fp8_turbomind_packed_scales,
        128,
        layer._qpn8_fallback_k_ld,
        layer._qpn8_fallback_q_ld,
        False,
    )
    torch.testing.assert_close(kernel.apply_weights(layer, x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("m", [9, 20, 40, 64, 512])
def test_volta_without_turbomind_copy_keeps_one_layout(default_vllm_config, m):
    # A 32 GiB V100 pipeline stage of DeepSeek-V4 runs out of memory when every
    # block FP8 weight is held twice; without the copy, rows beyond M=8 take
    # the dense prefill operator that Turing uses.
    default_vllm_config.kernel_config.sm70_fp8.block_qpn8_volta_turbomind_prefill = (
        False
    )
    layer, reference = _layer(1536, 4096)
    kernel = QPN8Fp8BlockScaledMMLinearKernel(_config(1536, 4096))
    kernel.process_weights_after_loading(layer)
    assert not hasattr(layer, "_sm70_block_fp8_turbomind_packed_weight")
    assert not hasattr(layer, "_qpn8_fallback_k_ld")
    x = torch.randn(m, 4096, device="cuda", dtype=torch.float16) * 0.1
    y = kernel.apply_weights(layer, x)
    expected = x.float() @ reference.t()
    assert ((y.float() - expected).norm() / expected.norm()).item() < 1e-3


@pytest.mark.parametrize("turbomind_copy", [True, False])
def test_volta_warmup_follows_the_held_layout(default_vllm_config, turbomind_copy):
    # The coordinated Volta warmup used to read TurboMind attributes that block
    # QPN8 layers never carry and failed the engine start.
    if torch.cuda.get_device_capability()[0:2] not in ((7, 0), (7, 2)):
        pytest.skip("the coordinated dense warmup runs on Volta only")
    from vllm.model_executor.warmup.awq_sm70_warmup import (
        _iter_unique_fp8_dense_layers,
        _warmup_fp8_dense_layers,
    )

    default_vllm_config.kernel_config.sm70_fp8.block_qpn8_volta_turbomind_prefill = (
        turbomind_copy
    )
    layer, _ = _layer(1536, 4096)
    QPN8Fp8BlockScaledMMLinearKernel(_config(1536, 4096)).process_weights_after_loading(
        layer
    )
    model = torch.nn.Module()
    model.proj = layer
    layers = list(_iter_unique_fp8_dense_layers(model))
    assert layers == ([(layer, False)] if turbomind_copy else [])
    # Only rows beyond the native QPN8 bound use the TurboMind copy.
    assert _warmup_fp8_dense_layers(layers, [1, 8, 9, 64]) == (
        2 if turbomind_copy else 0
    )


@pytest.mark.parametrize(
    ("max_num_seqs", "speculative_tokens", "explicit", "copy"),
    [
        (1, 5, None, False),  # one request verifying six rows stays on QPN8
        (8, 0, None, False),
        (9, 0, None, True),
        (2, 5, None, True),  # twelve decode rows reach the TurboMind copy
        (1, 5, True, True),
        (4, 5, False, False),
    ],
)
def test_volta_turbomind_copy_follows_decode_rows(
    default_vllm_config, max_num_seqs, speculative_tokens, explicit, copy
):
    if torch.cuda.get_device_capability()[0:2] not in ((7, 0), (7, 2)):
        pytest.skip("only Volta holds the TurboMind copy")
    default_vllm_config.scheduler_config.max_num_seqs = max_num_seqs
    if speculative_tokens:
        default_vllm_config.speculative_config = SimpleNamespace(
            num_speculative_tokens=speculative_tokens
        )
    default_vllm_config.kernel_config.sm70_fp8.block_qpn8_volta_turbomind_prefill = (
        explicit
    )
    layer, _ = _layer(1536, 4096)
    QPN8Fp8BlockScaledMMLinearKernel(_config(1536, 4096)).process_weights_after_loading(
        layer
    )
    assert hasattr(layer, "_sm70_block_fp8_turbomind_packed_weight") == copy


@pytest.mark.parametrize("m", [0, 1, 5, 8, 9, 20, 40, 64])
@pytest.mark.parametrize("turbomind_copy", [True, False])
def test_block_qpn8_fused_gated_silu_matches_reference(
    default_vllm_config, m, turbomind_copy
):
    default_vllm_config.kernel_config.sm70_fp8.block_qpn8_volta_turbomind_prefill = (
        turbomind_copy
    )
    layer, reference = _layer(2048, 1024)
    layer.prefix = "model.layers.0.mlp.gate_up_proj"
    layer.output_partition_sizes = [1024, 1024]
    kernel = QPN8Fp8BlockScaledMMLinearKernel(_config(2048, 1024))
    kernel.process_weights_after_loading(layer)
    assert getattr(layer, "sm70_fp8_gated_silu", False)
    x = torch.randn(m, 1024, device="cuda", dtype=torch.float16) * 0.1
    y = kernel.apply_fused_silu_and_mul(layer, x)
    assert y is not None and y.shape == (m, 1024) and y.dtype == torch.float16
    if m:
        gate_up = x.float() @ reference.t()
        gate, up = gate_up.chunk(2, dim=-1)
        expected = torch.nn.functional.silu(gate) * up
        assert ((y.float() - expected).norm() / expected.norm()).item() < 2e-3


def test_sm70_block_native_and_ba_split_revision_ops_bound():
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 is required")
    assert hasattr(torch.ops._C, "fp8_qpn8_block_native_sm70_out")
    assert hasattr(torch.ops._C, "fp8_qpn8_ba_split_geometry_revision")
    assert torch.ops._C.fp8_qpn8_ba_split_geometry_revision() >= 1


@pytest.mark.parametrize("m", [20, 36, 40, 72, 128, 192])
def test_block_qpn8_row_blocks_and_cudagraph_reuse_prefill_workspace(
    default_vllm_config, m
):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 is required")
    default_vllm_config.kernel_config.sm70_fp8.block_qpn8_volta_turbomind_prefill = (
        False
    )
    layer, reference = _layer(2048, 1024)
    layer.prefix = "model.layers.0.mlp.gate_up_proj"
    layer.output_partition_sizes = [1024, 1024]
    kernel = QPN8Fp8BlockScaledMMLinearKernel(_config(2048, 1024))
    kernel.process_weights_after_loading(layer)
    assert not hasattr(layer, "_sm70_block_fp8_turbomind_packed_weight")

    x = torch.randn(m, 1024, device="cuda", dtype=torch.float16) * 0.1
    y_lin = kernel.apply_weights(layer, x)
    expected_lin = x.float() @ reference.t()
    assert ((y_lin.float() - expected_lin).norm() / expected_lin.norm()).item() < 1e-3

    y_gated = kernel.apply_fused_silu_and_mul(layer, x)
    assert y_gated is not None
    gate, up = expected_lin.chunk(2, dim=-1)
    expected_gated = torch.nn.functional.silu(gate) * up
    assert (
        (y_gated.float() - expected_gated).norm() / expected_gated.norm()
    ).item() < 2e-3

    # Verify CUDA graph capture of large-M prefill reuses the pre-allocated
    # primary dense weight buffer instead of allocating K*N inside graph_pool.
    if m > 128:
        from vllm.model_executor.kernels.linear.qpn.fp8_block import (
            _ensure_block_qpn8_prefill_workspace,
        )

        preallocated_ptr = _ensure_block_qpn8_prefill_workspace(
            x.device, 2048 * 1024
        ).data_ptr()
        pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            assert (
                _ensure_block_qpn8_prefill_workspace(x.device, 2048 * 1024).data_ptr()
                == preallocated_ptr
            )
            # First call initializes the cuBLAS stream workspace in graph_pool;
            # second call must allocate only the output tensor (no 4 MiB weight).
            _ = kernel.apply_weights(layer, x)
            before_second = torch.cuda.memory_allocated()
            captured = kernel.apply_weights(layer, x)
            after_second = torch.cuda.memory_allocated()
        assert after_second - before_second == captured.nbytes
        x.normal_(0, 0.1)
        graph.replay()
        torch.accelerator.synchronize()
        ref_replay = x.float() @ reference.t()
        assert (
            (captured.float() - ref_replay).norm() / ref_replay.norm()
        ).item() < 1e-3


@pytest.mark.parametrize(
    ("capture_sizes", "kv_bytes"), [([8], None), ([8, 1024], None), ([8], 1 << 30)]
)
def test_prefill_workspace_is_held_only_for_captured_prefill(
    default_vllm_config, capture_sizes, kv_bytes
):
    # The shared FP16 prefill workspace keeps captured layers from each holding
    # a K*N buffer in the graph pool. An engine whose graphs stop at 8 rows runs
    # every prefill eagerly; a workspace held for its whole life would come out
    # of the KV cache (GLM-5.3 on PP5: 128 MiB per stage, 39 322 tokens less).
    # A fixed KV byte budget plans no activation peak, so it keeps the workspace.
    from vllm.model_executor.kernels.linear.qpn.fp8 import clear_sm70_fp8_workspaces
    from vllm.model_executor.kernels.linear.qpn.fp8_block import (
        _block_qpn8_prefill_workspaces,
    )
    from vllm.model_executor.layers.quantization.utils import sm70_layer_workspaces

    default_vllm_config.kernel_config.sm70_fp8.block_qpn8_volta_turbomind_prefill = (
        False
    )
    default_vllm_config.compilation_config.cudagraph_capture_sizes = capture_sizes
    default_vllm_config.cache_config.kv_cache_memory_bytes = kv_bytes
    clear_sm70_fp8_workspaces()
    pool = sm70_layer_workspaces.workspace_pool(
        "_block_qpn8_prefill_workspaces", _block_qpn8_prefill_workspaces
    )
    primary = (torch.cuda.current_device(), -1)
    layer, reference = _layer(2048, 1024)
    kernel = QPN8Fp8BlockScaledMMLinearKernel(_config(2048, 1024))
    kernel.process_weights_after_loading(layer)
    resident = max(capture_sizes) > 8 or kv_bytes is not None
    if resident:
        assert pool[primary].numel() >= 2048 * 1024
    else:
        assert primary not in pool
    # 1024 rows take the FP16 prefill on SM70 and SM75 alike.
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16) * 0.1
    y = kernel.apply_weights(layer, x)
    expected = x.float() @ reference.t()
    assert ((y.float() - expected).norm() / expected.norm()).item() < 1e-3
    assert (primary in pool) == resident
    clear_sm70_fp8_workspaces()
