# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from torch import nn

import vllm.envs as envs
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.model_executor.warmup import awq_sm70_warmup as warmup


def _grouped_fp8_layer() -> nn.Module:
    layer = nn.Module()
    layer.sm70_fp8_turbomind = True
    layer.sm70_fp8_bmm = True
    layer.sm70_fp8_bmm_output_size = 64
    layer.sm70_fp8_k_ld = 128
    layer.sm70_fp8_q_ld = 64
    layer.output_size_per_partition = 192
    layer.weight = nn.Parameter(
        torch.empty((3, 128, 64), dtype=torch.uint8), requires_grad=False
    )
    layer.weight_scale_inv = nn.Parameter(
        torch.empty((3, 1, 64), dtype=torch.float32), requires_grad=False
    )
    return layer


def test_fp8_grouped_bmm_decode_defaults_on_with_rollback(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_FP8_GROUPED_BMM_DECODE", raising=False)
    assert envs.VLLM_SM70_FP8_GROUPED_BMM_DECODE

    monkeypatch.setenv("VLLM_SM70_FP8_GROUPED_BMM_DECODE", "0")
    assert not envs.VLLM_SM70_FP8_GROUPED_BMM_DECODE


def test_fp8_warmup_discovers_grouped_bmm_by_per_group_shape():
    layer = _grouped_fp8_layer()
    model = nn.Sequential(layer)

    discovered = list(warmup._iter_unique_fp8_dense_layers(model))

    assert discovered == [(layer, False)]


def test_fp8_warmup_skips_static_qpn8_dispatch():
    layer = _grouped_fp8_layer()
    layer.sm70_fp8_qpn8 = True
    model = nn.Sequential(layer)

    assert list(warmup._iter_unique_fp8_dense_layers(model)) == []


def test_fp8_warmup_matches_grouped_bmm_runtime_slice(monkeypatch):
    layer = _grouped_fp8_layer()
    calls = []
    monkeypatch.setattr(torch.ops._C, "fp8_gemm_sm70_out_meta", object(), raising=False)

    def record_call(out, x, weight, scales, group_size, k_ld, q_ld, gated_silu):
        calls.append(
            SimpleNamespace(
                out_shape=tuple(out.shape),
                x_shape=tuple(x.shape),
                weight_shape=tuple(weight.shape),
                scale_shape=tuple(scales.shape),
                group_size=group_size,
                k_ld=k_ld,
                q_ld=q_ld,
                gated_silu=gated_silu,
            )
        )

    monkeypatch.setattr(warmup.sm70_ops, "fp8_gemm_sm70_out", record_call)

    count = warmup._warmup_fp8_dense_layers([(layer, False)], [1, 4])

    assert count == 2
    assert [call.out_shape for call in calls] == [(1, 64), (4, 64)]
    assert [call.x_shape for call in calls] == [(1, 128), (4, 128)]
    assert all(call.weight_shape == (128, 64) for call in calls)
    assert all(call.scale_shape == (1, 64) for call in calls)
    assert all(call.group_size == 128 for call in calls)
    assert all(call.k_ld == 128 and call.q_ld == 64 for call in calls)
    assert all(not call.gated_silu for call in calls)


def test_fp8_warmup_includes_one_launch_grouped_decode(monkeypatch):
    layer = _grouped_fp8_layer()
    layer.sm70_fp8_bmm_groups = 2
    layer.sm70_fp8_bmm_grouped_decode = True
    layer.weight = nn.Parameter(
        torch.empty((2, 128, 64), dtype=torch.uint8), requires_grad=False
    )
    layer.weight_scale_inv = nn.Parameter(
        torch.empty((2, 1, 64), dtype=torch.float16), requires_grad=False
    )
    layer.sm70_fp8_bmm_grouped_offsets = torch.arange(3, dtype=torch.int32)
    layer.sm70_fp8_bmm_grouped_ptrs_w = torch.empty(2, dtype=torch.int64)
    layer.sm70_fp8_bmm_grouped_ptrs_s = torch.empty(2, dtype=torch.int64)
    dense_calls = []
    grouped_calls = []
    monkeypatch.setattr(torch.ops._C, "fp8_gemm_sm70_out_meta", object(), raising=False)
    monkeypatch.setattr(
        warmup.sm70_ops,
        "fp8_gemm_sm70_out",
        lambda *args: dense_calls.append(args),
    )
    monkeypatch.setattr(
        warmup.sm70_ops,
        "fp8_moe_gemm_sm70_per_expert_dispatch_out",
        lambda *args: grouped_calls.append(args),
    )

    count = warmup._warmup_fp8_dense_layers([(layer, False)], [1, 4])

    assert count == 3
    assert len(dense_calls) == 2
    assert len(grouped_calls) == 1
    call = grouped_calls[0]
    assert tuple(call[0].shape) == (2, 64)
    assert tuple(call[1].shape) == (2, 128)
    assert call[2] is layer.sm70_fp8_bmm_grouped_offsets
    assert call[3] is layer.sm70_fp8_bmm_grouped_ptrs_w
    assert call[4] is layer.sm70_fp8_bmm_grouped_ptrs_s
    assert call[5:] == (2, 128, 64, 128, False)


def test_fp8_grouped_bmm_decode_uses_one_dispatch(monkeypatch):
    layer = _grouped_fp8_layer()
    layer.sm70_fp8_bmm_groups = 2
    layer.sm70_fp8_bmm_grouped_decode = True
    layer.sm70_fp8_bmm_grouped_offsets = torch.arange(3, dtype=torch.int32)
    layer.sm70_fp8_bmm_grouped_ptrs_w = torch.empty(2, dtype=torch.int64)
    layer.sm70_fp8_bmm_grouped_ptrs_s = torch.empty(2, dtype=torch.int64)
    layer.output_size_per_partition = 128
    calls = []

    def grouped(out, x, *args):
        calls.append((out, x, args))
        out.copy_(torch.arange(out.numel(), dtype=out.dtype).reshape_as(out))

    def fail_dense(*args):
        raise AssertionError("dense fallback used")

    monkeypatch.setattr(
        warmup.sm70_ops,
        "fp8_moe_gemm_sm70_per_expert_dispatch_out",
        grouped,
    )
    monkeypatch.setattr(
        warmup.sm70_ops,
        "fp8_gemm_sm70_out",
        fail_dense,
    )
    x = torch.empty((1, 2, 128), dtype=torch.float16)

    out = object.__new__(Fp8LinearMethod).apply(layer, x)

    assert tuple(out.shape) == (1, 2, 64)
    assert len(calls) == 1
    grouped_out, grouped_x, args = calls[0]
    assert tuple(grouped_out.shape) == (2, 64)
    assert tuple(grouped_x.shape) == (2, 128)
    assert args[0] is layer.sm70_fp8_bmm_grouped_offsets
    assert args[1] is layer.sm70_fp8_bmm_grouped_ptrs_w
    assert args[2] is layer.sm70_fp8_bmm_grouped_ptrs_s
    assert args[3:] == (2, 128, 64, 128, False)
    torch.testing.assert_close(
        out,
        torch.arange(128, dtype=torch.float16).reshape(1, 2, 64),
    )


def test_fp8_grouped_bmm_decode_retains_multirow_fallback(monkeypatch):
    layer = _grouped_fp8_layer()
    layer.sm70_fp8_bmm_groups = 2
    layer.sm70_fp8_bmm_grouped_decode = True
    layer.sm70_fp8_bmm_output_size = 64
    dense_calls = []

    def dense(out, x, *args):
        dense_calls.append((tuple(out.shape), tuple(x.shape), args))
        out.zero_()

    def fail_grouped(*args):
        raise AssertionError("grouped decode must remain batch-one only")

    monkeypatch.setattr(warmup.sm70_ops, "fp8_gemm_sm70_out", dense)
    monkeypatch.setattr(
        warmup.sm70_ops,
        "fp8_moe_gemm_sm70_per_expert_dispatch_out",
        fail_grouped,
    )
    x = torch.empty((2, 2, 128), dtype=torch.float16)

    out = object.__new__(Fp8LinearMethod).apply(layer, x)

    assert tuple(out.shape) == (2, 2, 64)
    assert [call[:2] for call in dense_calls] == [
        ((2, 64), (2, 128)),
        ((2, 64), (2, 128)),
    ]


def test_fp8_warmup_supports_modelopt_turbomind_layout(monkeypatch):
    layer = nn.Module()
    layer.sm70_modelopt_fp8_turbomind = True
    layer.sm70_modelopt_fp8_k_ld = 128
    layer.sm70_modelopt_fp8_q_ld = 64
    layer.output_size_per_partition = 64
    layer.weight = nn.Parameter(
        torch.empty((128, 64), dtype=torch.uint8), requires_grad=False
    )
    layer.weight_scale = nn.Parameter(
        torch.empty((1, 64), dtype=torch.float16), requires_grad=False
    )
    model = nn.Sequential(layer)
    calls = []
    monkeypatch.setattr(torch.ops._C, "fp8_gemm_sm70_out_meta", object(), raising=False)
    monkeypatch.setattr(
        warmup.sm70_ops,
        "fp8_gemm_sm70_out",
        lambda out, x, weight, scales, group_size, k_ld, q_ld, gated_silu: calls.append(
            (out.shape, x.shape, scales, group_size, k_ld, q_ld)
        ),
    )

    discovered = list(warmup._iter_unique_fp8_dense_layers(model))
    count = warmup._warmup_fp8_dense_layers(discovered, [1, 4])

    assert discovered == [(layer, False)]
    assert count == 2
    assert [tuple(call[0]) for call in calls] == [(1, 64), (4, 64)]
    assert all(call[2] is layer.weight_scale for call in calls)
    assert all(call[3:] == (128, 128, 64) for call in calls)


def _nvfp4_moe_layer() -> nn.Module:
    layer = nn.Module()
    layer.sm70_nvfp4_moe = True
    layer.moe_config = SimpleNamespace(experts_per_token=8)
    layer.swiglu_limit = None
    layer.sm70_nvfp4_num_experts = 256
    layer.sm70_nvfp4_w13_k_dim = 2048
    layer.sm70_nvfp4_w13_n_dim = 256
    layer.sm70_nvfp4_w2_k_dim = 128
    layer.sm70_nvfp4_w2_n_dim = 2048
    layer.sm70_nvfp4_group_size = 16
    layer.sm70_nvfp4_graph_safe_max_tokens = 18
    layer.sm70_nvfp4_compact_grouped_max_tokens = 10
    layer.w13_tm_weight = nn.Parameter(
        torch.empty((1, 1), dtype=torch.uint8), requires_grad=False
    )
    layer.w13_strided_ptrs_w = torch.empty(1, dtype=torch.uint8)
    layer.w13_strided_ptrs_s = torch.empty(1, dtype=torch.uint8)
    layer.w2_strided_ptrs_w = torch.empty(1, dtype=torch.uint8)
    layer.w2_strided_ptrs_s = torch.empty(1, dtype=torch.uint8)
    layer._nvfp4_sm70_dense_expert_ids = torch.arange(256, dtype=torch.int32)
    return layer


def test_nvfp4_moe_warmup_discovers_and_uses_compact_decode_shapes(monkeypatch):
    layer = _nvfp4_moe_layer()
    model = nn.Sequential(layer)
    calls = []
    monkeypatch.setattr(
        torch.ops._C, "nvfp4_moe_dense_stage_sm70_out", object(), raising=False
    )

    def record_call(
        out,
        x,
        offsets,
        expert_ids,
        ptrs_w,
        ptrs_s,
        num_experts,
        k,
        n,
        group_size,
    ):
        calls.append(
            (
                tuple(out.shape),
                tuple(x.shape),
                offsets.tolist(),
                expert_ids.tolist(),
                num_experts,
                k,
                n,
                group_size,
            )
        )

    monkeypatch.setattr(warmup.sm70_ops, "nvfp4_moe_dense_stage_sm70_out", record_call)
    monkeypatch.setattr(
        warmup,
        "_prepare_compact_slot_groups",
        lambda sorted_ids, offsets, active_ids: (
            offsets.copy_(torch.arange(offsets.numel(), dtype=torch.int32)),
            active_ids.copy_(sorted_ids),
        ),
    )
    monkeypatch.setattr(
        torch.ops._C,
        "silu_and_mul",
        lambda out, gate_up: out.zero_(),
        raising=False,
    )

    discovered = list(warmup._iter_unique_nvfp4_moe_layers(model))
    count = warmup._warmup_nvfp4_moe_decode_layers(discovered, [1, 2])

    assert discovered == [layer]
    assert count == 4
    assert [call[4] for call in calls] == [8, 8, 16, 16]
    assert calls[0][2] == list(range(9))
    assert calls[0][3] == list(range(8))
    assert calls[2][2] == list(range(17))
    assert calls[2][3] == list(range(16))


def test_nvfp4_moe_warmup_includes_opted_in_cuda_graph_shapes(monkeypatch):
    monkeypatch.setattr(warmup.envs, "VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS", 640)
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(
                cudagraph_capture_sizes=[1, 2, 4, 5, 8, 9, 18, 20, 40, 60, 80, 81]
            )
        )
    )

    assert warmup._get_nvfp4_moe_token_counts(
        worker, [_nvfp4_moe_layer()], [1, 2, 4, 8]
    ) == [*range(1, 11), 18, 20, 40, 60, 80]


def test_nvfp4_moe_warmup_uses_slot_compact_through_b10(monkeypatch):
    layer = _nvfp4_moe_layer()
    calls = []
    monkeypatch.setattr(
        torch.ops._C, "nvfp4_moe_dense_stage_sm70_out", object(), raising=False
    )
    monkeypatch.setattr(
        warmup.sm70_ops,
        "nvfp4_moe_dense_stage_sm70_out",
        lambda *args: calls.append(args),
    )
    monkeypatch.setattr(
        warmup,
        "_prepare_compact_slot_groups",
        lambda sorted_ids, offsets, active_ids: (
            offsets.copy_(torch.arange(offsets.numel(), dtype=torch.int32)),
            active_ids.copy_(sorted_ids),
        ),
    )
    monkeypatch.setattr(
        torch.ops._C,
        "silu_and_mul",
        lambda out, gate_up: out.zero_(),
        raising=False,
    )

    assert warmup._warmup_nvfp4_moe_decode_layers([layer], [9, 10]) == 4
    assert [call[6] for call in calls] == [72, 72, 80, 80]
    assert all(call[2].numel() == call[6] + 1 for call in calls)
    assert all(call[3].numel() == call[6] for call in calls)
    assert calls[0][2].tolist() == list(range(73))
    assert calls[2][2].tolist() == list(range(81))


def test_nvfp4_moe_warmup_uses_full_expert_groups_above_compact_b10(monkeypatch):
    layer = _nvfp4_moe_layer()
    calls = []
    monkeypatch.setattr(
        torch.ops._C, "nvfp4_moe_dense_stage_sm70_out", object(), raising=False
    )
    monkeypatch.setattr(
        warmup.sm70_ops,
        "nvfp4_moe_dense_stage_sm70_out",
        lambda *args: calls.append(args),
    )
    monkeypatch.setattr(
        torch.ops._C,
        "silu_and_mul",
        lambda out, gate_up: out.zero_(),
        raising=False,
    )

    assert warmup._warmup_nvfp4_moe_decode_layers([layer], [11]) == 2
    assert [call[6] for call in calls] == [256, 256]
    assert all(call[2].numel() == 257 for call in calls)
    assert all(call[3].numel() == 256 for call in calls)
    assert calls[0][2][:89].tolist() == list(range(89))
    assert calls[0][2][89:].tolist() == [88] * (257 - 89)


def test_nvfp4_moe_warmup_supports_mtp4_c16_width(monkeypatch):
    layer = _nvfp4_moe_layer()
    calls = []
    monkeypatch.setattr(
        torch.ops._C, "nvfp4_moe_dense_stage_sm70_out", object(), raising=False
    )
    monkeypatch.setattr(
        warmup.sm70_ops,
        "nvfp4_moe_dense_stage_sm70_out",
        lambda *args: calls.append(args),
    )
    monkeypatch.setattr(
        torch.ops._C,
        "silu_and_mul",
        lambda out, gate_up: out.zero_(),
        raising=False,
    )

    assert warmup._warmup_nvfp4_moe_decode_layers([layer], [80]) == 2
    assert [tuple(call[0].shape) for call in calls] == [(640, 256), (640, 2048)]
    assert [call[6] for call in calls] == [256, 256]
    offsets = calls[0][2]
    assert offsets.numel() == 257
    assert offsets[:4].tolist() == [0, 3, 6, 9]
    assert offsets[-1].item() == 640


def test_fp8_coordinated_warmup_leader_broadcasts_rank0_lut(monkeypatch):
    import vllm.distributed.parallel_state as parallel_state

    calls = []
    broadcasts = []
    barriers = []

    def broadcast_object(payload, src):
        broadcasts.append((payload, src))
        return payload

    def warmup_layers(layers, m_values):
        calls.append((layers, m_values))
        return 5

    tp_group = SimpleNamespace(
        world_size=4,
        rank_in_group=0,
        broadcast_object=broadcast_object,
        barrier=lambda: barriers.append(True),
    )
    monkeypatch.setenv("VLLM_SM70_FP8_TUNE_SMALL_SHAPES", "1")
    monkeypatch.setenv("VLLM_SM70_FP8_COORDINATED_TUNING", "1")
    warmup.envs.disable_envs_cache()
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(
        warmup,
        "_warmup_fp8_dense_layers",
        warmup_layers,
    )
    monkeypatch.setattr(warmup, "_export_lut_bytes", lambda device: (b"lut", 7))
    monkeypatch.setattr(
        warmup,
        "_import_lut_bytes",
        lambda device, payload: (_ for _ in ()).throw(AssertionError()),
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda device: None)

    layers = [(nn.Module(), False)]
    count = warmup._warmup_fp8_dense_layers_coordinated(
        layers,
        [1, 4],
        torch.device("cuda:0"),
    )

    assert count == 5
    assert calls == [(layers, [1, 4]), (layers, [1, 4])]
    assert broadcasts == [(b"lut", 0)]
    assert barriers == [True]


def test_fp8_coordinated_warmup_follower_imports_rank0_lut(monkeypatch):
    import vllm.distributed.parallel_state as parallel_state

    calls = []
    imports = []
    barriers = []

    def warmup_layers(layers, m_values):
        calls.append((layers, m_values))
        return 5

    def import_lut(device, payload):
        imports.append((device, payload))
        return 7

    tp_group = SimpleNamespace(
        world_size=4,
        rank_in_group=2,
        broadcast_object=lambda payload, src: b"lut",
        barrier=lambda: barriers.append(True),
    )
    monkeypatch.setenv("VLLM_SM70_FP8_TUNE_SMALL_SHAPES", "1")
    monkeypatch.setenv("VLLM_SM70_FP8_COORDINATED_TUNING", "1")
    warmup.envs.disable_envs_cache()
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(
        warmup,
        "_warmup_fp8_dense_layers",
        warmup_layers,
    )
    monkeypatch.setattr(
        warmup,
        "_export_lut_bytes",
        lambda device: (_ for _ in ()).throw(AssertionError()),
    )
    monkeypatch.setattr(
        warmup,
        "_import_lut_bytes",
        import_lut,
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda device: None)

    layers = [(nn.Module(), False)]
    count = warmup._warmup_fp8_dense_layers_coordinated(
        layers,
        [1, 4],
        torch.device("cuda:2"),
    )

    assert count == 5
    assert calls == [(layers, [1, 4])]
    assert imports == [(torch.device("cuda:2"), b"lut")]
    assert barriers == [True]


def test_fp8_explicit_lut_reuse_allows_dynamic_cache_import(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_FP8_TUNE_SMALL_SHAPES", "1")
    monkeypatch.setenv("VLLM_SM70_FP8_REUSE_IMPORTED_CACHE", "1")
    warmup.envs.disable_envs_cache()

    assert not warmup._lut_cache_disabled_for_dynamic_quant_dispatch(
        has_awq_dense=False,
        has_fp8_dense=True,
        fp4_kinds=set(),
    )


def test_fp8_dynamic_tuning_skips_stale_lut_by_default(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_FP8_TUNE_SMALL_SHAPES", "1")
    monkeypatch.delenv("VLLM_SM70_FP8_REUSE_IMPORTED_CACHE", raising=False)
    warmup.envs.disable_envs_cache()

    assert warmup._lut_cache_disabled_for_dynamic_quant_dispatch(
        has_awq_dense=False,
        has_fp8_dense=True,
        fp4_kinds=set(),
    )
