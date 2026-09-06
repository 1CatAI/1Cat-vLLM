# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace as NS

import pytest
import torch

from vllm import envs
from vllm.model_executor.layers import sm70_flashinfer_batch as fi


def metadata(rows=8, **changes):
    fields = dict(
        num_prefills=0,
        num_prefill_tokens=0,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_decodes=rows,
        num_decode_tokens=rows,
        non_spec_state_indices_tensor=torch.arange(rows, dtype=torch.int32),
    )
    fields.update(changes)
    return NS(**fields)


def test_opt_in_only(monkeypatch):
    envs.disable_envs_cache()
    monkeypatch.delenv("VLLM_SM70_FLASHINFER_BATCH", raising=False)
    assert not envs.VLLM_SM70_FLASHINFER_BATCH


@pytest.mark.parametrize("rows", [2, 4, 8, 16, 32, 64])
def test_decode_shape_is_not_a_server_configuration(rows):
    assert fi.uniform_decode(metadata(rows), rows)


@pytest.mark.parametrize(
    "changes",
    [
        dict(num_prefills=1),
        dict(num_prefill_tokens=1),
        dict(num_spec_decodes=1),
        dict(num_spec_decode_tokens=1),
        dict(num_decode_tokens=9),
        dict(non_spec_state_indices_tensor=None),
    ],
)
def test_prefill_spec_and_invalid_metadata_fall_back(changes):
    assert not fi.uniform_decode(metadata(**changes), 8)


def test_padding_uses_scheduler_owned_slots_without_host_values():
    meta = metadata(5, non_spec_state_indices_tensor=torch.full((8,), -1))
    assert fi.uniform_decode(meta, 8)
    assert not fi.uniform_decode(meta, 16)


def test_m1_and_unknown_keep_original_paths():
    assert not fi.uniform_decode(metadata(1), 1)
    assert not fi.uniform_decode(None, 8)
    assert not fi.uniform_decode(metadata(65), 65)


def test_unprepared_gdn_does_not_touch_projection_or_state():
    assert not fi.try_gdn(NS(), torch.empty(8, 2560), None, None, None, None, None)


def test_unbound_aot_placeholders_fall_back_before_transpose():
    layer = NS(_sm70_fi_ready=True)
    empty = torch.empty(0)
    assert not fi.try_gdn(
        layer, torch.empty(8, 2560), None, None, empty, empty, metadata()
    )


def test_unprepared_qsa_does_not_allocate_workspace():
    q = torch.empty(8, 6, 256, dtype=torch.float16)
    assert fi.try_qsa(q, None, None, None, None, None, None) is None


def test_unprepared_mqa_does_not_allocate_or_read_metadata():
    q = torch.empty(8, 4, 128, dtype=torch.float16)
    assert not fi.try_mqa(q, None, None, None, None, None, 4, 1.0, None, None)


@pytest.mark.parametrize("rows,workers", [(4, 160), (8, 160), (16, 320)])
def test_mqa_dispatch_preserves_int64_positions_and_caller_outputs(
    monkeypatch, rows, workers
):
    # CPU-only dispatch test: substitute the op, not a CUDA implementation.
    device = torch.device("cpu")
    monkeypatch.setitem(fi._MQA_SMS, device, 80)
    seen = []
    monkeypatch.setattr(
        torch.ops._C_flashinfer_mqa_sm70,
        "run",
        lambda *args: seen.append(args),
        raising=False,
    )
    q = torch.empty(rows, 4, 128, dtype=torch.float16)
    k = torch.empty(rows, 196, 1, 128, dtype=torch.float16)
    table = torch.empty(rows, 1, dtype=torch.int32)
    requests = torch.arange(rows, dtype=torch.int32)
    positions = torch.arange(rows, dtype=torch.int64)
    lengths = torch.full_like(requests, 196)
    out = torch.empty(rows, 196)
    visible = torch.empty_like(requests)
    assert fi.try_mqa(
        q, k, table, requests, positions, lengths, 4, 128**0.5, out, visible
    )
    assert len(seen) == 1
    assert seen[0][4] is positions
    assert seen[0][6] is out and seen[0][7] is visible
    assert seen[0][-1] == workers
    assert seen[0][8].shape == (workers + 1, 2)
    # Other dtype/layout/components fall back locally, without attempting CUDA.
    assert not fi.try_mqa(
        q.float(), k, table, requests, positions, lengths, 4, 1.0, out, visible
    )
    assert not fi.try_mqa(
        q, k, table, requests.long(), positions, lengths, 4, 1.0, out, visible
    )
    assert len(seen) == 1


def test_cpu_prepare_is_a_noop(monkeypatch):
    monkeypatch.setattr(
        torch.ops, "load_library", lambda *_: pytest.fail("unexpected load")
    )
    fi.prepare(torch.nn.Module(), torch.device("cpu"))


@pytest.mark.parametrize("existing", ["first", "second"])
def test_preloaded_component_prevents_duplicate_fragment_registration(
    monkeypatch, existing
):
    from types import SimpleNamespace

    fake_ops = SimpleNamespace(first=SimpleNamespace(), second=SimpleNamespace())
    getattr(fake_ops, existing).run = lambda: None
    monkeypatch.setattr(torch, "ops", fake_ops)
    monkeypatch.setattr(
        fi.importlib.util, "find_spec", lambda *_: pytest.fail("duplicate lookup")
    )
    fi.load_native_fragment("unused", ("first", "second"))


@pytest.mark.parametrize("origin", [None, "/package/vllm/native.abi3.so"])
def test_native_fragment_missing_or_wheel_resolved(monkeypatch, origin):
    from types import SimpleNamespace

    loaded = []
    monkeypatch.setattr(
        torch,
        "ops",
        SimpleNamespace(first=SimpleNamespace(), load_library=loaded.append),
    )
    monkeypatch.setattr(
        fi.importlib.util,
        "find_spec",
        lambda _: SimpleNamespace(origin=origin) if origin else None,
    )
    fi.load_native_fragment("unused", ("first",))
    assert loaded == ([origin] if origin else [])


@pytest.mark.parametrize("prepared", [False, True])
def test_unsupported_gdn_reload_cannot_keep_stale_derived_weights(
    monkeypatch, prepared
):
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn

    class UnsupportedGDN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_k_heads, self.num_v_heads, self.tp_size = 16, 48, 4
            self.in_proj_ba = NS(weight=None)
            self.in_proj_qkvz = NS(weight=None)
            self._sm70_fi_ready = prepared

    for key in ("VLLM_SM70_FLASHINFER_GDN_LIBRARY", "VLLM_SM70_FLASHINFER_QSA_LIBRARY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        qwen_gdn_linear_attn, "QwenGatedDeltaNetAttention", UnsupportedGDN
    )
    monkeypatch.setattr(fi, "current_platform", NS(is_device_capability=lambda _: True))
    monkeypatch.setattr(
        fi, "get_current_vllm_config", lambda: NS(speculative_config=None)
    )
    monkeypatch.setattr(fi, "load_native_fragment", lambda *args: None)
    layer = UnsupportedGDN()
    if prepared:
        with pytest.raises(RuntimeError, match="rebuild CUDA graphs"):
            fi.prepare(layer, torch.device("cuda"))
    else:
        fi.prepare(layer, torch.device("cuda"))


def test_weight_reload_preserves_captured_pointer():
    module = torch.nn.Module()
    weight = torch.randn(5, 3).half()
    fi.copy_derived_buffer(module, "packed", weight.t())
    pointer = module.packed.data_ptr()
    weight.fill_(2)
    fi.copy_derived_buffer(module, "packed", weight.t())
    assert module.packed.data_ptr() == pointer
    assert not module.state_dict()
    torch.testing.assert_close(module.packed, weight.t())
    with pytest.raises(RuntimeError, match="rebuild CUDA graphs"):
        fi.copy_derived_buffer(module, "packed", torch.empty(4, 5))
