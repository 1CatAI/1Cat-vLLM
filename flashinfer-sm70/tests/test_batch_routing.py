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


def test_unprepared_qsa_does_not_allocate_workspace():
    q = torch.empty(8, 6, 256, dtype=torch.float16)
    assert fi.try_qsa(q, None, None, None, None, None, None) is None


def test_cpu_prepare_is_a_noop(monkeypatch):
    monkeypatch.setattr(
        torch.ops, "load_library", lambda *_: pytest.fail("unexpected load")
    )
    fi.prepare(torch.nn.Module(), torch.device("cpu"))


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
