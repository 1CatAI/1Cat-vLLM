# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.distributed.parallel_state import _ep_device_communicator_required


@pytest.mark.parametrize("required_mode", [None, "ep", "dp", "pcp", "eplb"])
def test_ep_metadata_only_is_limited_to_pure_tp(required_mode):
    config = SimpleNamespace(
        enable_expert_parallel=required_mode == "ep",
        data_parallel_size=2 if required_mode == "dp" else 1,
        prefill_context_parallel_size=2 if required_mode == "pcp" else 1,
        enable_eplb=required_mode == "eplb",
    )
    assert _ep_device_communicator_required(config) == (required_mode is not None)
