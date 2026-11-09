# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm.compilation.passes.fusion.allreduce_rms_fusion import AllReduceFusionPass
from vllm.config.utils import Range


@pytest.mark.parametrize("end", [8192, 8193, 262144])
def test_sm70_push_norm_covers_mixed_compile_range(end):
    fusion = SimpleNamespace(
        disabled=False, sm70_tp4_push_mode=True, max_token_num=8192
    )
    assert AllReduceFusionPass.is_applicable_for_range(fusion, Range(1, end))


def test_non_sm70_fusion_keeps_workspace_range_limit():
    fusion = SimpleNamespace(disabled=False, max_token_num=8192)
    assert AllReduceFusionPass.is_applicable_for_range(fusion, Range(1, 8192))
    assert not AllReduceFusionPass.is_applicable_for_range(fusion, Range(1, 8193))
