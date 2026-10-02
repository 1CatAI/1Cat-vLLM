# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import envs
from vllm.models.qwen4_exp.nvidia.ops.sm70_qsa_tuning import (
    SM70_QSA_TUNING,
    legacy_qsa_tuning,
)


@pytest.mark.parametrize(
    "name,field",
    (
        ("VLLM_SM70_QSA_INDEXER_SCORE_TILE_MB", "score_tile_mb"),
        ("VLLM_SM70_QSA_INDEXER_CUBLAS_MIN_ROWS", "cublas_min_rows"),
        (
            "VLLM_SM70_QSA_INDEXER_CUBLAS_MIN_SCORE_ELEMENTS",
            "cublas_min_score_elements",
        ),
        ("VLLM_SM70_QSA_XQA_PAGE4_MIN_ROWS", "xqa_page4_min_rows"),
    ),
)
@pytest.mark.parametrize("override", (None, "0", "64", "4096", "-1", "invalid"))
def test_legacy_parser_and_unset_tuning_are_preserved(
    name, field, override, monkeypatch
):
    envs.disable_envs_cache()
    default = getattr(SM70_QSA_TUNING, field)
    monkeypatch.delenv(name, raising=False)
    if override is not None:
        monkeypatch.setenv(name, override)
    if override == "invalid":
        with pytest.raises(ValueError):
            legacy_qsa_tuning(name, default)
    else:
        assert legacy_qsa_tuning(name, default) == (
            default if override is None else int(override)
        )
