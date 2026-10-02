# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.config import ModelConfig, VllmConfig
from vllm.v1.spec_decode import dflash


def test_draft_config_copy_skips_non_init_fields():
    # VllmConfig declares sm70_acceleration_report with Field(init=False).
    # dataclasses.replace passes it to the constructor anyway, which pydantic
    # rejects; the DFlash/DSpark proposer copies the target config on load.
    config = VllmConfig(
        model_config=ModelConfig(model="facebook/opt-125m", max_model_len=512)
    )
    config.sm70_acceleration_report = {"profile": "test"}
    copy = dflash.replace(config, additional_config={"draft": True})
    assert copy.additional_config == {"draft": True}
