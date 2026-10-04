# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Architecture adapters, independent of Transformers model construction."""


def get_gguf_adapter(config, tp_size=1):
    if config.get_text_config().model_type == "qwen4_exp_text":
        from .qwen4exp import Qwen4ExpAdapter

        return Qwen4ExpAdapter(config.get_text_config(), tp_size=tp_size)
    if config.get_text_config().model_type == "qwen3_5_moe_text":
        from .qwen35_moe import Qwen35MoeAdapter

        return Qwen35MoeAdapter(config.get_text_config(), tp_size=tp_size)
    if config.get_text_config().model_type == "qwen3_5_text":
        from .qwen35 import Qwen35Adapter

        return Qwen35Adapter(config.get_text_config(), tp_size=tp_size)
    return None
