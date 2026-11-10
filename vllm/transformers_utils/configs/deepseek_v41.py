# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 configuration identity and its nested text configuration.

Configuration support does not register the V4.1 forward as a V4 model.
Runtime context limits are enforced by the implementation, independently of
the checkpoint's architectural maximum.
"""

from copy import deepcopy
from typing import Any

from transformers import PretrainedConfig


class DeepseekV41TextConfig(PretrainedConfig):
    model_type = "deepseek_v41_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 129280,
        hidden_size: int = 5120,
        moe_intermediate_size: int = 2304,
        num_hidden_layers: int = 40,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 1,
        head_dim: int = 512,
        max_position_embeddings: int = 1048576,
        rope_theta: float = 10000.0,
        rope_scaling: dict[str, Any] | None = None,
        rope_parameters: dict[str, Any] | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = deepcopy(rope_scaling)
        self.rope_parameters = deepcopy(rope_scaling or rope_parameters)
        super().__init__(**kwargs)
        if hasattr(self, "dspark_target_layer_ids"):
            # V4.1 exposes pre-block HC means. Supply the generic capture
            # interface directly, avoiding the legacy post-block +1 mapping.
            self.eagle_aux_hidden_state_layer_ids = list(self.dspark_target_layer_ids)


class DeepseekV41Config(PretrainedConfig):
    model_type = "deepseek_v41"
    sub_configs = {"text_config": DeepseekV41TextConfig}

    def __init__(
        self,
        text_config: dict[str, Any] | DeepseekV41TextConfig | None = None,
        vision_config: dict[str, Any] | None = None,
        **kwargs,
    ):
        if text_config is None:
            self.text_config = DeepseekV41TextConfig()
        elif isinstance(text_config, DeepseekV41TextConfig):
            self.text_config = text_config
        elif isinstance(text_config, dict):
            if text_config.get("model_type", "deepseek_v41_text") != (
                "deepseek_v41_text"
            ):
                raise ValueError("V4.1 requires a deepseek_v41_text configuration")
            self.text_config = DeepseekV41TextConfig(**deepcopy(text_config))
        else:
            raise TypeError("Invalid V4.1 text configuration")
        # Preserve the checkpoint metadata for round-trip loading. The first
        # implementation is text-only; this field does not enable vision.
        self.vision_config = deepcopy(vision_config)
        super().__init__(**kwargs)
