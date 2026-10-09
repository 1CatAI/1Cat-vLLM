# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 config (port of upstream vLLM b6d8e8af ``transformers_utils/configs/deepseek_v41.py``).

The official config.json nests the text model under ``text_config`` (model_type ``deepseek_v41_text``) and
the vision tower under ``vision_config``. The 1Cat SM70 port is text-only: the text fields are flattened onto
the top level (the model code reads flat attributes, as for V4), the vision fields are kept only as
``vision_*`` metadata and every vision/aligner tensor is load-skipped by
``DeepseekV41ForCausalLM.skip_checkpoint_weight`` (PORT_DESIGN §3.7).

Importing this module also registers the ``deepseek_v41_fp8`` quantization method (see
``_register_v41_quant_method``): it is the first DeepSeek-V4.1 code every process (engine, API server,
workers unpickling this config) executes, and the method must exist before
``ModelConfig._verify_quantization`` asks the registered overrides to claim the checkpoint.
"""

from typing import Any

from transformers import PretrainedConfig


class DeepseekV41Config(PretrainedConfig):
    model_type = "deepseek_v41"

    def __init__(
        self,
        text_config: dict[str, Any] | None = None,
        vision_config: dict[str, Any] | None = None,
        **kwargs,
    ):
        text_config = dict(text_config or {})
        vision_config = dict(vision_config or {})

        # ``rope_scaling`` is a property in Transformers v5 (backed by ``rope_parameters``): capture it here
        # and restore after super().__init__, which re-standardizes rope params. A flat (text-only) config
        # carries it in kwargs instead of text_config.
        rope_scaling = text_config.pop("rope_scaling", None)
        if rope_scaling is None:
            rope_scaling = kwargs.pop("rope_scaling", None)
        self.text_model_type = text_config.pop("model_type", None)

        for key, value in text_config.items():
            # Don't clobber PretrainedConfig properties (e.g. is_encoder_decoder).
            if isinstance(getattr(type(self), key, None), property):
                continue
            setattr(self, key, value)

        super().__init__(**kwargs)
        if rope_scaling is not None:
            self.rope_parameters = dict(rope_scaling)

        # The V4.1 quantization_config carries the expert dtype; the model code reads it from hf_config.
        quant_cfg = getattr(self, "quantization_config", None) or {}
        if not hasattr(self, "expert_dtype") and "expert_dtype" in quant_cfg:
            self.expert_dtype = quant_cfg["expert_dtype"]

        # Vision tower: metadata only (text-only port; vision.* / aligner.* / image_* are load-skipped).
        self.vision_n_layers = vision_config.get("num_hidden_layers", 0)
        self.vision_dim = vision_config.get("hidden_size", 1024)
        self.vision_n_heads = vision_config.get("num_attention_heads", 16)
        self.vision_inter_dim = vision_config.get("intermediate_size", 2816)
        self.vision_patch_size = vision_config.get("patch_size", 14)
        self.vision_rope_theta = vision_config.get("rope_theta", 10000.0)
        self.vision_downsample_ratio = vision_config.get("downsample_ratio", 3)
        self.vision_max_n_token = vision_config.get("max_image_tokens", 1024)
        self.vision_min_pixels = vision_config.get("min_pixels", 295936)
        self.vision_max_wh_ratio = vision_config.get("max_wh_ratio")


def _register_v41_quant_method() -> None:
    # Imported for its side effect: @register_quantization_config("deepseek_v41_fp8").
    import vllm.models.deepseek_v41.quant_config  # noqa: F401


_register_v41_quant_method()
