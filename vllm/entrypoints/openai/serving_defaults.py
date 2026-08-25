# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-model serving defaults for the OpenAI-compatible endpoints.

Some models are only correct when served with a specific decoding recipe
(e.g. document-OCR models require their anti-repetition logits processor and
``skip_special_tokens=False``). This registry lets the serving layer apply a
model's recipe by default — a client gets correct, deterministic output
without knowing any engine knobs — while still allowing per-request
overrides through the existing request fields (``vllm_xargs``,
``skip_special_tokens``, ...), clamped to registered bounds so a public API
cannot be abused with pathological parameter values.

Plugins register defaults for their models at import time via
:func:`register_model_serving_defaults`; in-tree models register below.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class ModelServingDefaults:
    """Serving-layer recipe for one model architecture."""

    # Fill-in values for sampling fields the request left unset. Only keys
    # understood by the protocol's `default_sampling_params` channel are
    # meaningful here (e.g. "temperature", "skip_special_tokens").
    sampling_defaults: dict[str, Any] = field(default_factory=dict)

    # Default `SamplingParams.extra_args` entries (e.g. the anti-repetition
    # processor parameters). Request-supplied `vllm_xargs` keys win, subject
    # to `extra_args_bounds`.
    extra_args_defaults: dict[str, Any] = field(default_factory=dict)

    # Inclusive (lo, hi) bounds for integer `vllm_xargs` keys a client may
    # override. Values outside the bounds are rejected (HTTP 400), not
    # silently clamped, so misconfigured clients notice.
    extra_args_bounds: dict[str, tuple[int, int]] = field(default_factory=dict)

    # Fully-qualified names (or bare class names) of logits processors that
    # MUST be loaded in the engine for this model's recipe to hold. Startup
    # fails loudly when one is missing, instead of serving silently-degraded
    # output.
    required_logits_processors: tuple[str, ...] = ()

    # Kwargs for SamplingParams.repetition_detection applied when the request
    # does not set the field. Safety net for degeneration patterns that evade
    # the n-gram processor (e.g. slow-drift table loops longer than its ban
    # window): generation terminates instead of producing thousands of junk
    # tokens. A request-supplied value always wins.
    repetition_detection_defaults: dict[str, int] | None = None

    # Optional multi-image request profile (e.g. multi-page document
    # parsing uses a different anti-repetition window). ``None`` falls back
    # to the single-image values above. Bounds apply to both profiles.
    sampling_defaults_multi_image: dict[str, Any] | None = None
    extra_args_defaults_multi_image: dict[str, Any] | None = None

    def sampling_defaults_for(self, num_image_items: int) -> dict[str, Any]:
        if num_image_items > 1 and self.sampling_defaults_multi_image is not None:
            return self.sampling_defaults_multi_image
        return self.sampling_defaults

    def extra_args_defaults_for(self, num_image_items: int) -> dict[str, Any]:
        if num_image_items > 1 and self.extra_args_defaults_multi_image is not None:
            return self.extra_args_defaults_multi_image
        return self.extra_args_defaults


_REGISTRY: dict[str, ModelServingDefaults] = {}


def register_model_serving_defaults(
    architecture: str, defaults: ModelServingDefaults
) -> None:
    """Register (or replace) the serving defaults for a model architecture."""
    if architecture in _REGISTRY:
        logger.warning(
            "Overriding serving defaults previously registered for %s.",
            architecture,
        )
    _REGISTRY[architecture] = defaults


def get_model_serving_defaults(
    architectures: Iterable[str] | None,
) -> ModelServingDefaults | None:
    """Resolve defaults for the first registered architecture, if any."""
    for arch in architectures or ():
        defaults = _REGISTRY.get(arch)
        if defaults is not None:
            return defaults
    return None


def _matches_processor(entry: Any, required: str) -> bool:
    name = entry if isinstance(entry, str) else getattr(entry, "__name__", "")
    short = required.rsplit(":", 1)[-1].rsplit(".", 1)[-1]
    return name == required or name.rsplit(":", 1)[-1].rsplit(".", 1)[-1] == short


def validate_required_logits_processors(
    defaults: ModelServingDefaults,
    loaded_processors: list[Any] | None,
    model_name: str,
) -> None:
    """Fail startup when the model's recipe processors are not loaded.

    The processors are loaded through the engine's ``--logits-processors``
    channel; the serving defaults only supply their per-request parameters.
    Serving without the processor would produce the documented looping
    behavior with no visible error, so this is fail-closed.
    """
    for required in defaults.required_logits_processors:
        if not any(
            _matches_processor(entry, required) for entry in (loaded_processors or [])
        ):
            raise ValueError(
                f"Model {model_name} requires the logits processor "
                f"{required!r} for correct serving, but it is not loaded. "
                f"Start the server with --logits-processors {required}."
            )


def apply_repetition_detection_default(defaults, sampling_params) -> None:
    """Fill SamplingParams.repetition_detection from the model defaults when
    the request left it unset."""
    if (
        defaults.repetition_detection_defaults
        and getattr(sampling_params, "repetition_detection", None) is None
    ):
        from vllm.sampling_params import RepetitionDetectionParams

        sampling_params.repetition_detection = RepetitionDetectionParams(
            **defaults.repetition_detection_defaults
        )


def merge_extra_args(
    defaults: ModelServingDefaults,
    request_xargs: Mapping[str, Any] | None,
    *,
    num_image_items: int = 1,
) -> dict[str, Any]:
    """Overlay request ``vllm_xargs`` on the model's recipe defaults.

    Request keys win; integer keys with registered bounds are validated and
    a ValueError (surfaced as HTTP 400) is raised for out-of-range values.
    The defaults profile is selected by the request's image count (multi-page
    recipes may use e.g. a different anti-repetition window).
    """
    merged: dict[str, Any] = dict(defaults.extra_args_defaults_for(num_image_items))
    for key, value in (request_xargs or {}).items():
        bounds = defaults.extra_args_bounds.get(key)
        if bounds is not None:
            lo, hi = bounds
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"vllm_xargs[{key!r}] must be an integer in "
                    f"[{lo}, {hi}], got {value!r}."
                )
            if not lo <= value <= hi:
                raise ValueError(
                    f"vllm_xargs[{key!r}]={value} is outside the allowed "
                    f"range [{lo}, {hi}] for this model."
                )
        merged[key] = value
    return merged


# --------------------------------------------------------------------------
# In-tree registrations.
# --------------------------------------------------------------------------

# DeepSeek-OCR: the checkpoint README's serving recipe. The anti-repetition
# processor is mandatory per the model's documentation ("prevent looping on
# coordinate tokens"); `skip_special_tokens=False` and greedy decoding are
# part of the transcription contract. Values: ngram_size=30, window_size=90,
# whitelist = {<td>, </td>} token ids.
register_model_serving_defaults(
    "DeepseekOCRForCausalLM",
    ModelServingDefaults(
        sampling_defaults={
            "temperature": 0.0,
            "skip_special_tokens": False,
        },
        extra_args_defaults={
            "ngram_size": 30,
            "window_size": 90,
            "whitelist_token_ids": [128821, 128822],
        },
        extra_args_bounds={
            "ngram_size": (1, 512),
            "window_size": (1, 8192),
        },
        # Grounded in the observed failure (arXiv page-09 endpoint battery,
        # w13): near-identical empty table rows repeated for ~6.5k tokens,
        # evading the 90-token n-gram ban window. min_count=8 keeps legitimate
        # dense tables (separators repeat, but full-row patterns rarely repeat
        # 8x identically) while terminating clear degeneration.
        repetition_detection_defaults={
            "max_pattern_size": 60,
            "min_pattern_size": 2,
            "min_count": 8,
        },
        required_logits_processors=(
            "vllm.model_executor.models.deepseek_ocr.NGramPerReqLogitsProcessor",
        ),
        # The DeepSeek-OCR README documents a single recipe with no
        # multi-page variant — the single-image profile applies to any
        # image count (multi profiles stay None).
    ),
)
