# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import math
from typing import TYPE_CHECKING, Any, Literal, get_args

from pydantic import Field, SkipValidation, field_validator, model_validator
from typing_extensions import Self

from vllm.config import LoadConfig
from vllm.config.cache import CacheDType
from vllm.config.kernel import MoEBackend
from vllm.config.model import ModelConfig
from vllm.config.parallel import ParallelConfig
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_hf_text_config
from vllm.utils.hashing import safe_hash
from vllm.utils.import_utils import LazyLoader, has_arctic_inference
from vllm.v1.attention.backends.registry import AttentionBackendEnum

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    import vllm.model_executor.layers.quantization as me_quant
else:
    PretrainedConfig = Any

    me_quant = LazyLoader(
        "model_executor", globals(), "vllm.model_executor.layers.quantization"
    )

logger = init_logger(__name__)

MTPModelTypes = Literal[
    "deepseek_mtp",
    "mimo_mtp",
    "mimo_v2_mtp",
    "glm4_moe_mtp",
    "glm4_moe_lite_mtp",
    "glm_ocr_mtp",
    "ernie_mtp",
    "nemotron_h_mtp",
    "exaone_moe_mtp",
    "exaone4_5_mtp",
    "qwen3_next_mtp",
    "qwen4_exp_mtp",
    "qwen3_5_mtp",
    "longcat_flash_mtp",
    "mtp",
    "pangu_ultra_moe_mtp",
    "step3p5_mtp",
    "hy_v3_mtp",
    "gemma4_mtp",
]
NgramGPUTypes = Literal["ngram_gpu"]
DFlashModelTypes = Literal["dflash"]
DFlashDDTreeModelTypes = Literal["dflash_ddtree"]
DSparkModelTypes = Literal["dspark"]
DDTreeBuildMode = Literal["best_first", "root_leaf", "spine_leaf"]
EagleModelTypes = Literal[
    "eagle",
    "eagle3",
    "extract_hidden_states",
    MTPModelTypes,
    DFlashModelTypes,
    DFlashDDTreeModelTypes,
    DSparkModelTypes,
]
SpeculativeMethod = Literal[
    "ngram",
    "medusa",
    "mlp_speculator",
    "draft_model",
    "suffix",
    "custom_class",
    EagleModelTypes,
    NgramGPUTypes,
]
RejectionSampleMethod = Literal["standard", "synthetic"]
DraftSampleMethod = Literal["greedy", "probabilistic"]


def _get_dflash2_checkpoint_draft_tokens(hf_config: Any) -> int | None:
    """Return the checkpoint-native DFlash2 draft width when declared."""
    dflash_config = getattr(hf_config, "dflash_config", None) or {}
    selector_top_k = int(dflash_config.get("selector_top_k", 0) or 0)
    block_size = int(dflash_config.get("block_size", 0) or 0)
    if selector_top_k <= 0 or block_size <= 1:
        return None
    return block_size - 1


def get_dflash_model_draft_tokens(speculative_config: Any) -> int:
    """Return the block width produced by the DFlash checkpoint itself.

    Lookup-augmented DFlash2 may hand a wider proposal to the target, but its
    model/convolutions must retain the block size used during training.
    """
    verify_tokens = int(getattr(speculative_config, "num_speculative_tokens", 0) or 0)
    if getattr(speculative_config, "method", None) != "dflash" or not getattr(
        speculative_config, "ngram_assist", False
    ):
        return verify_tokens

    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    hf_config = getattr(draft_model_config, "hf_config", None)
    trained_tokens = _get_dflash2_checkpoint_draft_tokens(hf_config)
    if trained_tokens is not None and trained_tokens < verify_tokens:
        return trained_tokens
    return verify_tokens


def uses_adaptive_dflash_lookup(speculative_config: Any) -> bool:
    """Whether DFlash2 can switch between model and augmented widths."""
    verify_tokens = int(getattr(speculative_config, "num_speculative_tokens", 0) or 0)
    return (
        getattr(speculative_config, "method", None) == "dflash"
        and bool(getattr(speculative_config, "ngram_assist", False))
        and 0 < get_dflash_model_draft_tokens(speculative_config) < verify_tokens
    )


@config
class SpeculativeConfig:
    """Configuration for speculative decoding."""

    enforce_eager: bool | None = None
    """Override the default enforce_eager from model_config"""
    # General speculative decoding control
    num_speculative_tokens: int = Field(default=None, gt=0)  # type: ignore[assignment]
    """The number of speculative tokens, if provided. It will default to the
    number in the draft model config if present; selector-based DFlash2 uses
    checkpoint ``block_size - 1``. Otherwise, it is required."""
    model: str | None = None
    """The name of the draft model, eagle head, or additional weights, if
    provided."""
    method: SpeculativeMethod | None = None
    """The name of the speculative method to use. If users provide and set the
    `model` param, the speculative method type will be detected automatically
    if possible, if `model` param is not provided, the method name must be
    provided.

    If using `ngram` method, the related configuration `prompt_lookup_max` and
    `prompt_lookup_min` should be considered."""
    draft_tensor_parallel_size: int | None = Field(default=None, ge=1)
    """The degree of the tensor parallelism for the draft model. Can only be 1
    or the same as the target model's tensor parallel size."""
    tensor_parallel_size: int | None = None
    """Users should pass "draft_tensor_parallel_size". This parameter's purpose is to
    warn users when they mistakenly provide the wrong argument."""

    # Draft model configuration
    quantization: me_quant.QuantizationMethods | str | None = None
    """Quantization method that was used to quantize the draft model weights.
    If `None`, we assume the model weights are not quantized. Note that it only
    takes effect when using the draft model-based speculative method."""
    moe_backend: MoEBackend | None = None
    """MoE backend to use for the draft model. When `None`, the draft model
    inherits the target model's `--moe-backend` setting. Useful when the
    drafter and generator require different MoE kernels (e.g. quantized
    generator with unquantized drafter)."""
    attention_backend: AttentionBackendEnum | None = None
    """Attention backend to use for the draft model. When `None`, the backend is
    automatically selected. Useful when the drafter requires a different attention
    backend (e.g. DFlash needs a non-causal-capable backend like FLASH_ATTN)."""
    kv_cache_dtype: CacheDType | None = None
    """KV cache dtype for the draft model. When ``None``, the draft inherits
    the target cache dtype, except for SM70 DFlash with a quantized target,
    which is resolved to ``auto`` during draft model loading."""
    max_model_len: int | None = Field(default=None, ge=1)
    """The maximum model length of the draft model. Used when testing the
    ability to skip speculation for some sequences."""
    revision: str | None = None
    """The specific model version to use for the draft model. It can be a
    branch name, a tag name, or a commit id. If unspecified, will use the
    default version."""
    code_revision: str | None = None
    """The specific revision to use for the draft model code on Hugging Face
    Hub. It can be a branch name, a tag name, or a commit id. If unspecified,
    will use the default version."""
    index_share_for_mtp_iteration: bool | None = None
    """Override whether MTP iterations reuse the first step's sparse indices.
    If ``None``, use the value from the draft model's Hugging Face config."""

    # Advanced control
    disable_padded_drafter_batch: bool = False
    """Disable input padding for speculative decoding. If set to True,
    speculative input batches can contain sequences of different lengths,
    which may only be supported by certain attention backends. This currently
    only affects the EAGLE method of speculation."""
    use_local_argmax_reduction: bool = False
    """Use vocab-parallel local argmax instead of all-gathering full logits
    for draft token generation. Reduces communication from O(vocab_size) to
    O(2 * tp_size) per token. Only applies to greedy draft selection in
    non-tree speculation."""

    # Ngram proposer configuration
    prompt_lookup_max: int | None = Field(default=None, ge=1)
    """Maximum size of ngram token window when using Ngram proposer, required
    when method is set to ngram."""
    prompt_lookup_min: int | None = Field(default=None, ge=1)
    """Minimum size of ngram token window when using Ngram proposer, if
    provided. Defaults to 1."""
    ngram_assist: bool = False
    """Combine prompt-ngram lookup with DFlash2. At the checkpoint-native
    width, full hits may skip the DFlash2 query and selector. If
    ``num_speculative_tokens`` is wider than the checkpoint block, DFlash2
    keeps its trained width and lookup may fill the extra target-verification
    positions. Only valid with ``method='dflash'`` and a DFlash2 selector
    capability."""

    # Alternative drafting strategies
    parallel_drafting: bool = False
    """Enable parallel drafting, where all speculative tokens are generated
    in parallel rather than sequentially. This can improve performance but
    requires the speculative model be trained to support parallel drafting.
    Only compatible with EAGLE and draft model methods."""

    # DFlash DDTree experimental configuration.
    ddtree_budget: int | None = Field(default=None, ge=1)
    """Maximum number of non-root DDTree verifier nodes. Defaults to
    num_speculative_tokens for method='dflash_ddtree'."""
    ddtree_top_k: int | None = Field(default=None, ge=1)
    """Per-depth DFlash candidate count used to build DDTree branches. Defaults
    to the DDTree node budget, matching the reference implementation."""
    ddtree_chain_seed: bool = False
    """Seed the DDTree with the top-1 chain before best-first branch expansion.
    Defaults to ``False`` to match the reference DDTree best-first expansion.
    Set to ``True`` to force a top-1 spine before branch expansion."""
    ddtree_tree_mode: DDTreeBuildMode = "best_first"
    """DDTree topology. ``best_first`` preserves cumulative-score expansion.
    ``root_leaf``/``spine_leaf`` preserve the top-1 chain and attach alternative
    candidates only as leaves under the top-1 spine."""
    ddtree_disable_tree_verify: bool = False
    """Use the flat DFlash path instead of DDTree tree verification."""

    # required configuration params passed from engine
    target_model_config: SkipValidation[ModelConfig] = None  # type: ignore
    """The configuration of the target model."""
    target_parallel_config: SkipValidation[ParallelConfig] = None  # type: ignore
    """The parallel configuration for the target model."""

    # params generated in the post-init stage
    draft_model_config: SkipValidation[ModelConfig] = None  # type: ignore
    """The configuration of the draft model initialized internal."""
    draft_parallel_config: SkipValidation[ParallelConfig] = None  # type: ignore
    """The parallel configuration for the draft model initialized internal."""

    # Suffix decoding configuration
    suffix_decoding_max_tree_depth: int = 24
    """The maximum depth of the suffix decoding global and prompt trees. The
    tree depth limits the sum of the prefix match and speculation lengths."""

    suffix_decoding_max_cached_requests: int = 10000
    """The maximum number of requests to cache in the global suffix tree. If
    exceeded, will trigger eviction in FIFO order. If set to 0, the global
    suffix tree is disabled and past responses are not cached (prompt trees
    are still used)."""

    suffix_decoding_max_spec_factor: float = 1.0
    """The maximum spec factor for suffix decoding. The spec factor controls
    speculation lengths based on the prefix match length: max_spec_tokens =
    max_spec_factor * prefix_match_length."""

    suffix_decoding_min_token_prob: float = 0.1
    """The minimum token probability for suffix decoding. Will only speculate
    tokens with estimated probability (based on frequency counts) greater than
    or equal to this value."""

    draft_load_config: LoadConfig | None = None
    """Load config for the draft model. If not specified, will use the load
    config from the target model."""

    rejection_sample_method: RejectionSampleMethod = "standard"
    """The rejection sampling method to use. 'standard' uses probabilistic
    rejection sampling (with or without cached draft logits, controlled by
    draft_sample_method). 'synthetic' accepts draft tokens with a decaying
    probability calibrated to synthetic_acceptance_rate."""

    synthetic_acceptance_rates: list[float] | None = None
    """Per-position *unconditional* acceptance rates for synthetic rejection
    sampling. Position i's entry is the marginal probability that the first
    i+1 draft tokens are all accepted; the list must have length
    num_speculative_tokens, each entry in [0, 1], and be monotonically
    non-increasing. Only valid when rejection_sample_method is 'synthetic'.
    Mutually exclusive with synthetic_acceptance_length."""

    synthetic_acceptance_length: float | None = None
    """Target mean acceptance length for synthetic rejection sampling, in
    [1, num_speculative_tokens + 1]. Resolved internally to
    synthetic_acceptance_rates. Only valid when rejection_sample_method is 'synthetic'.
    Mutually exclusive with synthetic_acceptance_rates."""

    @staticmethod
    def _acceptance_length_to_rates(length: float, n: int) -> list[float]:
        """Mean acceptance length to unconditional per-position rates, using
        the minimum-variance schedule."""
        num_drafts = length - 1  # expected number of accepted draft tokens
        num_full = int(num_drafts)
        return (
            [1.0] * num_full + [num_drafts - num_full] + [0.0] * (n - num_full - 1)
        )[:n]

    @staticmethod
    def _resolve_synthetic_acceptance_rates(
        n: int,
        rates: list[float] | None,
        length: float | None,
    ) -> list[float]:
        """Return per-position unconditional acceptance rates from exactly one
        of `rates` or `length` (validates range, length, and monotonicity)."""
        if (rates is None) == (length is None):
            raise ValueError(
                "rejection_sample_method='synthetic' requires exactly one of "
                "synthetic_acceptance_rates or synthetic_acceptance_length."
            )
        if rates is not None:
            if len(rates) != n:
                raise ValueError(
                    f"synthetic_acceptance_rates must have length {n}, got {rates}."
                )
            if not all(0.0 <= r <= 1.0 for r in rates):
                raise ValueError(
                    f"synthetic_acceptance_rates entries must be in [0, 1], "
                    f"got {rates}."
                )
            if any(rates[i] > rates[i - 1] for i in range(1, n)):
                raise ValueError(
                    f"synthetic_acceptance_rates must be non-increasing, got {rates}."
                )
            return list(rates)
        assert length is not None
        if not 1.0 <= length <= float(n + 1):
            raise ValueError(
                f"synthetic_acceptance_length must be in [1, {n + 1}], got {length}."
            )
        return SpeculativeConfig._acceptance_length_to_rates(length, n)

    draft_sample_method: DraftSampleMethod = "greedy"
    """How the draft model samples tokens. 'greedy' always picks the argmax
    token, and the draft probabilities are treated as one-hot during rejection
    sampling. 'probabilistic' samples stochastically from the draft
    distribution and uses the full draft logits for the probability ratio test
    during rejection sampling. This comes at the cost of additional GPU memory
    usage."""

    dspark_confidence_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    """Conditional-acceptance threshold for DSpark prefix scheduling. A value
    of zero keeps every proposed position. Positive values retain only the
    leading positions whose calibrated confidence is at least the threshold."""

    dspark_confidence_temperatures: list[float] | None = None
    """Optional positive per-position temperatures applied to DSpark confidence
    logits before prefix scheduling. The list must match num_speculative_tokens."""

    dspark_max_verification_tokens: int | None = Field(default=None, ge=0)
    """Optional cap on DSpark draft tokens submitted to target verification.
    DSpark still generates the checkpoint's complete block; only a prefix is
    scheduled, so values below the checkpoint block size remain lossless."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        # Eagle3 and extract_hidden_states affect the computation graph because
        # they return intermediate hidden states in addition to the final hidden state.
        uses_aux_hidden_states = (
            self.method
            in (
                "eagle3",
                "extract_hidden_states",
            )
            or self.use_dflash_family()
            or self.use_dspark()
        )
        factors.append(uses_aux_hidden_states)

        # The specific layers used also affect the computation graph
        if uses_aux_hidden_states and self.draft_model_config is not None:
            layer_ids = getattr(
                self.draft_model_config.hf_config,
                "eagle_aux_hidden_state_layer_ids",
                None,
            )
            if layer_ids is None and self.use_dspark():
                layer_ids = getattr(
                    self.draft_model_config.hf_config,
                    "dspark_target_layer_ids",
                    None,
                )
            if layer_ids is not None:
                # Convert to tuple to make it hashable
                factors.append(tuple(layer_ids))

        if self.use_dflash_ddtree() and not self.ddtree_disable_tree_verify:
            factors.append(
                (
                    "dflash_ddtree",
                    self.ddtree_budget,
                    self.ddtree_top_k,
                    self.ddtree_chain_seed,
                    self.ddtree_tree_mode,
                )
            )

        if self.method == "mtp" and self.draft_model_config is not None:
            factors.append(
                getattr(
                    self.draft_model_config.hf_config,
                    "index_share_for_mtp_iteration",
                    False,
                )
            )

        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    @staticmethod
    def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:
        initial_architecture = hf_config.architectures[0]
        if hf_config.model_type in (
            "deepseek_v3",
            "deepseek_v32",
            "glm_moe_dsa",
        ):
            hf_config.model_type = "deepseek_mtp"
        if hf_config.model_type == "deepseek_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["DeepSeekMTPModel"]}
            )
        if hf_config.model_type == "deepseek_v4":
            hf_config.model_type = "deepseek_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["DeepSeekV4MTPModel"]}
            )
        if hf_config.model_type in ("pangu_ultra_moe"):
            hf_config.model_type = "pangu_ultra_moe_mtp"
        if hf_config.model_type == "pangu_ultra_moe_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["OpenPanguMTPModel"]}
            )

        if hf_config.architectures[0] == "MiMoForCausalLM":
            hf_config.model_type = "mimo_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "num_hidden_layers": 0,
                    "n_predict": n_predict,
                    "architectures": ["MiMoMTPModel"],
                }
            )

        if (arch := hf_config.architectures[0]) in (
            "MiMoV2ForCausalLM",
            "MiMoV2OmniForCausalLM",
        ):
            from vllm.model_executor.models.mimo_v2_mtp import (
                _MIMO_V2_PRO_NUM_MTP_LAYERS,
            )

            mtp_arch_maps = {
                "MiMoV2ForCausalLM": "MiMoV2MTPModel",
                "MiMoV2OmniForCausalLM": "MiMoV2OmniMTPModel",
            }

            hf_config.model_type = "mimo_v2_mtp"
            # vLLM currently supports only the first MiMo-V2 MTP layer.
            n_predict = _MIMO_V2_PRO_NUM_MTP_LAYERS
            hf_config.update(
                {
                    "num_hidden_layers": 0,
                    "n_predict": n_predict,
                    "num_nextn_predict_layers": n_predict,
                    "architectures": [mtp_arch_maps[arch]],
                }
            )

        if hf_config.architectures[0] == "MiMoV2FlashForCausalLM":
            from vllm.model_executor.models.mimo_v2_mtp import (
                _MIMO_V2_FLASH_NUM_MTP_LAYERS,
            )

            hf_config.model_type = "mimo_v2_mtp"
            # vLLM currently supports only the first MiMo-V2 MTP layer.
            n_predict = _MIMO_V2_FLASH_NUM_MTP_LAYERS
            hf_config.update(
                {
                    "num_hidden_layers": 0,
                    "n_predict": n_predict,
                    "num_nextn_predict_layers": n_predict,
                    "architectures": ["MiMoV2MTPModel"],
                }
            )

        if hf_config.architectures[0] == "Glm4MoeForCausalLM":
            hf_config.model_type = "glm4_moe_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "n_predict": n_predict,
                    "architectures": ["Glm4MoeMTPModel"],
                }
            )

        if hf_config.architectures[0] == "Glm4MoeLiteForCausalLM":
            hf_config.model_type = "glm4_moe_lite_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "num_hidden_layers": 0,
                    "n_predict": n_predict,
                    "architectures": ["Glm4MoeLiteMTPModel"],
                }
            )

        if hf_config.architectures[0] == "GlmOcrForConditionalGeneration":
            hf_config.model_type = "glm_ocr_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "num_hidden_layers": 0,
                    "n_predict": n_predict,
                    "architectures": ["GlmOcrMTPModel"],
                }
            )

        if hf_config.model_type == "ernie4_5_moe":
            hf_config.model_type = "ernie_mtp"
        if hf_config.model_type == "ernie_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["ErnieMTPModel"]}
            )

        if hf_config.architectures[0] == "NemotronH_Super_Omni_Reasoning_V3":
            # Promote VLM's text_config so MTP detection below fires correctly
            hf_config = hf_config.text_config

        if (
            hf_config.model_type in {"nemotron_h", "nemotron_h_puzzle"}
            and hasattr(hf_config, "num_nextn_predict_layers")
            and hf_config.num_nextn_predict_layers > 0
        ):
            # Check if this is an MTP variant
            hf_config.model_type = "nemotron_h_mtp"
        if hf_config.model_type == "nemotron_h_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["NemotronHMTPModel"]}
            )

        if hf_config.model_type == "qwen3_next":
            hf_config.model_type = "qwen3_next_mtp"
        if hf_config.model_type == "qwen3_next_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["Qwen3NextMTP"]}
            )

        if hf_config.model_type in {"qwen4_exp", "qwen4_exp_text"}:
            hf_config.model_type = "qwen4_exp_mtp"
        if hf_config.model_type == "qwen4_exp_mtp":
            text_config = get_hf_text_config(hf_config)
            n_predict = getattr(
                text_config,
                "mtp_num_hidden_layers",
                getattr(text_config, "num_nextn_predict_layers", None),
            )
            hf_config.update(
                {
                    "hc_mult": int(text_config.hc_count),
                    "n_predict": n_predict,
                    "architectures": ["Qwen4ExpMTP"],
                    "index_share_for_mtp_iteration": getattr(
                        text_config, "index_share_for_mtp_iteration", True
                    ),
                }
            )

        if hf_config.model_type == "exaone_moe":
            hf_config.model_type = "exaone_moe_mtp"
        if hf_config.model_type == "exaone_moe_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["ExaoneMoeMTP"]}
            )
        if "exaone4_5" in hf_config.model_type:
            hf_config.model_type = "exaone4_5_mtp"
        if hf_config.model_type == "exaone4_5_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["Exaone4_5_MTP"]}
            )
        if hf_config.model_type in ("qwen3_5", "qwen3_5_moe"):
            is_moe = hf_config.model_type == "qwen3_5_moe"
            hf_config.model_type = "qwen3_5_mtp"
            n_predict = getattr(hf_config, "mtp_num_hidden_layers", None)
            hf_config.update(
                {
                    "n_predict": n_predict,
                    "architectures": ["Qwen3_5MoeMTP" if is_moe else "Qwen3_5MTP"],
                }
            )
        if hf_config.model_type == "intern_s2_preview":
            text_config = getattr(hf_config, "text_config", None)
            is_moe = getattr(text_config, "model_type", None) == "qwen3_5_moe_text"
            hf_config.model_type = "qwen3_5_mtp"
            n_predict = getattr(text_config, "mtp_num_hidden_layers", None)
            hf_config.update(
                {
                    "n_predict": n_predict,
                    "architectures": ["Qwen3_5MoeMTP" if is_moe else "Qwen3_5MTP"],
                }
            )
        if hf_config.model_type == "longcat_flash":
            hf_config.model_type = "longcat_flash_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["LongCatFlashMTPModel"]}
            )

        if hf_config.model_type in ("step3p5", "step3p7") or hf_config.architectures[
            0
        ] in ("Step3p5ForCausalLM", "Step3p7ForConditionalGeneration"):
            quantization_config = getattr(hf_config, "quantization_config", None)
            hf_config = getattr(hf_config, "text_config", hf_config)
            if (
                quantization_config is not None
                and getattr(hf_config, "quantization_config", None) is None
            ):
                hf_config.update({"quantization_config": quantization_config})
            hf_config.model_type = "step3p5_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
            hf_config.update({"n_predict": n_predict, "architectures": ["Step3p5MTP"]})

        if initial_architecture == "MistralLarge3ForCausalLM":
            hf_config.update({"architectures": ["EagleMistralLarge3ForCausalLM"]})

        if hf_config.model_type == "hy_v3":
            hf_config.model_type = "hy_v3_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["HYV3MTPModel"]}
            )

        if hf_config.model_type == "gemma4_assistant":
            hf_config.model_type = "gemma4_mtp"
            text_config = getattr(hf_config, "text_config", hf_config)
            # The assistant runs all decoder layers in a single forward
            # call to produce one draft token, so n_predict=1.
            # num_kv_shared_layers must be 0: cross-model KV sharing is
            # set up by the proposer after model construction.
            if hasattr(text_config, "num_kv_shared_layers"):
                text_config.num_kv_shared_layers = 0
            hf_config.update({"n_predict": 1, "architectures": ["Gemma4MTPModel"]})

        return hf_config

    def __post_init__(self):
        # Note: "method" is a new parameter that helps to extend the
        # configuration of non-model-based proposers, and the "model" parameter
        # will be used to set the draft model, eagle head, or additional weight
        # when needed. If users do not specify "method", the speculative method
        # will be detected automatically if possible. If the speculative method
        # can not be detected, it will be considered as the "draft_model" by
        # default.

        # infer method from user args
        # Check if the model field contains a custom module path (e.g., 'pkg.Mod')
        if (
            self.model is not None
            and "." in self.model
            and not self.model.startswith(("http://", "https://", "file://"))
            and "/" not in self.model  # not a HuggingFace repo (org/model)
        ):
            # Treat as a custom class path
            self.method = "custom_class"
        elif self.method is None:
            if self.model in ("ngram", "[ngram]"):
                self.method = "ngram"
            else:
                self.method = "draft_model"

        if self.method in get_args(MTPModelTypes) and self.method != "mtp":
            logger.warning(
                "method `%s` is deprecated and replaced with mtp.", self.method
            )
            self.method = "mtp"

        if self.model is None and self.num_speculative_tokens is not None:
            if self.method == "mtp":
                if self.target_model_config is None:
                    raise ValueError("target_model_config must be present for mtp")
                if self.target_model_config.hf_text_config.model_type == "deepseek_v32":
                    # FIXME(luccafong): cudagraph with v32 MTP is not supported,
                    # remove this when the issue is fixed.
                    self.enforce_eager = True
                # use the draft model from the same model:
                self.model = self.target_model_config.model
                # Align the quantization of draft model for cases such as
                # --quantization fp8 with a bf16 checkpoint.
                if not self.quantization:
                    self.quantization = self.target_model_config.quantization
            elif self.method == "dspark":
                if self.target_model_config is None:
                    raise ValueError("target_model_config must be present for dspark")
                self.model = self.target_model_config.model
                if not self.quantization:
                    self.quantization = self.target_model_config.quantization
            elif self.method in ("ngram", "[ngram]"):
                self.model = "ngram"
            elif self.method == "ngram_gpu":
                self.model = "ngram_gpu"
            elif self.method == "suffix":
                self.model = "suffix"
            elif self.method == "extract_hidden_states":
                self.model = "extract_hidden_states"
            elif self.method == "custom_class":
                # method was set explicitly, but model should already contain the
                # custom module path. If not, this is a configuration error.
                if self.model is None:
                    raise ValueError(
                        "method='custom_class' requires 'model' to contain the "
                        "custom proposer module path (e.g., 'my_module.MyProposer')."
                    )
            else:
                raise ValueError(
                    "num_speculative_tokens was provided but without speculative model."
                )

        if self.method in ("ngram", "[ngram]"):
            self.method = "ngram"

        if self.ngram_assist:
            if self.prompt_lookup_min is None and self.prompt_lookup_max is None:
                self.prompt_lookup_min = 5
                self.prompt_lookup_max = 5
            elif self.prompt_lookup_min is None:
                self.prompt_lookup_min = self.prompt_lookup_max
            elif self.prompt_lookup_max is None:
                self.prompt_lookup_max = self.prompt_lookup_min
            assert self.prompt_lookup_min is not None
            assert self.prompt_lookup_max is not None
            if self.prompt_lookup_min > self.prompt_lookup_max:
                raise ValueError(
                    f"prompt_lookup_min={self.prompt_lookup_min} must "
                    f"be <= prompt_lookup_max={self.prompt_lookup_max}"
                )

        if self.method in ("ngram", "ngram_gpu"):
            # Set default values if not provided
            if self.prompt_lookup_min is None and self.prompt_lookup_max is None:
                # TODO(woosuk): Tune these values. They are arbitrarily chosen.
                self.prompt_lookup_min = 5
                self.prompt_lookup_max = 5
            elif self.prompt_lookup_min is None:
                if self.prompt_lookup_max is None:
                    raise ValueError(
                        "Either prompt_lookup_max or prompt_lookup_min must be "
                        "provided when using the ngram method."
                    )
                self.prompt_lookup_min = self.prompt_lookup_max
            elif self.prompt_lookup_max is None:
                if self.prompt_lookup_min is None:
                    raise ValueError(
                        "Either prompt_lookup_max or prompt_lookup_min must be "
                        "provided when using the ngram method."
                    )
                self.prompt_lookup_max = self.prompt_lookup_min

            # Validate values
            if self.prompt_lookup_min > self.prompt_lookup_max:
                raise ValueError(
                    f"prompt_lookup_min={self.prompt_lookup_min} must "
                    f"be <= prompt_lookup_max={self.prompt_lookup_max}"
                )

            # TODO: current we still need extract vocab_size from target model
            # config, in future, we may try refactor it out, and set
            # draft related config as None here.
            self.draft_model_config = self.target_model_config
            self.draft_parallel_config = self.target_parallel_config
        elif self.method == "suffix":
            self._validate_suffix_decoding()
        elif self.method == "custom_class":
            # Custom class proposer does not need a draft model.
            # It will dynamically load the user-provided class at runtime.
            logger.warning_once(
                "Using a custom class-based proposer backend. This is an "
                "experimental feature and the proposer interface is subject to "
                "breaking changes in future vLLM releases."
            )
            self.prompt_lookup_max = 0
            self.prompt_lookup_min = 0
            self.draft_model_config = self.target_model_config
            self.draft_parallel_config = self.target_parallel_config
        elif self.method == "extract_hidden_states":
            from vllm.transformers_utils.configs.extract_hidden_states import (
                ExtractHiddenStatesConfig,
            )

            # ExtractHiddenStatesModel is instantiated manually in load_model()
            # We just need to store the target model config for KV cache shape info
            self.model = "extract_hidden_states"
            self.prompt_lookup_max = 0
            self.prompt_lookup_min = 0

            if hasattr(self.draft_model_config, "hf_config"):
                hf_config = self.draft_model_config.hf_config.to_dict()
            elif (
                isinstance(self.draft_model_config, dict)
                and "hf_config" in self.draft_model_config
            ):
                hf_config = self.draft_model_config["hf_config"]
            else:
                hf_config = {}

            self.draft_model_config = copy.copy(self.target_model_config)
            self.draft_model_config.hf_config = ExtractHiddenStatesConfig(
                self.draft_model_config.hf_config, **hf_config
            )
            self.update_arch_()
            self.draft_parallel_config = self.target_parallel_config

        else:
            if not self.ngram_assist:
                self.prompt_lookup_max = 0
                self.prompt_lookup_min = 0

            if self.model is not None:
                self.draft_model_config = ModelConfig(
                    model=self.model,
                    runner="draft",
                    tokenizer=self.target_model_config.tokenizer,
                    tokenizer_mode=self.target_model_config.tokenizer_mode,
                    trust_remote_code=self.target_model_config.trust_remote_code,
                    allowed_local_media_path=self.target_model_config.allowed_local_media_path,
                    allowed_media_domains=self.target_model_config.allowed_media_domains,
                    dtype=self.target_model_config.dtype,
                    seed=self.target_model_config.seed,
                    revision=self.revision,
                    code_revision=self.code_revision,
                    tokenizer_revision=self.target_model_config.tokenizer_revision,
                    max_model_len=self.max_model_len,  # type: ignore[arg-type]
                    spec_target_max_model_len=self.target_model_config.max_model_len,
                    quantization=self.quantization,
                    enforce_eager=self.target_model_config.enforce_eager,
                    max_logprobs=self.target_model_config.max_logprobs,
                    hf_overrides=SpeculativeConfig.hf_config_override,
                    config_format=self.target_model_config.config_format,
                )

                # Automatically detect the method
                if (
                    self.method in ("eagle", "eagle3")
                    or self.use_dflash_family()
                    or self.use_dspark()
                ):
                    pass
                # examples:
                # yuhuili/EAGLE-LLaMA3-Instruct-8B
                # yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
                # AngelSlim/Qwen3-8B_eagle3
                elif "eagle-" in self.draft_model_config.model.lower():
                    self.method = "eagle"
                elif "eagle3" in self.draft_model_config.model.lower():
                    self.method = "eagle3"
                elif "dflash" in self.draft_model_config.model.lower():
                    self.method = "dflash"
                elif self.draft_model_config.hf_config.model_type == "medusa":
                    self.method = "medusa"
                elif self.draft_model_config.hf_config.model_type == "mlp_speculator":
                    self.method = "mlp_speculator"
                elif self.draft_model_config.hf_config.model_type in get_args(
                    MTPModelTypes
                ):
                    self.method = "mtp"
                    self._inherit_target_rope_for_extended_native_mtp()
                    if (
                        self.num_speculative_tokens > 1
                        and self.draft_model_config.hf_config.model_type
                        != "step3p5_mtp"
                    ):
                        logger.warning(
                            "Enabling num_speculative_tokens > 1 will run "
                            "multiple times of forward on same MTP layer"
                            ",which may result in lower acceptance rate"
                        )
                elif self.method == "draft_model":
                    pass
                else:
                    raise NotImplementedError(
                        f"Unsupported speculative method: '{self.method}'"
                    )

                # Replace hf_config for EAGLE draft_model
                if self.method in ("eagle", "eagle3") or self.use_dflash_family():
                    from vllm.transformers_utils.configs.eagle import EAGLEConfig
                    from vllm.transformers_utils.configs.speculators import (
                        SpeculatorsConfig,
                    )

                    if isinstance(
                        self.draft_model_config.hf_config,
                        (EAGLEConfig, SpeculatorsConfig),
                    ):
                        pass
                    else:
                        eagle_method = (
                            "dflash" if self.use_dflash_family() else self.method
                        )
                        eagle_config = EAGLEConfig(
                            self.draft_model_config.hf_config,
                            method=eagle_method,
                            model_type="eagle",
                        )
                        self.draft_model_config.hf_config = eagle_config
                        self.update_arch_()

                if self.use_dspark():
                    draft_hf_config = self.draft_model_config.hf_config
                    draft_hf_config.model_type = "deepseek_v4"
                    draft_hf_config.architectures = ["DSparkDraftModel"]
                    self.update_arch_()

                if self.use_dflash_family() or self.use_dspark():
                    self.parallel_drafting = True

                if self.num_speculative_tokens is not None and hasattr(
                    self.draft_model_config.hf_config, "num_lookahead_tokens"
                ):
                    self.draft_model_config.hf_config.num_lookahead_tokens = (
                        self.num_speculative_tokens
                    )

                n_predict = getattr(
                    self.draft_model_config.hf_config, "n_predict", None
                )
                if n_predict is not None:
                    if self.num_speculative_tokens is None:
                        # Default to max value defined in draft model config.
                        self.num_speculative_tokens = n_predict
                    elif (
                        self.num_speculative_tokens > n_predict
                        and self.num_speculative_tokens % n_predict != 0
                    ):
                        # Ensure divisibility for MTP module reuse.
                        raise ValueError(
                            f"num_speculative_tokens:{self.num_speculative_tokens}"
                            f" must be divisible by {n_predict=}"
                        )

                if self.num_speculative_tokens is None and self.use_dflash():
                    native_draft_tokens = _get_dflash2_checkpoint_draft_tokens(
                        self.draft_model_config.hf_config
                    )
                    if native_draft_tokens is not None:
                        self.num_speculative_tokens = native_draft_tokens
                        logger.info_once(
                            "Defaulting DFlash2 num_speculative_tokens to %d "
                            "from checkpoint block_size.",
                            native_draft_tokens,
                        )

                if self.num_speculative_tokens is None:
                    raise ValueError(
                        "A speculative model was provided, but "
                        "`num_speculative_tokens` was not provided"
                    )

                if self.use_dspark():
                    dspark_block_size = getattr(
                        self.draft_model_config.hf_config,
                        "dspark_block_size",
                        None,
                    )
                    if (
                        dspark_block_size is not None
                        and self.num_speculative_tokens < dspark_block_size
                    ):
                        raise ValueError(
                            "DSpark requires num_speculative_tokens >= "
                            f"dspark_block_size ({dspark_block_size}); got "
                            f"{self.num_speculative_tokens}. Smaller values "
                            "produce incorrect output."
                        )
                    if (
                        self.dspark_max_verification_tokens is not None
                        and self.dspark_max_verification_tokens
                        > self.num_speculative_tokens
                    ):
                        raise ValueError(
                            "dspark_max_verification_tokens must be <= "
                            f"num_speculative_tokens ({self.num_speculative_tokens}); "
                            f"got {self.dspark_max_verification_tokens}."
                        )
                    temperatures = self.dspark_confidence_temperatures
                    if temperatures is not None:
                        if len(temperatures) != self.num_speculative_tokens:
                            raise ValueError(
                                "dspark_confidence_temperatures must have length "
                                f"{self.num_speculative_tokens}; got {temperatures}."
                            )
                        if not all(
                            math.isfinite(value) and value > 0.0
                            for value in temperatures
                        ):
                            raise ValueError(
                                "dspark_confidence_temperatures entries must be "
                                f"finite and positive; got {temperatures}."
                            )
                    self._verify_dspark_final_stage_ownership()

                if self.use_dflash_ddtree() and self.ddtree_budget is None:
                    self.ddtree_budget = self.num_speculative_tokens

                self.draft_tensor_parallel_size = (
                    SpeculativeConfig._verify_and_get_draft_tp(
                        self.target_parallel_config,
                        self.draft_tensor_parallel_size,
                        self.draft_model_config.hf_config,
                    )
                )

                self.draft_model_config.max_model_len = (
                    SpeculativeConfig._maybe_override_draft_max_model_len(
                        self.max_model_len,
                        self.draft_model_config.max_model_len,
                        self.target_model_config.max_model_len,
                    )
                )

                self.draft_parallel_config = (
                    SpeculativeConfig.create_draft_parallel_config(
                        self.target_parallel_config,
                        self.draft_tensor_parallel_size,
                        # MRV2 constructs complete DFlash/DSpark drafters on the
                        # target model's final PP rank. They are local to that
                        # stage rather than partitioned across target PP ranks.
                        speculative_draft_pipeline_parallel_size=(
                            1 if self.use_dspark() or self.use_dflash() else None
                        ),
                    )
                )

        if self.index_share_for_mtp_iteration is not None:
            if self.method != "mtp" or self.draft_model_config is None:
                raise ValueError(
                    "index_share_for_mtp_iteration is only supported with method='mtp'"
                )
            self.draft_model_config.hf_config.index_share_for_mtp_iteration = (
                self.index_share_for_mtp_iteration
            )
        return self

    def _verify_dspark_final_stage_ownership(self) -> None:
        """Validate the layer contract for a final-stage-local drafter."""
        from vllm.distributed.utils import get_pp_indices

        pp_size = self.target_parallel_config.pipeline_parallel_size
        num_layers = self.target_model_config.get_total_num_hidden_layers()
        last_start, last_end = get_pp_indices(num_layers, pp_size - 1, pp_size)
        layer_ids = tuple(
            getattr(
                self.draft_model_config.hf_config,
                "dspark_target_layer_ids",
                (),
            )
        )
        outside_last_stage = [
            layer_id for layer_id in layer_ids if not last_start <= layer_id < last_end
        ]
        if not layer_ids or outside_last_stage:
            raise ValueError(
                "DSpark's final-stage-local drafter requires every "
                "dspark_target_layer_id to belong to the final target pipeline "
                f"stage [{last_start}, {last_end}); got {layer_ids}. Adjust "
                "VLLM_PP_LAYER_PARTITION or the DSpark layer contract."
            )

    def _inherit_target_rope_for_extended_native_mtp(self) -> None:
        """Validate and inherit target YaRN for an extended native drafter."""
        if (
            self.method != "mtp"
            or self.max_model_len is None
            or self.target_model_config is None
            or self.draft_model_config is None
            or self.model != self.target_model_config.model
        ):
            return

        draft_text_config = get_hf_text_config(self.draft_model_config.hf_config)
        draft_native_limit = getattr(draft_text_config, "max_position_embeddings", None)
        if draft_native_limit is None or self.max_model_len <= draft_native_limit:
            return

        target_text_config = get_hf_text_config(self.target_model_config.hf_config)
        rope_parameters = getattr(target_text_config, "rope_parameters", None)
        if (
            not isinstance(rope_parameters, dict)
            or rope_parameters.get("rope_type") != "yarn"
        ):
            raise ValueError(
                "Extending native MTP beyond max_position_embeddings requires "
                "explicit target YaRN rope_parameters; got "
                f"{rope_parameters!r}."
            )

        original_limit = rope_parameters.get("original_max_position_embeddings")
        factor = rope_parameters.get("factor")
        try:
            original_limit_value = float(original_limit)
            factor_value = float(factor)
            draft_native_limit_value = float(draft_native_limit)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Native MTP YaRN extension requires numeric "
                "original_max_position_embeddings and factor."
            ) from error

        if (
            not math.isfinite(original_limit_value)
            or not math.isfinite(factor_value)
            or original_limit_value <= 0
            or factor_value <= 1
            or original_limit_value != draft_native_limit_value
        ):
            raise ValueError(
                "Native MTP YaRN extension requires the target's positive "
                "original_max_position_embeddings to match the drafter's "
                "native max_position_embeddings and factor to be greater "
                f"than one; got original={original_limit!r}, "
                f"native={draft_native_limit!r}, factor={factor!r}."
            )

        validated_yarn_limit = int(original_limit_value * factor_value)
        validated_limit = min(
            validated_yarn_limit,
            self.target_model_config.max_model_len,
        )
        if self.max_model_len > validated_limit:
            raise ValueError(
                f"Native MTP max_model_len={self.max_model_len} exceeds the "
                f"validated target YaRN limit={validated_limit}."
            )

        draft_text_config.rope_parameters = copy.deepcopy(rope_parameters)
        logger.info(
            "Extended native MTP context from %s to %s with validated target "
            "YaRN parameters (factor=%s).",
            draft_native_limit,
            self.max_model_len,
            factor,
        )

    def _validate_suffix_decoding(self):
        if not has_arctic_inference():
            raise ImportError(
                "Arctic Inference is required for suffix decoding. "
                "Install via `pip install arctic-inference==0.2.0`."
            )
        if self.num_speculative_tokens is None:
            # Suffix decoding decides the actual number of speculative tokens
            # dynamically and treats num_speculative_tokens as a maximum limit.
            self.num_speculative_tokens = self.suffix_decoding_max_tree_depth
            logger.warning(
                "Defaulted num_speculative_tokens to %s for suffix decoding.",
                self.num_speculative_tokens,
            )
        # Validate values
        if self.suffix_decoding_max_tree_depth < 1:
            raise ValueError(
                f"suffix_decoding_max_tree_depth="
                f"{self.suffix_decoding_max_tree_depth} must be >= 1"
            )
        if self.suffix_decoding_max_cached_requests < 0:
            raise ValueError(
                f"suffix_decoding_max_cached_requests="
                f"{self.suffix_decoding_max_cached_requests} must be >= 0"
            )
        if self.suffix_decoding_max_spec_factor < 0:
            raise ValueError(
                f"suffix_decoding_max_spec_factor="
                f"{self.suffix_decoding_max_spec_factor} must be >= 0"
            )
        if not 0 <= self.suffix_decoding_min_token_prob <= 1:
            raise ValueError(
                f"suffix_decoding_min_token_prob="
                f"{self.suffix_decoding_min_token_prob} must be in [0, 1]"
            )

    @staticmethod
    def _maybe_override_draft_max_model_len(
        speculative_max_model_len: int | None,
        draft_max_model_len: int,
        target_max_model_len: int,
    ) -> int:
        """Determine the max sequence len for the draft model. This is usually
        the draft_max_model_len, but may be the target_max_model_len if it is
        less than the draft_max_model_len, or may be speculative_max_model_len
        if it is specified.

        This is necessary so that sequences do not exceed the capacity of the
        draft model or the target model.

        speculative_max_model_len is mainly used for testing that sequences can
        skip speculation.
        """

        if speculative_max_model_len is not None:
            if speculative_max_model_len > draft_max_model_len:
                raise ValueError(
                    f"{speculative_max_model_len=} cannot be "
                    f"larger than {draft_max_model_len=}"
                )

            if speculative_max_model_len > target_max_model_len:
                raise ValueError(
                    f"{speculative_max_model_len=} cannot be "
                    f"larger than {target_max_model_len=}"
                )

            return speculative_max_model_len

        result = min(
            draft_max_model_len,
            target_max_model_len,
        )
        if result != draft_max_model_len:
            logger.info(
                "Overriding draft model max model len from %d to %d",
                draft_max_model_len,
                result,
            )
        return result

    @staticmethod
    def _verify_and_get_draft_tp(
        target_parallel_config: ParallelConfig,
        speculative_draft_tensor_parallel_size: int | None,
        draft_hf_config: PretrainedConfig,
    ) -> int:
        """
        Verifies and adjusts the tensor parallel size for a draft model
        specified using speculative_draft_tensor_parallel_size.
        """
        # If speculative_draft_tensor_parallel_size is unset then set it
        # appropriately else verify that it is set correctly.
        if speculative_draft_tensor_parallel_size is None:
            if draft_hf_config.model_type == "mlp_speculator":
                speculative_draft_tensor_parallel_size = 1
                if target_parallel_config.tensor_parallel_size > 1:
                    logger.warning(
                        "%s cannot currently be run with tp>1; "
                        "setting speculative_draft_tensor_parallel_size=1",
                        draft_hf_config.model_type,
                    )
            else:
                speculative_draft_tensor_parallel_size = (
                    target_parallel_config.tensor_parallel_size
                )
        elif speculative_draft_tensor_parallel_size not in (
            1,
            target_parallel_config.tensor_parallel_size,
        ):
            raise ValueError(
                f"{speculative_draft_tensor_parallel_size=} cannot be "
                f"other value than 1 or target model tensor_parallel_size"
            )
        return speculative_draft_tensor_parallel_size

    def update_arch_(self):
        """
        EagleConfig and ExtractHiddenStatesConfig update architectures, so update all
        architectures-related fields in self.draft_model_config
        """
        self.draft_model_config.hf_text_config = get_hf_text_config(
            self.draft_model_config.hf_config
        )
        self.draft_model_config.model_arch_config = (
            self.draft_model_config.get_model_arch_config()
        )
        model_info, arch = self.draft_model_config.registry.inspect_model_cls(
            self.draft_model_config.architectures,
            self.draft_model_config,
        )
        self.draft_model_config._model_info = model_info
        self.draft_model_config._architecture = arch

    @staticmethod
    def create_draft_parallel_config(
        target_parallel_config: ParallelConfig,
        speculative_draft_tensor_parallel_size: int,
        speculative_draft_pipeline_parallel_size: int | None = None,
    ) -> ParallelConfig:
        """Create a parallel config for use by the draft worker.

        This is mostly a copy of the target parallel config, except the draft
        tensor size and, for final-stage-local drafters, pipeline size.
        """
        draft_parallel_config = ParallelConfig(
            pipeline_parallel_size=(
                target_parallel_config.pipeline_parallel_size
                if speculative_draft_pipeline_parallel_size is None
                else speculative_draft_pipeline_parallel_size
            ),
            tensor_parallel_size=speculative_draft_tensor_parallel_size,
            distributed_executor_backend=target_parallel_config.distributed_executor_backend,
            max_parallel_loading_workers=target_parallel_config.max_parallel_loading_workers,
            disable_custom_all_reduce=target_parallel_config.disable_custom_all_reduce,
            ray_workers_use_nsight=target_parallel_config.ray_workers_use_nsight,
            placement_group=target_parallel_config.placement_group,
        )

        return draft_parallel_config

    @field_validator("attention_backend", mode="before")
    @classmethod
    def _parse_attention_backend(cls, value: Any) -> Any:
        if isinstance(value, str):
            if value.lower() == "auto":
                return None
            return AttentionBackendEnum[value.upper()]
        return value

    @model_validator(mode="after")
    def _verify_args(self) -> Self:
        if self.tensor_parallel_size is not None:
            raise ValueError(
                "'tensor_parallel_size' is not a valid argument in the "
                "speculative_config. Please pass 'draft_tensor_parallel_size' instead."
            )

        if self.num_speculative_tokens is None:
            raise ValueError(
                "num_speculative_tokens must be provided with "
                "speculative model unless the draft model config contains an "
                "n_predict parameter."
            )

        if self.num_speculative_tokens <= 0:
            raise ValueError(
                "Expected num_speculative_tokens to be greater "
                f"than zero ({self.num_speculative_tokens})."
            )
        if self.ngram_assist:
            if not self.use_dflash():
                raise ValueError("ngram_assist is only supported with method='dflash'.")
            draft_hf_config = getattr(self.draft_model_config, "hf_config", None)
            dflash_config = getattr(draft_hf_config, "dflash_config", None) or {}
            if int(dflash_config.get("selector_top_k", 0) or 0) <= 0:
                raise ValueError("ngram_assist requires DFlash2 selector capability.")
        if self.use_dflash_ddtree():
            if self.ddtree_budget is None:
                self.ddtree_budget = self.num_speculative_tokens
            if (
                not self.ddtree_disable_tree_verify
                and self.ddtree_budget < self.num_speculative_tokens
            ):
                raise ValueError(
                    "ddtree_budget must be >= num_speculative_tokens when "
                    "dflash_ddtree tree verification is enabled."
                )

        if self.rejection_sample_method == "synthetic":
            # Consolidate to per-position rates
            self.synthetic_acceptance_rates = self._resolve_synthetic_acceptance_rates(
                self.num_speculative_tokens,
                self.synthetic_acceptance_rates,
                self.synthetic_acceptance_length,
            )
            self.synthetic_acceptance_length = None
        elif (
            self.synthetic_acceptance_rates is not None
            or self.synthetic_acceptance_length is not None
        ):
            raise ValueError(
                "synthetic_acceptance_rates / synthetic_acceptance_length "
                "are only valid with rejection_sample_method='synthetic'."
            )

        if self.draft_model_config:
            self.draft_model_config.verify_with_parallel_config(
                self.draft_parallel_config
            )

        self.verify_equal_vocab_size_if_draft_model()
        return self

    def verify_equal_vocab_size_if_draft_model(self):
        if (
            self.method == "draft_model"
            and self.target_model_config is not None
            and self.draft_model_config is not None
        ):
            target_vocab_size = self.target_model_config.get_vocab_size()
            draft_vocab_size = self.draft_model_config.get_vocab_size()
            if target_vocab_size != draft_vocab_size:
                raise ValueError(
                    f"Target and draft model should have the same vocabulary size. "
                    f"Target model vocab_size={target_vocab_size}. "
                    f"Draft model vocab_size={draft_vocab_size}. "
                    f"Using models with different tokenizers can cause out-of-bounds "
                    f"errors during speculative decoding."
                )

    @property
    def max_num_new_slots_for_drafting(self) -> int:
        """
        Calculate the maximum number of new slots that might be added to the batch
        when drafting.
        """
        if self.use_dflash():
            # MRV2 DFlash uses one bonus query followed by K mask queries. The
            # scheduler already owns the target query slot, so K more slots
            # must be available to the drafter.
            return self.num_speculative_tokens

        slots_per_req = 0  # for serial non-draft-model methods, no change needed
        if self.parallel_drafting:
            # For parallel drafting, we need one new slot per 'masked' token
            slots_per_req = self.num_speculative_tokens - 1
        if self.uses_draft_model():
            # For draft model-based speculation, we need one new slot per request
            # Since we do not slice the draft tokens
            slots_per_req += 1
        return slots_per_req

    @property
    def max_num_new_target_slots_for_drafting(self) -> int:
        """Return extra slots inserted into the target runner's input batch.

        MRV2 DFlash builds its K+1 query batch in a separate ``InputBuffers``
        instance. Its K draft slots therefore consume draft-runner capacity,
        not the target scheduler's prefill budget. Other proposers retain the
        existing shared-batch accounting.
        """
        if self.use_dflash():
            return 0
        return self.max_num_new_slots_for_drafting

    def use_gemma4_mtp(self) -> bool:
        return (
            self.method == "mtp"
            and self.draft_model_config is not None
            and getattr(self.draft_model_config.hf_config, "model_type", None)
            == "gemma4_mtp"
        )

    def use_step3p5_mtp(self) -> bool:
        return (
            self.method == "mtp"
            and self.draft_model_config is not None
            and getattr(self.draft_model_config.hf_config, "model_type", None)
            == "step3p5_mtp"
        )

    def use_qwen4_exp_mtp(self) -> bool:
        """Return whether Qwen4Exp needs its dedicated proposer."""
        return (
            self.method == "mtp"
            and self.draft_model_config is not None
            and getattr(self.draft_model_config.hf_config, "model_type", None)
            == "qwen4_exp_mtp"
        )

    def use_eagle(self) -> bool:
        return (
            self.method in ("eagle", "eagle3", "mtp")
            or self.use_dflash_family()
            or self.use_dspark()
        )

    def use_eagle_kv_cache(self) -> bool:
        """Whether prefix hits need Eagle's final target-block recompute."""
        # MRV2 DFlash caches its projected draft context KV alongside the
        # target KV. It only needs target hidden states for the uncached suffix,
        # so dropping an additional target block is both unnecessary and can
        # eliminate all prefix hits when hybrid page unification makes blocks
        # large. Keep DDTree and DSpark on their established V1 semantics.
        return self.use_eagle() and not self.use_dflash()

    def use_dflash(self) -> bool:
        return self.method == "dflash"

    def use_dflash_ddtree(self) -> bool:
        return self.method == "dflash_ddtree"

    def use_dflash_family(self) -> bool:
        return self.use_dflash() or self.use_dflash_ddtree()

    def use_dspark(self) -> bool:
        return self.method == "dspark"

    def num_speculative_state_tokens(self) -> int:
        num_spec_tokens = self.num_speculative_tokens or 0
        if self.use_dflash_ddtree() and not self.ddtree_disable_tree_verify:
            return max(num_spec_tokens, self.ddtree_budget or num_spec_tokens)
        return num_spec_tokens

    def uses_draft_model(self) -> bool:
        return self.method == "draft_model"

    def uses_extract_hidden_states(self) -> bool:
        return self.method == "extract_hidden_states"

    def use_ngram_gpu(self) -> bool:
        return self.method == "ngram_gpu"

    def __repr__(self) -> str:
        method = self.method
        model = (
            None
            if method
            in (
                "ngram",
                "suffix",
                "extract_hidden_states",
                "custom_class",
            )
            else self.draft_model_config.model
        )
        num_spec_tokens = self.num_speculative_tokens
        return f"SpeculativeConfig({method=}, {model=}, {num_spec_tokens=})"
