# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
from collections.abc import Callable
from dataclasses import asdict, fields
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import Field, field_validator

from vllm.config.sm70_moe import Sm70MoEConfig
from vllm.config.sm70_native import Sm70NativeConfig
from vllm.config.sm70_runtime import Sm70RuntimeConfig
from vllm.config.utils import config, get_hash_factors, hash_factors
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


SM70_AWQ_LINEAR_ALIASES = {
    "enabled": "VLLM_SM70_AWQ_TURBOMIND",
    "prefill_exact_dense": "VLLM_SM70_AWQ_PREFILL_EXACT_DENSE",
    "fused_silu": "VLLM_SM70_AWQ_MLP_ENGINE",
}
SM70_FP8_LINEAR_ALIASES = {
    "enabled": "VLLM_SM70_FP8_TURBOMIND",
    "dequant_fallback": "VLLM_SM70_FP8_DEQUANT_FALLBACK",
    "qpn8": "VLLM_SM70_FP8_QPN8",
    "qpn8_pp2_tp4": "VLLM_SM70_FP8_QPN8_PP2_TP4",
    "qpn8_shared_gate": "VLLM_SM70_FP8_QPN8_PP2_TP4_SHARED_GATE",
    "prescaled_decode": "VLLM_SM70_FP8_PRESCALED_M1_DECODE",
    "prescaled_shared_gate": "VLLM_SM70_FP8_PRESCALED_M1_SHARED_GATE",
    "prefill_prescaled": "VLLM_SM70_FP8_PREFILL_PRESCALED",
    "prefill_exact_dense": "VLLM_SM70_FP8_PREFILL_EXACT_DENSE",
    "prefill_visible_dense_mm": "VLLM_SM70_FP8_PREFILL_VISIBLE_DENSE_MM",
    "gated_silu": "VLLM_SM70_FP8_DENSE_GATED_SILU",
    "batch_prescaled": "VLLM_SM70_FP8_BATCH_PRESCALED",
}
SM70_NVFP4_LINEAR_ALIASES = {
    "qpn2": "VLLM_SM70_NVFP4_QPN2",
    "prefill": "VLLM_SM70_NVFP4_QPN2_PREFILL",
    "shared_weight": "VLLM_SM70_NVFP4_QPN2_SHARED_WEIGHT",
    "shared_scales": "VLLM_SM70_NVFP4_QPN2_SHARED_SCALES",
    "prefill_min_m": "VLLM_SM70_NVFP4_QPN2_PREFILL_MIN_M",
}


@config
class IrOpPriorityConfig:
    """
    Configuration for vLLM IR op priority for dispatching/lowering during the
    forward pass. Each member is a list of strings, which will be installed
    in worker init via vllm.ir.ops.<op_name>.set_default().
    A single comma-separated string is accepted as well,

    If specified manually, platform defaults will be appended to the lists.
    See KernelConfig.set_platform_defaults().
    """

    rms_norm: list[str] = Field(default_factory=list)
    """Priority list for vllm.ir.ops.rms_norm"""

    fused_add_rms_norm: list[str] = Field(default_factory=list)
    """Priority list for vllm.ir.ops.fused_add_rms_norm"""

    def compute_hash(self) -> str:
        """
        Produces a hash unique to the pass configuration.
        Any new fields that affect compilation should be added to the hash.
        Any future fields that don't affect compilation should be excluded.

        Also, manually add IR op impl UUIDs to make sure they affect the compile cache.
        """
        factors = get_hash_factors(self, set())

        # Implementations are hidden from Dynamo,
        # so they don't show up in the traced files list.
        from vllm.ir.op import IrOp

        assert "_impls" not in factors
        factors["_impls"] = {
            name: {
                provider: IrOp.registry[name].impls[provider].uuid() for provider in p
            }
            for name, p in asdict(self).items()  # type: ignore[call-overload]
        }

        return hash_factors(factors)

    @field_validator("*", mode="before")
    @classmethod
    def _to_list_str(cls, value: str | list[str]):
        if isinstance(value, str):
            value = value.replace(" ", "").split(",")

        assert all(isinstance(v, str) for v in value)
        return value

    def _iter_op_priorities(self):
        """
        Yield (IrOp, priority_list) for each field, after importing platform
        kernels and validating each entry.
        """
        from vllm.ir.op import IrOp
        from vllm.platforms import current_platform

        current_platform.import_ir_kernels()

        for field in fields(self):  # type: ignore[arg-type]
            op_priority = getattr(self, field.name)
            assert op_priority is not None, (
                f"IR op priority for {field.name} must be set"
            )
            logger.debug("Setting IR op priority for %s to %s", field.name, op_priority)
            yield IrOp.registry[field.name], op_priority

    def set_default(self) -> None:
        """
        Permanently set the IR op priority for all op members.
        """
        for ir_op, op_priority in self._iter_op_priorities():
            ir_op.set_default(op_priority)

    @contextlib.contextmanager
    def set_priority(self):
        """
        Context manager to set the IR op priority for all op members.
        It also imports IR kernel implementations for the current platform
        to ensure all implementations are made available.
        """
        with contextlib.ExitStack() as stack:
            for ir_op, op_priority in self._iter_op_priorities():
                stack.enter_context(ir_op.set_priority(op_priority))
            yield

    @classmethod
    def with_default(
        cls, default: list[str], /, **kwargs: list[str]
    ) -> "IrOpPriorityConfig":
        """
        A helper to create an IrOpPriorityConfig where fields not specified in kwargs
        use the given default list.
        """
        for field in fields(cls):  # type: ignore[arg-type]
            if field.name not in kwargs:
                kwargs[field.name] = list(default)

        return cls(**kwargs)


MoEBackend = Literal[
    "auto",
    "triton",
    "deep_gemm",
    "deep_gemm_mega_moe",
    "cutlass",
    "flashinfer_trtllm",
    "flashinfer_cutlass",
    "flashinfer_cutedsl",
    "flashinfer_b12x",
    "marlin",
    "sm70_skinny",
    "humming",
    "triton_unfused",
    "aiter",
    "emulation",
]

LinearBackend = Literal[
    "auto",
    "turbomind",
    "cutlass",
    "flashinfer_cutlass",
    "flashinfer_trtllm",
    "flashinfer_cudnn",
    "marlin",
    "triton",
    "deep_gemm",
    "torch",
    "aiter",
    "machete",
    "fbgemm",
    "conch",
    "exllama",
    "emulation",
]


@config
class Sm70NvFp4Config:
    """Per-engine weight-only NVFP4 policy; None retains legacy auto selection.

    Resolve before loading layers. Native capability and local layout checks
    remain with the linear kernels. Explicit fields override deprecated envs.
    """

    active: bool = Field(default=False, init=False)
    """A loaded provider uses this format; inactive options do not salt graphs."""
    sources: dict[str, str] = Field(default_factory=dict, init=False)
    """Initialization provenance for explanations, excluded from graph hashing."""
    native: Sm70NativeConfig = Field(default_factory=Sm70NativeConfig)
    """Captured native selectors/tuning; unused formats do not affect graph keys."""
    dense_qpn2: bool = True
    """Allow native QPN2 with FP16 dense prefill on supported Turing workers."""
    qpn2: bool | None = None
    """Enable QPN2 small-M kernels; auto follows the qualified draft workload."""
    prefill: bool | None = None
    """Enable the existing bounded QPN2 prefill dispatcher."""
    shared_weight: bool | None = None
    """Share packed codes with TurboMind instead of retaining a second copy."""
    shared_scales: bool | None = None
    """Allow compact scales when batch layouts and native ABI permit them."""
    prefill_min_m: int | None = None
    """User override for the retained prefill threshold, normally 1024 rows."""
    qualified: bool = Field(default=False, init=False)
    """Whether the draft/state contract has passed the retained quality gate."""
    resolved: bool = Field(default=False, init=False)
    """Prevent reparsing process environment when a config is reused."""

    def resolve(self, *, qualified: bool, active: bool = True) -> None:
        from vllm import envs

        self.active = self.active or active
        if self.resolved:
            return
        self.qualified = qualified
        defaults = {
            "qpn2": qualified,
            "prefill": qualified,
            "shared_weight": True,
            "shared_scales": True,
            "prefill_min_m": 256,
        }
        for field, default in defaults.items():
            name = SM70_NVFP4_LINEAR_ALIASES[field]
            self.sources[field] = (
                "configuration"
                if getattr(self, field) is not None
                else name
                if envs.is_set(name)
                else "default"
            )
            if envs.is_set(name):
                logger.warning_once(
                    "%s is deprecated; use kernel_config.sm70_nvfp4.%s. "
                    "Explicit configuration takes precedence.",
                    name,
                    field,
                )
            if getattr(self, field) is None:
                setattr(
                    self, field, getattr(envs, name) if envs.is_set(name) else default
                )
        self.resolved = True


@config
class Sm70AwqConfig:
    """Per-engine AWQ policy; native support is checked by the linear kernel."""

    sources: dict[str, str] = Field(default_factory=dict, init=False)
    """Initialization provenance for explanations, excluded from graph hashing."""
    native: Sm70NativeConfig = Field(default_factory=Sm70NativeConfig)
    """Captured native selectors/tuning; unused formats do not affect graph keys."""
    enabled: bool | None = None
    """Use TurboMind AWQ; auto preserves the legacy backend preference."""
    prefill_exact_dense: bool | None = None
    """Use bounded exact-dense prefill for quality-qualified projections."""
    fused_silu: bool | None = None
    """Experimental fused gate/up epilogue; auto remains disabled."""
    resolved: bool = Field(default=False, init=False)
    """Whether the compatibility adapter has resolved this engine's policy."""

    def resolve(self) -> None:
        from vllm import envs

        if self.resolved:
            return
        for field, name in SM70_AWQ_LINEAR_ALIASES.items():
            self.sources[field] = (
                "configuration"
                if getattr(self, field) is not None
                else name
                if envs.is_set(name)
                else "default"
            )
            if envs.is_set(name):
                logger.warning_once(
                    "%s is deprecated for dense linear layers; use "
                    "kernel_config.sm70_awq.%s. Explicit configuration wins.",
                    name,
                    field,
                )
            if getattr(self, field) is None:
                value = getattr(envs, name)
                if field == "batch_prescaled":
                    value = value == "1"
                if field == "enabled":
                    value = envs.use_sm70_turbomind(value)
                setattr(self, field, value)
        self.resolved = True


@config
class Sm70Fp8Config:
    """Per-engine serialized block-FP8 variants; native tuning stays unchanged.

    None preserves the legacy default and override precedence. Resolution is
    idempotent and never writes process environment. Shared aliases still serve
    unmigrated online and compressed-tensors FP8 loaders.
    """

    sources: dict[str, str] = Field(default_factory=dict, init=False)
    """Initialization provenance for explanations, excluded from graph hashing."""
    native: Sm70NativeConfig = Field(default_factory=Sm70NativeConfig)
    """Captured native selectors/tuning; unused formats do not affect graph keys."""
    enabled: bool | None = None
    """Use TurboMind; auto retains the shared legacy backend preference."""
    block_qpn8: bool = True
    """Use native weight-only block QPN8 when its kernel capabilities match."""
    block_qpn8_volta_turbomind_prefill: bool | None = None
    """Keep a second, TurboMind-packed copy of each block QPN8 weight on Volta
    for rows beyond M=8. False holds one packed layout and serves those rows
    through the dense FP16 prefill that Turing uses. Auto keeps the copy only
    when decode can exceed 8 rows (max_num_seqs * (speculative tokens + 1)):
    below that it serves just prefill, where the dense path keeps pace."""
    dequant_fallback: bool | None = None
    """Keep the legacy dense dequantization route available when requested."""
    qpn8: bool | None = None
    """Use the retained projection-qualified QPN8 weight layout."""
    qpn8_pp2_tp4: bool | None = None
    """Use the measured serialized PP2/TP4 QPN8 pipeline variant."""
    qpn8_shared_gate: bool | None = None
    """Enable the measured non-fused shared-expert QPN8 variant."""
    prescaled_decode: bool | None = None
    """Use reversible UE8M0 scale shifts in the qualified M1 decode lane."""
    prescaled_shared_gate: bool | None = None
    """Allow the qualified shared-expert M1 prescaled variant."""
    legacy_prefill_fast_selector: bool = Field(default=True, init=False)
    """Resolved native selector, retained for existing layout admission."""
    prefill_prescaled: bool | None = None
    """Prepare the retained exact-8K pre-scaled projection variant."""
    prefill_exact_dense: bool | None = None
    """Use the bounded exact-dense prefill workspace."""
    prefill_visible_dense_mm: bool | None = None
    """Diagnostic visible dense MM for existing AsyncTP experiments."""
    legacy_grouped_bmm_decode: bool = Field(default=True, init=False)
    """Resolved native grouped decode selector used during weight preparation."""
    gated_silu: bool | None = None
    """Prepare the existing fused gate/up epilogue."""
    batch_prescaled: bool | None = None
    """Retain the channel-FP8 exact batch-scale preparation experiment."""
    explicit_enables: tuple[str, ...] = Field(default=(), init=False)
    """Retain the legacy error for an explicit route with missing native ops."""
    resolved: bool = Field(default=False, init=False)
    """Whether compatibility values have been frozen for this engine."""
    force_marlin: bool = Field(default=False, init=False)
    """Retained legacy backend rollback, resolved alongside enabled."""

    def resolve(self) -> None:
        from vllm import envs

        if self.resolved:
            return
        aliases = SM70_FP8_LINEAR_ALIASES
        explicit = []
        generic_is_auto = self.qpn8 is None
        specific_is_auto = self.qpn8_pp2_tp4 is None
        generic_override = envs.is_set(aliases["qpn8"])
        specific_override = envs.is_set(aliases["qpn8_pp2_tp4"])
        for field, name in aliases.items():
            self.sources[field] = (
                "configuration"
                if getattr(self, field) is not None
                else name
                if envs.is_set(name)
                else "default"
            )
            if envs.is_set(name):
                logger.warning_once(
                    "%s is deprecated for serialized FP8 linear layers; use "
                    "kernel_config.sm70_fp8.%s. Explicit configuration wins.",
                    name,
                    field,
                )
            value = getattr(self, field)
            if value is None:
                value = getattr(envs, name)
                if field == "batch_prescaled":
                    value = value == "1"
                if field == "enabled":
                    value = envs.use_sm70_turbomind(value)
                setattr(self, field, value)
                if envs.is_set(name) and value:
                    explicit.append(field)
            elif value:
                explicit.append(field)
        if specific_is_auto:
            # An explicit generic rollback wins over a specific legacy enable.
            if generic_override and not self.qpn8:
                self.qpn8_pp2_tp4 = False
                self.sources["qpn8_pp2_tp4"] = "qpn8 rollback: " + self.sources["qpn8"]
            elif (
                not specific_override
                and generic_override
                or not generic_is_auto
                and not specific_override
            ):
                self.qpn8_pp2_tp4 = self.qpn8
                self.sources["qpn8_pp2_tp4"] = (
                    "qpn8 inheritance: " + self.sources["qpn8"]
                )
        self.explicit_enables = tuple(explicit)
        self.force_marlin = envs.force_sm70_marlin()
        self.legacy_grouped_bmm_decode = envs.VLLM_SM70_FP8_GROUPED_BMM_DECODE
        self.legacy_prefill_fast_selector = envs.VLLM_SM70_FP8_PREFILL_FAST_SELECTOR
        self.resolved = True


@config
class Sm70GgufConfig:
    """Operation-level policy for native GGUF storage on Volta."""

    active: bool = Field(default=False, init=False)
    """A loaded provider uses this format; inactive options do not salt graphs."""
    native: Sm70NativeConfig = Field(default_factory=Sm70NativeConfig)
    """Captured native selectors/tuning; unused formats do not affect graph keys."""
    enabled: bool = True
    """Admit the packaged native extension when the operator supports the format."""

    projection_planes: bool = True
    """Use measured M8 shared-activation projection planes with canonical fallback."""

    projection_plane_scope: Literal["all", "gated_pair", "iq3_xxs"] = "all"
    """Select all planes, gated pairs, or XXS-containing layers for comparisons."""

    qkv_three_format_planes: bool = True
    """Admit measured M8 QKV planes mixing Q4_K, IQ4_XS and IQ3 formats."""

    iq2_signed_nibbles: bool = True
    """Expand IQ2 grids losslessly for qualified M8 gate/up and down shapes."""

    small_m_dp4a: bool = True
    """Use Q8_1 activations and FP32 integer dots for calibrated small GGUF batches."""

    lut4_expert_dp4a: bool = True
    """Admit canonical IQ4 gate/up integer dots at calibrated expert shapes."""

    small_m_hmma: bool = True
    """Use one coalesced integer bank and fused FP16 MMA projections at M1..8."""

    q8_expert_intermediate: bool = True
    """Encode routed intermediates once in qualified integer expert gate/up."""

    grouped_mma_gate_up: bool = False
    """Run routed IQ3 gate/up at M1..8 as expert-grouped FP16 MMA on repacked planes."""

    grouped_mma_release_raw: bool = True
    """Free the original-block IQ3 gate/up banks once the MMA planes replace them."""

    prefill_min_m: int = 8
    """Use dequantization plus tensor-core FP16 GEMM from this token count."""

    @field_validator("prefill_min_m")
    @classmethod
    def _positive_prefill_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("GGUF prefill_min_m must be positive")
        return value


@config
class Sm70RingConfig:
    """Small FP16 collectives on a verified four-GPU NVLink ring."""

    enabled: bool = True
    """Admit the ring operator when topology and peer atomics are supported."""

    max_bytes: int = Field(default=25600, gt=0, le=102400)
    """Largest calibrated input payload; larger messages retain NCCL."""


@config
class Sm70SparseConfig:
    """Per-engine sparse attention policy; individual operators guard layouts."""

    indexer_decode_cublas: bool = True
    """Share paged index keys between query heads and rows when eligible."""
    decode_bmm: bool = True
    """Gather packed FP8 keys for eligible FP16 sparse decode matmuls."""
    prefill_bmm: bool = True
    """Use bounded batched matmuls for eligible FP16 sparse prefill."""
    active: bool = Field(default=False, init=False)
    """Whether the engine's metadata describes sparse indexed attention."""
    reason: str | None = Field(default=None, init=False)
    """Startup capability rejection; individual calls also check tensor layouts."""


@config
class KernelConfig:
    """Configuration for kernel selection and warmup behavior."""

    sm70_runtime: Sm70RuntimeConfig = Field(default_factory=Sm70RuntimeConfig)
    """Per-engine auxiliary warmup policy, outside compiled computation."""

    sm70_mxfp4: Sm70NativeConfig = Field(default_factory=Sm70NativeConfig)
    """Native MXFP4 linear policy, captured only when its weights are prepared."""
    sm70_moe: Sm70MoEConfig = Field(default_factory=Sm70MoEConfig)
    """Per-engine MoE stage policy; legacy switches resolve at construction."""

    ir_op_priority: IrOpPriorityConfig = Field(default_factory=IrOpPriorityConfig)
    """
    vLLM IR op priority for dispatching/lowering during the forward pass.
    Platform defaults appended automatically during VllmConfig.__post_init__.
    """

    enable_flashinfer_autotune: bool = None  # type: ignore[assignment]
    """If True, run FlashInfer autotuning during kernel warmup."""

    moe_backend: MoEBackend = "auto"
    """Backend for MoE expert computation kernels. Available options:

    - "auto": Automatically select the best backend based on model and hardware
    - "triton": Use Triton-based fused MoE kernels
    - "deep_gemm": Use DeepGEMM kernels (FP8 block-quantized only)
    - "deep_gemm_mega_moe": Use DeepGEMM mega MoE kernels
    - "cutlass": Use vLLM CUTLASS kernels
    - "flashinfer_trtllm": Use FlashInfer with TRTLLM-GEN kernels
    - "flashinfer_cutlass": Use FlashInfer with CUTLASS kernels
    - "flashinfer_cutedsl": Use FlashInfer with CuteDSL kernels (FP4 only)
    - "flashinfer_b12x": Use FlashInfer CuteDSL fused MoE for SM12x
      (RTX Pro 6000 / DGX Spark)
    - "marlin": Use Marlin kernels (weight-only quantization)
    - "sm70_skinny": Use the skinny QPN kernels for NVFP4 and MXFP4 on SM70/SM75
      (weight-only quantization)
    - "humming": Use Humming Mixed Precision kernels
    - "triton_unfused": Use Triton unfused MoE kernels
    - "aiter": Use AMD AITer kernels (ROCm only)
    - "emulation": use BF16/FP16 GEMM, dequantizing weights and
                   running QDQ on activations.
    """

    linear_backend: LinearBackend = "auto"
    """Backend for quantized linear layer GEMM kernels. Available options:

    - "auto": Automatically select the best backend based on model and hardware
    - "turbomind": SM70 compressed-tensors NVFP4 weight-only kernels
    - "cutlass": Use CUTLASS-based kernels
    - "flashinfer_cutlass": Use FlashInfer with CUTLASS kernels
    - "flashinfer_trtllm": Use FlashInfer with TensorRT-LLM kernels
    - "flashinfer_cudnn": Use FlashInfer with cuDNN kernels
    - "marlin": Use Marlin kernels
    - "triton": Use Triton-based kernels
    - "deep_gemm": Use DeepGEMM kernels
    - "torch": Use PyTorch native scaled_mm kernels
    - "aiter": Use AMD AITer kernels (ROCm only)
    - "machete": Use Machete kernels (mixed-precision)
    - "fbgemm": Use FBGEMM kernels
    - "conch": Use Conch mixed-precision kernels
    - "exllama": Use Exllama mixed-precision kernels
    - "emulation": Use slow dequant-to-BF16 emulation (for testing only)"""

    sm70_packed_topk_gather: bool = True
    """Gather SM70 TP2/TP4 compact candidates in one lossless message."""

    sm70_decode_strategy: Literal["shared", "legacy"] = "shared"
    """Use shared FP16/E4M3 XQA partition planning when the native ABI declares
    support. E4M3 retains FP32 partials; older artifacts retain legacy adaptive
    planning with an explicit fallback. Legacy selects the retained policy."""

    sm70_fp16_grouped_short_splits: bool = True
    """Use K32 splits for FP16 q8/B1 grouped verification at 129..2048 tokens."""

    sm70_rmsnorm_gated_exact: bool | None = None
    """Native gated norm; auto follows the Flash-Next model quality boundary."""

    def resolve_sm70_rmsnorm_gated(self, *, qualified: bool) -> None:
        if self.sm70_rmsnorm_gated_exact is not None:
            return
        import os

        from vllm import envs

        name = "VLLM_SM70_RMSNORM_GATED_EXACT"
        self.sm70_rmsnorm_gated_exact = (
            bool(envs.environment_variables[name]())
            if name in os.environ
            else qualified
        )

    sm70_nvfp4: Sm70NvFp4Config = Field(default_factory=Sm70NvFp4Config)
    """SM70 compressed-tensors NVFP4 policy, resolved per engine."""

    sm70_awq: Sm70AwqConfig = Field(default_factory=Sm70AwqConfig)
    """SM70 dense AWQ policy, resolved per engine."""

    sm70_fp8: Sm70Fp8Config = Field(default_factory=Sm70Fp8Config)
    """SM70 serialized block-FP8 variant policy, resolved per engine."""

    sm70_gguf: Sm70GgufConfig = Field(default_factory=Sm70GgufConfig)
    """Native GGUF admission and Volta tensor-core prefill policy."""

    sm70_ring: Sm70RingConfig = Field(default_factory=Sm70RingConfig)
    """SM70 ring collective policy, resolved from actual peer capabilities."""

    hc_ll_optimized_loads: bool = True
    """Prefetch HC down weights and pad the up shared-memory rows on SM70."""

    hc_ll_shard: bool = True
    """Use qualified TP4 sharded HC for M1..20 with direct NVLink forwarding."""
    collective_kernel_selections: dict[str, Any] = Field(
        default_factory=dict, init=False
    )
    """Observed per-group collective capabilities and rejection reasons."""

    sm70_sparse: Sm70SparseConfig = Field(default_factory=Sm70SparseConfig)
    """SM70 sparse attention policy; admission uses actual tensor capabilities."""

    sm70_skinny_moe: bool = True
    """Admit compatible NVFP4/MXFP4 skinny MoE kernels on SM70/SM75."""

    sm70_skinny_moe_applicable: bool = Field(default=False, init=False)
    """Whether a loaded MoE family consults the skinny kernel policy."""

    moe_kernel_selections: dict[str, Any] = Field(
        default_factory=dict, init=False, repr=False
    )
    """Observed MoE capability decisions, excluded from compilation hashing."""

    fused_fp16_aux_gemv: bool = True
    """Fuse compatible auxiliary projections already using exact FP16 GEMV."""

    fused_fp16_aux_gemv_applicable: bool = Field(default=False, init=False)
    """Whether loaded auxiliary projections admit the exact GEMV fusion."""

    linear_kernel_selections: dict[str, Any] = Field(
        default_factory=dict, init=False, repr=False
    )
    """Observed selector decisions for loaded local layouts; diagnostic only."""

    sm70_fused_side_projections: bool = False
    """Compute the GDN a/b and QSA indexer q/k FP16 projections inside the
    small-M GGUF input projection launch (packed dense_mv planes)."""

    sm70_qsa_prep: bool = False
    """Fuse QSA q/k GemmaRMSNorm, partial NeoX RoPE and the FP16 K/V cache
    write into one SM70 launch for decode batches."""

    sm70_draft_hot_vocab: int = 0
    """Greedy MTP drafts choose among this many lowest token ids (BPE merge
    order) spread evenly across TP ranks; 0 keeps the full vocabulary. Only
    the draft proposal changes; verification still uses the full head."""

    sm70_draft_single_graph: bool = False
    """Capture all MTP draft decode steps, including their slot mappings and
    attention metadata, as one CUDA graph instead of one replay per step."""

    sm70_top1x: bool = False
    """Exchange TP-local greedy (value, id) pairs in two direct-NVLink hops
    instead of an all-gather."""

    sm70_greedy_verify: bool = False
    """Verify greedy MTP drafts from TP-local target argmax pairs instead of
    gathering full-vocabulary logits for the rejection sampler."""

    sm70_gdn_verify: bool = False
    """Run single-request MTP GDN verification with the SM70 sequential CUDA
    recurrence instead of the Triton fused kernel."""

    sm70_hcx: bool = False
    """Leave block outputs as TP partials and run all-reduce, HC combine/norm,
    HC down and HC up as one SM70 kernel for verification batches up to 8."""

    sm70_hcx_output_projection: bool = True
    """Fuse eligible output projections into HCX when HCX is enabled. Disable
    to compare the separate projection and HC boundary with identical weights."""

    sm70_hcx_diagnostics: bool = False
    """Record M5 HCX inputs and outputs in owned graph buffers for numerical
    diagnosis. Requires separate output projections; timings are diagnostic."""

    qsa_dense_short_context: bool = False
    """Attend densely, without index selection, in context-bucketed decode graphs
    whose bucket does not exceed the indexer budget (where QSA selects every
    visible token)."""

    qsa_auto_e4m3: bool = True
    """Default eligible calibrated QSA caches to E4M3 without speculation."""
    qsa_auto_e4m3_active: bool = Field(default=False, init=False)
    """Whether automatic calibrated QSA storage was selected."""
    qsa_auto_e4m3_reason: str | None = Field(default=None, init=False)
    """Startup reason when calibrated automatic storage cannot be selected."""

    sm70_qsa_device_history: bool = True
    """Read target device E4M3/FP16 QSA history directly with FP32 PV at
    M1..20, H6, D256. Speculative draft attention retains its existing path."""
    qsa_host_kv: bool = False
    """Keep QSA attention history in pinned host storage on SM70."""
    qsa_host_kv_dtype: Literal["fp8_e4m3", "float16"] = "fp8_e4m3"
    """Authoritative target history format; FP16 isolates placement error."""
    qsa_host_kv_draft_dtype: Literal["fp8_e4m3", "float16"] = "float16"
    """Preserve speculative cache precision independently of target storage."""
    qsa_host_kv_device_reference: bool = False
    """Keep identical encoded history on device for controlled placement A/B."""
    qsa_host_kv_hot_tokens: int = Field(default=8192, gt=0, multiple_of=16)
    """Per-layer device hot-page capacity; collisions use exact host gathers."""
    qsa_host_kv_active: bool = Field(default=False, init=False)
    """Whether the host QSA cache geometry has been admitted."""
    qsa_host_kv_reason: str | None = Field(default=None, init=False)
    """Reason the requested host QSA storage is unavailable."""

    ple_disk_cascade: bool = True
    """Allow resident FP8 PLE tiers to spill to mapped checkpoint storage."""
    ple_disk_release_pages: bool = False
    """Release file-backed PLE mappings after gathers to reduce resident RAM."""
    ple_disk_row_gather: bool = True
    """Admit byte-preserving native CPU gathers for retained mapped PLE rows."""
    ple_disk_row_readers: dict[str, Any] = Field(
        default_factory=dict, init=False, repr=False
    )
    """Observed CPU row-reader admission and startup byte/performance checks."""
    ple_disk_cascade_active: bool = Field(default=False, init=False)
    """Resolved FP8 storage, dtype and pipeline capability admission."""
    ple_disk_cascade_reason: str | None = Field(default=None, init=False)
    """Startup reason when the disk cascade cannot serve this configuration."""

    ple_pinned_decode: bool = True
    """Admit calibrated packed PLE decode rows from rank-local pinned storage."""
    ple_pinned_decode_active: bool = Field(default=False, init=False)
    """Whether every TP rank admitted the complete pinned decode table."""
    ple_pinned_decoders: dict[str, Any] = Field(
        default_factory=dict, init=False, repr=False
    )
    """Observed pinned row decoder admission, capacity and fallback reasons."""

    ple_result_transport: Literal["auto", "cuda", "mapped"] = "auto"
    """Select CPU PLE result transport by local operator/resource capability."""
    ple_result_transports: dict[str, Any] = Field(
        default_factory=dict, init=False, repr=False
    )
    """Observed per-layer result transport and small pinned-buffer sizes."""

    ple_input_prepare: bool = True
    """Fuse qualified SM70 PLE context gathering and query-boundary staging."""
    ple_input_preparations: dict[str, Any] = Field(
        default_factory=dict, init=False, repr=False
    )
    """Observed PLE input operator selection and fallback reasons."""

    @field_validator("moe_backend", mode="before")
    @classmethod
    def _normalize_moe_backend(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.lower().replace("-", "_")
        return value

    @field_validator("linear_backend", mode="before")
    @classmethod
    def _normalize_linear_backend(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.lower().replace("-", "_")
        return value

    def resolve_attention_history(self, cfg) -> bool:
        from vllm.models.qwen4_exp.common.kv_policy import resolve_qsa_host_kv

        return resolve_qsa_host_kv(cfg)

    @property
    def capture_all_draft_steps(self) -> bool:
        """Policy consumed by the generic multistep draft graph manager."""
        return self.sm70_draft_single_graph

    def top1_exchange_callback(self):
        if not self.sm70_top1x:
            return None
        from vllm.models.qwen4_exp.nvidia.sm70_hcx import maybe_top1_exchange

        return maybe_top1_exchange

    def gdn_verification_callback(self):
        if not self.sm70_gdn_verify:
            return None
        from vllm.model_executor.layers.mamba.gdn.sm70_verify import native_verifier

        return native_verifier()

    def sample_greedy(self, *args):
        from vllm.v1.worker.gpu.spec_decode.sm70_greedy_verify import (
            maybe_sample_greedy,
        )

        return maybe_sample_greedy(*args)

    def compute_hash(self) -> str:
        """
        Produces a hash unique to the pass configuration.
        Any new fields that affect compilation should be added to the hash.
        Any future fields that don't affect compilation should be excluded.
        """
        ignored_factors = {
            "enable_flashinfer_autotune",
            "sm70_runtime",  # Warmup does not alter compiled model computation.
            "ir_op_priority",  # handled separately below
            "linear_kernel_selections",
            "collective_kernel_selections",
            "moe_kernel_selections",
            "sm70_skinny_moe_applicable",
            "fused_fp16_aux_gemv_applicable",
            "ple_disk_cascade_reason",
            "ple_result_transports",
            "ple_pinned_decoders",
            "ple_input_prepare",  # Input staging is outside the compiled model.
            "ple_input_preparations",
            "ple_disk_row_gather",  # CPU-only I/O; no compiled model change
            "ple_disk_row_readers",
            "qsa_auto_e4m3_reason",
            "qsa_host_kv_reason",
        }
        if not self.sm70_skinny_moe_applicable:
            ignored_factors.add("sm70_skinny_moe")
        if not self.fused_fp16_aux_gemv_applicable:
            ignored_factors.add("fused_fp16_aux_gemv")
        if not self.qsa_auto_e4m3_active:
            ignored_factors.update({"qsa_auto_e4m3", "qsa_auto_e4m3_active"})
        if not self.ple_disk_cascade_active:
            ignored_factors.update(
                {
                    "ple_disk_cascade",
                    "ple_disk_release_pages",
                    "ple_disk_cascade_active",
                }
            )
        if not self.sm70_awq.resolved:
            # An unused format must not perturb another format's graph cache.
            ignored_factors.add("sm70_awq")
        if not self.sm70_fp8.resolved:
            ignored_factors.add("sm70_fp8")
        if not self.sm70_moe.resolved:
            ignored_factors.add("sm70_moe")
        if not self.sm70_sparse.active:
            ignored_factors.add("sm70_sparse")
        for family in ("nvfp4", "gguf"):
            policy = getattr(self, "sm70_" + family)
            if not policy.active and not policy.native.values:
                ignored_factors.add("sm70_" + family)
        if not self.sm70_mxfp4.values:
            ignored_factors.add("sm70_mxfp4")
        factors = get_hash_factors(self, ignored_factors)
        if self.sm70_moe.resolved:
            factors["sm70_moe"] = self.sm70_moe.compute_hash()
        for family in ("awq", "fp8", "nvfp4", "gguf"):
            name = "sm70_" + family
            if name in factors:
                type_name, entries = cast(Any, factors[name])
                entries = dict(entries)
                native = getattr(self, name).native
                entries.pop("native", None)
                entries.pop("active", None)
                entries.pop("sources", None)
                if native.values:
                    entries["native"] = tuple(sorted(native.hash_options().items()))
                factors[name] = (type_name, tuple(sorted(entries.items())))
        if self.sm70_mxfp4.values:
            factors["sm70_mxfp4"] = self.sm70_mxfp4.hash_options()
        factors["ir_op_priority"] = self.ir_op_priority.compute_hash()
        return hash_factors(factors)

    @field_validator("enable_flashinfer_autotune", mode="wrap")
    @classmethod
    def _skip_none_validation(cls, value: Any, handler: Callable) -> Any:
        """Skip validation if the value is `None` when initialization is delayed."""
        if value is None:
            return value
        return handler(value)

    def set_platform_defaults(self, vllm_config: "VllmConfig") -> None:
        """Set platform-specific defaults for the kernel config."""
        import torch

        from vllm.platforms import current_platform

        model = vllm_config.model_config
        text = getattr(model, "hf_text_config", None)
        self.sm70_sparse.active = bool(getattr(text, "index_head_dim", None))
        self.sm70_sparse.reason = (
            "no indexed sparse-attention metadata"
            if not self.sm70_sparse.active
            else (
                "requires CUDA compute capability 7.x"
                if not current_platform.is_cuda()
                or not current_platform.is_device_capability_family(70)
                else ("requires FP16 queries" if model.dtype != torch.float16 else None)
            )
        )

        platform_op_priority = current_platform.get_default_ir_op_priority(vllm_config)
        logger.debug(
            "Setting platform-specific IR op priority defaults: %s, user-defined: %s",
            platform_op_priority,
            self.ir_op_priority,
        )
        for op_name, op_priority in asdict(platform_op_priority).items():
            current_op_priority: list[str] = getattr(self.ir_op_priority, op_name)
            if current_op_priority is None:
                setattr(self.ir_op_priority, op_name, op_priority)
            else:
                # Append platform-specific priorities
                # Must be idempotent because vllm_config.set_platform_defaults() may be
                # called multiple times (due to VllmConfig.__post_init__ manual call).
                unique_op_priority = [
                    op for op in op_priority if op not in current_op_priority
                ]
                current_op_priority.extend(unique_op_priority)

        logger.info(
            "Final IR op priority after setting platform defaults: %s",
            self.ir_op_priority,
        )


def capture_sm70_fp8_linear_config() -> Sm70Fp8Config:
    """Capture the shared serialized/channel/ModelOpt policy during loading."""
    from vllm.config import get_current_vllm_config_or_none

    engine = get_current_vllm_config_or_none()
    policy = engine.kernel_config.sm70_fp8 if engine else Sm70Fp8Config()
    policy.resolve()
    return policy
