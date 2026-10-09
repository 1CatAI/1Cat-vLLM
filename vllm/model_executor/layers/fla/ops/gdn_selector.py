# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Existing GDN backend selection and its initialized stage contract."""

import functools
from dataclasses import dataclass
from typing import Literal

import torch

from vllm.config import VllmConfig
from vllm.config.gdn import GdnConfig, resolve_gdn_config
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


@functools.cache
def _is_libs_cu13_install_intact() -> bool:
    """Return True if every file installed by ``nvidia-cutlass-dsl-libs-cu13``
    matches the SHA-256 declared in its wheel ``RECORD``.

    ``nvidia-cutlass-dsl-libs-base`` and ``nvidia-cutlass-dsl-libs-cu13``
    both ship into the shared ``nvidia_cutlass_dsl/`` namespace and
    write many of the same on-disk paths (the runtime ``.so``, the MLIR
    Python bindings, cuTe-DSL Python sources, ...) with different
    content. Whichever wheel extracts last wins; with a parallel
    installer (e.g. ``uv``) the order is racy and the resulting venv
    can end up with a mix of files from both variants. The
    ``-libs-base`` variant fails MLIR legalization when JIT-compiling
    the FlashInfer Blackwell GDN prefill kernel, and any other
    cuTe-DSL-based kernel can break too if on-disk files diverge from
    what ``-libs-cu13``'s wheel expects. Tracked upstream at:

      * https://github.com/NVIDIA/cutlass/issues/3170
      * https://github.com/NVIDIA/cutlass/issues/3259

    This helper re-hashes every file the ``-libs-cu13`` wheel claims to
    own and compares against its declared SHA-256. Returns False on any
    error (uninstalled, missing RECORD, missing file, hash mismatch).
    Result is cached per-process.
    """
    import hashlib
    import importlib.metadata

    import pybase64 as base64

    try:
        dist = importlib.metadata.distribution("nvidia-cutlass-dsl-libs-cu13")
    except importlib.metadata.PackageNotFoundError:
        return False

    files = dist.files
    if not files:
        return False

    for pkg_path in files:
        file_hash = pkg_path.hash
        # Skip RECORD rows without a hash (RECORD itself, generated
        # ``.pyc`` files, ...) and any non-SHA-256 hash modes.
        if file_hash is None or not file_hash.value:
            continue
        if file_hash.mode != "sha256":
            continue
        try:
            with open(pkg_path.locate(), "rb") as f:
                digest = hashlib.sha256(f.read()).digest()
        except OSError:
            return False
        actual = base64.urlsafe_b64encode(digest).decode().rstrip("=")
        if actual != file_hash.value:
            return False

    return True


def _get_gdn_head_k_dim(vllm_config: VllmConfig) -> int | None:
    for config in (
        getattr(vllm_config.model_config, "hf_text_config", None),
        getattr(vllm_config.model_config, "hf_config", None),
    ):
        if config is None:
            continue
        head_k_dim = getattr(config, "linear_key_head_dim", None)
        if head_k_dim is not None:
            return int(head_k_dim)
    return None


def _resolve_gdn_prefill_backend(
    vllm_config: VllmConfig,
    *,
    policy: GdnConfig | None = None,
) -> tuple[str, Literal["triton", "flashinfer", "cutedsl", "flashqla_sm70"]]:
    """Resolve GDN prefill backend.

    FlashInfer's GDN prefill kernel is chosen when:
    * ``requested in ["flashinfer", "auto"]``;
    * ``platform == cuda``;
    * one of the following:
      - Hopper (SM90) — no further constraints;
      - Blackwell (SM10.x) with ``head_k_dim == 128``, ``cuda_runtime >= 13``,
        and an intact ``nvidia-cutlass-dsl-libs-cu13`` install on disk
        (see :func:`_is_libs_cu13_install_intact`).

    In-tree CuteDSL GDN prefill kernel is chosen when:
    * "cutedsl" is requested; (opt-in only)
    * Blackwell (SM10.x) with ``head_k_dim == 128``;
    """
    policy = resolve_gdn_config(vllm_config) if policy is None else policy
    assert policy.prefill_backend is not None
    backend = policy.prefill_backend

    if not current_platform.is_cuda():
        return backend, "triton"

    head_k_dim = _get_gdn_head_k_dim(vllm_config)
    model_dtype = getattr(vllm_config.model_config, "dtype", None)

    supports_flashinfer = False
    supports_cutedsl = False
    supports_flashqla_sm70 = False

    if current_platform.is_device_capability(90):
        supports_flashinfer = True
    elif head_k_dim == 128 and backend in ("auto", "flashqla_sm70"):
        capability = current_platform.get_device_capability()
        is_sm70 = (
            capability is not None and capability.major == 7 and capability.minor == 0
        )
        is_sm75 = (
            capability is not None and capability.major == 7 and capability.minor == 5
        )
        supports_model_dtype = model_dtype == torch.float16
        try:
            from flash_qla.ops.gated_delta_rule.chunk.sm70 import (  # noqa: F401
                chunk_gated_delta_rule_fwd_sm70_vlk_varlen,
            )
        except ImportError:
            supports_flashqla_sm70 = False
        else:
            supports_flashqla_sm70 = is_sm70 and supports_model_dtype
        if is_sm75:
            logger.warning_once(
                "FlashQLA-SM70 GDN prefill cannot run on Turing (sm75): the "
                "kernel asks for 86016 B of dynamic shared memory per block "
                "and Turing caps the opt-in limit at 65536 B, so the worker "
                "dies during engine init. Its VLK CUDA variant does fit but "
                "is slower than Triton/FLA from 2048 tokens per chunk "
                "upwards. Falling back to Triton/FLA."
            )
        if is_sm70 and not supports_model_dtype:
            logger.warning_once(
                "FlashQLA-SM70 GDN prefill is V100 production-validated only "
                "for fp16 model activations; model dtype %s falls back to "
                "Triton/FLA.",
                model_dtype,
            )
    elif (
        current_platform.is_device_capability_family(100)
        and head_k_dim == 128
        and current_platform.get_cuda_runtime_major() >= 13
    ):
        supports_flashinfer = _is_libs_cu13_install_intact()
        supports_cutedsl = True
        if not supports_flashinfer:
            logger.warning_once(
                "FlashInfer Blackwell GDN requires an intact nvidia-cutlass-dsl"
                "-libs-cu13 install, but some on-disk files do not match the "
                "SHA-256 declared in its RECORD (install-order race in "
                "nvidia-cutlass-dsl packaging -- see "
                "https://github.com/NVIDIA/cutlass/issues/3170 and "
                "https://github.com/NVIDIA/cutlass/issues/3259). Falling back "
                "to Triton/FLA. Repair with: pip install --force-reinstall "
                "--no-deps nvidia-cutlass-dsl-libs-cu13"
            )

    if backend in ("auto", "flashqla_sm70") and supports_flashqla_sm70:
        return backend, "flashqla_sm70"
    if backend in ["flashinfer", "auto"] and supports_flashinfer:
        return backend, "flashinfer"
    if backend == "cutedsl" and supports_cutedsl:
        return backend, "cutedsl"
    return backend, "triton"


def _log_gdn_backend_decision(
    vllm_config: VllmConfig,
    requested_backend: str,
    active_backend: str,
) -> None:
    """Log the GDN prefill backend choice in the attention-selector style."""
    head_k_dim = _get_gdn_head_k_dim(vllm_config)
    model_dtype = getattr(vllm_config.model_config, "dtype", None)
    chosen = {
        "flashinfer": "FlashInfer",
        "flashqla_sm70": "FlashQLA-SM70",
        "cutedsl": "CuteDSL",
        "triton": "Triton/FLA",
    }[active_backend]
    logger.info_once(
        "Using %s GDN prefill kernel (requested=%s, head_k_dim=%s, model_dtype=%s).",
        chosen,
        requested_backend,
        head_k_dim,
        model_dtype,
    )
    if active_backend == "flashinfer" and current_platform.is_device_capability(90):
        logger.warning_once(
            "FlashInfer GDN prefill is JIT-compiled; first run may take a "
            "while. Set --gdn-prefill-backend triton to skip JIT.",
        )


@dataclass(frozen=True)
class GdnBackendStages:
    """A backend's retained layout and numerical contract, not a launch record."""

    backend: str
    operator: str
    qk_normalization: str
    gate_conversion: str
    state_layout: str = "NHVK"
    output_layout: str = "BLHD"
    stages: tuple[str, ...] = ("recurrence",)


GDN_BACKEND_STAGES = {
    "triton": GdnBackendStages(
        "triton",
        "fla.chunk_gated_delta_rule",
        "in FLA when requested",
        "exp input -> log at input dtype; preserve FLA accumulation",
    ),
    "flashinfer": GdnBackendStages(
        "flashinfer",
        "flashinfer.gdn_prefill.chunk_gated_delta_rule",
        "external l2norm_fwd before contiguous packing",
        "exp input -> log at input dtype -> FP32 -> exp; FP32 beta/state",
    ),
    "cutedsl": GdnBackendStages(
        "cutedsl",
        "gdn_chunk_cutedsl.chunk_gated_delta_rule_cutedsl",
        "external l2norm_fwd",
        "exp input -> log at input dtype",
    ),
    "flashqla_sm70": GdnBackendStages(
        "flashqla_sm70",
        "flash_qla.chunk (original TileLang or VLK)",
        "external l2norm_fwd when requested",
        "original consumes log gate; VLK preserves gate_is_exp",
    ),
}


@dataclass(frozen=True)
class GdnExecutionPlan:
    """The existing selector's initialized result and prefill stage choices."""

    requested_backend: str
    prefill: GdnBackendStages
    original_prefill: bool
    indexed_prefill: bool
    direct_prefill_output: bool
    fallback_reason: str | None

    def explain(self) -> dict:
        from dataclasses import asdict

        return {"evidence": "static selection; not a native launch", **asdict(self)}


def select_gdn_execution(vllm_config: VllmConfig) -> GdnExecutionPlan:
    policy = resolve_gdn_config(vllm_config)
    requested, selected = _resolve_gdn_prefill_backend(vllm_config, policy=policy)
    policy.active_prefill_backend = selected
    return GdnExecutionPlan(
        requested,
        GDN_BACKEND_STAGES[selected],
        bool(policy.original_prefill),
        bool(policy.indexed_prefill),
        bool(policy.direct_prefill_output),
        "requested backend does not satisfy the retained device/dtype/package contract"
        if requested not in ("auto", selected)
        else None,
    )
