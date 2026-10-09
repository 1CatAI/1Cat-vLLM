# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Initialization inputs for the existing tensor observation points."""

import os
from dataclasses import asdict
from typing import Any

from pydantic import Field

from vllm.config.utils import config


def parse_int_filter(raw: str | None, *, ranges=True, strict=False, reverse=True):
    """Shared parser; callers declare their historical failure/range semantics."""
    if not raw:
        return None
    values: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ranges and "-" in item:
            start, end = (int(part.strip()) for part in item.split("-", 1))
            if strict and (start < 0 or end < start):
                raise ValueError(f"invalid range: {item}")
            if reverse and end < start:
                start, end = end, start
            values.update(range(start, end + 1))
        else:
            value = int(item)
            if strict and value < 0:
                raise ValueError(f"invalid index: {item}")
            values.add(value)
    return frozenset(values)


def parse_margin_steps(raw: str | None) -> set[int] | None:
    if raw is None:
        return None
    steps: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start < 0 or end < start:
                raise ValueError(f"invalid top-token margin step range: {item}")
            steps.update(range(start, end + 1))
            continue
        step = int(item)
        if step < 0:
            raise ValueError(f"invalid top-token margin step: {item}")
        steps.add(step)
    return steps


def parse_token_probe(raw: str | None) -> list[int]:
    if raw is None:
        return []
    token_ids: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        token_id = int(item)
        if token_id < 0:
            raise ValueError(f"invalid top-token margin probe token: {item}")
        token_ids.append(token_id)
    return token_ids


@config
class TensorDumpConfig:
    enabled: bool | None = None
    """Optional observer admission; separate from graph-copy selection."""
    strict_fail: bool | None = None
    """Historical comparison no-op warning, retaining its original parser."""
    directory: str | None = None
    """Output directory; an empty value disables this channel."""
    enable_file: str | None = None
    """Fixed trigger path; existence remains a dynamic check."""
    layers: str | None = None
    """Legacy-compatible layer expression, parsed once during initialization."""
    labels: str | None = None
    """Comma-separated label filter, preserving the existing observation names."""
    shapes: str | None = None
    """Comma-separated dimensions, such as 1x12x128."""
    counts: str | None = None
    """Direct observation-count filter."""
    steps: str | None = None
    """Graph replay/output step filter."""
    probes: str | None = None
    """Ordered token probe ids; duplicates retain their diagnostic meaning."""
    max_dumps: int | None = None
    """Existing channel budget; callers preserve zero/unlimited semantics."""
    max_elements: int | None = None
    """Maximum copied payload elements; nonpositive keeps the whole tensor."""
    mode: str | None = None
    """Captured diagnostic synchronization mode, retaining its fallback."""
    max_tokens: int | None = None
    """Maximum leading dimension; nonpositive or absent means unrestricted."""
    capture: bool | None = None
    """Retain capture-safe diagnostic copies."""
    direct_save: bool | None = None
    """Save eager observations as well as capture buffers."""
    metadata: bool | None = None
    """Capture metadata at the existing observation point."""
    state_indices: bool | None = None
    """Capture state-index tensors at the existing observation point."""
    sources: dict[str, str] = Field(default_factory=dict, init=False)
    """Initialization provenance, excluded from all computation hashes."""
    filters: dict[str, Any] = Field(default_factory=dict, init=False)
    """Parsed immutable filters; runtime never parses legacy text."""
    filter_errors: dict[str, str] = Field(default_factory=dict, init=False)
    """Malformed strict filters fail only when their old checkpoint consumes them."""

    def resolve(self, channel: str, *, bindings=None) -> None:
        if self.sources:
            return
        bindings = DUMP_BINDINGS[channel] if bindings is None else bindings
        for field, (alias, default, parser) in bindings.items():
            if getattr(self, field) is not None:
                self.sources[field] = "typed"
                continue
            raw = os.environ.get(alias, default)
            self.sources[field] = alias if alias in os.environ else "default"
            value: Any
            if parser == "flag":
                value = raw == "1"
            elif parser == "not_zero":
                value = raw != "0"
            elif parser in ("integer", "strict_integer", "strict_flag"):
                try:
                    value = int(raw) if raw is not None else None
                    if parser == "strict_flag":
                        value = bool(value)
                except ValueError as exc:
                    if parser in ("strict_integer", "strict_flag"):
                        self.filter_errors[field] = str(exc)
                    value = int(default) if default is not None else None
            elif parser == "strip":
                assert isinstance(raw, str)
                value = raw.strip()
            else:
                value = raw
            setattr(self, field, value)
        self.parse_filters(channel)

    def parse_filters(self, channel: str) -> None:
        for field in ("labels", "shapes"):
            raw = getattr(self, field)
            self.filters[field] = (
                frozenset(
                    item.strip() for item in (raw or "").split(",") if item.strip()
                )
                or None
            )
        if self.shapes:
            self.filters["shapes"] = frozenset(
                item.strip() for item in self.shapes.split(",") if item.strip()
            )
        raw = self.layers
        self.filter_errors.pop("layers", None)
        if channel in ("qwen_layer", "awq_buffers", "awq_compare") and (
            raw or ""
        ).strip().lower() in {"*", "all"}:
            layers = None
        else:
            fallback = (
                frozenset((0, 1))
                if channel in ("qwen_layer", "gdn_projection", "awq_buffers")
                else frozenset()
            )
            if channel == "gdn_compare":
                fallback = frozenset((0,))
            try:
                layers = parse_int_filter(
                    raw,
                    ranges=channel
                    in ("qwen_layer", "gdn_graph", "awq_buffers", "awq_compare"),
                    strict=channel in ("awq_buffers", "awq_compare"),
                )
                if channel in ("qwen_layer", "awq_buffers"):
                    layers = layers or fallback
                elif channel == "gdn_projection" and raw is not None:
                    layers = layers or frozenset()
                elif channel == "gdn_compare" and (raw is None or not raw.strip()):
                    layers = fallback
            except ValueError as exc:
                if channel == "awq_compare":
                    self.filter_errors["layers"] = str(exc)
                layers = fallback
        if channel == "awq_compare" and raw == "":
            layers = frozenset()
        self.filters["layers"] = layers
        if channel == "top_token_margin":
            try:
                self.filters["probes"] = tuple(parse_token_probe(self.probes))
            except ValueError as exc:
                self.filter_errors["probes"] = str(exc)
        for field in ("steps", "counts"):
            self.filter_errors.pop(field, None)
            raw = getattr(self, field)
            if field == "steps" and channel == "qwen_layer":
                raw = raw or self.counts
            try:
                if field == "steps" and channel in ("top_token_margin", "top1_sync"):
                    self.filters[field] = parse_margin_steps(raw)
                    continue
                self.filters[field] = parse_int_filter(
                    raw,
                    ranges=channel != "gdn_compare",
                    strict=channel
                    in ("mtp_step", "sample", "sample_sync", "awq_compare"),
                    reverse=channel != "compile_inputs",
                )
                if channel == "gdn_compare" and not self.filters[field]:
                    self.filters[field] = None
                if channel == "awq_compare" and raw == "":
                    self.filters[field] = frozenset()
            except ValueError as exc:
                if channel in ("mtp_step", "sample", "sample_sync"):
                    self.filters[field] = frozenset()
                elif channel == "gdn_compare":
                    self.filters[field] = None
                else:
                    self.filter_errors[field] = str(exc)

    def report(self) -> dict:
        result = asdict(self)
        result["filters"] = {
            name: (list(value) if name == "probes" else sorted(value))
            if value is not None
            else None
            for name, value in self.filters.items()
        }
        return result

    def value(self, field: str):
        if field in self.filter_errors:
            raise ValueError(self.filter_errors[field])
        return getattr(self, field)

    def parsed(self, field: str):
        if field in self.filter_errors:
            raise ValueError(self.filter_errors[field])
        return self.filters.get(field)

    def allows(self, field: str, value) -> bool:
        if field in self.filter_errors:
            raise ValueError(self.filter_errors[field])
        selected = self.filters.get(field)
        return selected is None or value in selected

    def allows_tokens(self, tensor) -> bool:
        return (
            not self.max_tokens
            or self.max_tokens <= 0
            or tensor.ndim == 0
            or tensor.shape[0] <= self.max_tokens
        )

    def can_save(self) -> bool:
        return bool(self.directory) and (
            not self.enable_file or os.path.exists(self.enable_file)
        )


# Names are explicit so the static parameter inventory can resolve each reader.
# Format: field -> (legacy name, original raw default, original parser).
DUMP_BINDINGS: dict[str, dict[str, tuple[str, str | None, str]]] = {
    "top_token_margin": {
        "directory": ("VLLM_SM70_DUMP_TOP_TOKEN_MARGIN_DIR", None, "text"),
        "enable_file": ("VLLM_SM70_DUMP_TOP_TOKEN_MARGIN_ENABLE_FILE", None, "text"),
        "steps": ("VLLM_SM70_DUMP_TOP_TOKEN_MARGIN_STEPS", None, "text"),
        "probes": ("VLLM_SM70_DUMP_TOP_TOKEN_MARGIN_PROBE_TOKENS", None, "text"),
        "max_dumps": (
            "VLLM_SM70_DUMP_TOP_TOKEN_MARGIN_MAX_REPORTS",
            "128",
            "strict_integer",
        ),
    },
    "top1_sync": {
        "steps": ("VLLM_SM70_SYNC_TOP1_ALLGATHER_STEPS", None, "text"),
        "mode": ("VLLM_SM70_SYNC_TOP1_ALLGATHER_MODE", "stream", "text"),
    },
    "qwen_layer": {
        "directory": ("VLLM_SM70_DUMP_QWEN_LAYER_DIR", None, "text"),
        "enable_file": ("VLLM_SM70_DUMP_QWEN_LAYER_ENABLE_FILE", None, "text"),
        "layers": ("VLLM_SM70_DUMP_QWEN_LAYER_IDS", "0,1", "text"),
        "labels": ("VLLM_SM70_DUMP_QWEN_LAYER_LABELS", "", "text"),
        "counts": ("VLLM_SM70_DUMP_QWEN_LAYER_COUNTS", None, "text"),
        "steps": ("VLLM_SM70_DUMP_QWEN_LAYER_GRAPH_STEPS", None, "text"),
        "max_dumps": ("VLLM_SM70_DUMP_QWEN_LAYER_MAX_DUMPS", "4", "integer"),
        "max_tokens": ("VLLM_SM70_DUMP_QWEN_LAYER_MAX_TOKENS", None, "integer"),
        "capture": ("VLLM_SM70_DUMP_QWEN_LAYER_GRAPH_BUFFERS", None, "flag"),
        "direct_save": ("VLLM_SM70_DUMP_QWEN_LAYER_DIRECT_SAVE", "1", "not_zero"),
    },
    "gdn_core": {
        "directory": ("VLLM_SM70_DUMP_GDN_CORE_DIR", None, "text"),
        "enable_file": ("VLLM_SM70_DUMP_GDN_CORE_ENABLE_FILE", None, "text"),
        "layers": ("VLLM_SM70_DUMP_GDN_CORE_LAYER_IDS", None, "text"),
        "max_dumps": ("VLLM_SM70_DUMP_GDN_CORE_MAX_DUMPS", "4", "integer"),
    },
    "gdn_projection": {
        "directory": ("VLLM_SM70_DUMP_GDN_PROJ_DIR", None, "text"),
        "enable_file": ("VLLM_SM70_DUMP_GDN_PROJ_ENABLE_FILE", None, "text"),
        "layers": ("VLLM_SM70_DUMP_GDN_PROJ_LAYER_IDS", "0,1", "text"),
        "max_dumps": ("VLLM_SM70_DUMP_GDN_PROJ_MAX_DUMPS", "4", "integer"),
    },
    "gdn_graph": {
        "directory": ("VLLM_SM70_DUMP_GDN_GRAPH_DIR", None, "text"),
        "enable_file": ("VLLM_SM70_DUMP_GDN_GRAPH_ENABLE_FILE", None, "text"),
        "layers": ("VLLM_SM70_DUMP_GDN_GRAPH_LAYER_IDS", None, "text"),
        "labels": ("VLLM_SM70_DUMP_GDN_GRAPH_LABELS", "", "text"),
        "shapes": ("VLLM_SM70_DUMP_GDN_GRAPH_SHAPES", None, "text"),
        "steps": ("VLLM_SM70_DUMP_GDN_GRAPH_STEPS", None, "text"),
        "capture": ("VLLM_SM70_DUMP_GDN_GRAPH_BUFFERS", None, "flag"),
        "metadata": ("VLLM_SM70_DUMP_GDN_GRAPH_METADATA", None, "flag"),
        "state_indices": ("VLLM_SM70_DUMP_GDN_GRAPH_STATE_INDICES", None, "flag"),
    },
    "gdn_compare": {
        "directory": ("VLLM_SM70_COMPARE_GDN_PACKED_DECODE_DIR", None, "text"),
        "enable_file": (
            "VLLM_SM70_COMPARE_GDN_PACKED_DECODE_ENABLE_FILE",
            None,
            "text",
        ),
        "layers": ("VLLM_SM70_COMPARE_GDN_PACKED_DECODE_LAYER_IDS", None, "text"),
        "steps": ("VLLM_SM70_COMPARE_GDN_PACKED_DECODE_STEPS", None, "text"),
        "max_dumps": (
            "VLLM_SM70_COMPARE_GDN_PACKED_DECODE_MAX_REPORTS",
            "256",
            "strict_integer",
        ),
    },
    "mtp_step": {
        "directory": ("VLLM_SM70_MTP_DUMP_STEP_DIR", None, "text"),
        "steps": ("VLLM_SM70_MTP_DUMP_STEP_STEPS", None, "text"),
        "max_dumps": ("VLLM_SM70_MTP_DUMP_STEP_MAX", "512", "strict_integer"),
        "max_elements": ("VLLM_SM70_MTP_DUMP_TENSOR_MAX", "512", "strict_integer"),
    },
    "sample": {
        "directory": ("VLLM_SM70_DUMP_SAMPLE_TENSORS_DIR", None, "text"),
        "enable_file": ("VLLM_SM70_DUMP_SAMPLE_TENSORS_ENABLE_FILE", None, "text"),
        "steps": ("VLLM_SM70_DUMP_SAMPLE_TENSORS_STEPS", None, "text"),
        "max_dumps": ("VLLM_SM70_DUMP_SAMPLE_TENSORS_MAX_STEPS", "0", "strict_integer"),
    },
    "awq_buffers": {
        "enabled": ("VLLM_SM70_DUMP_AWQ_MOE_BUFFERS", None, "flag"),
        "labels": ("VLLM_SM70_DUMP_AWQ_MOE_LABELS", "", "text"),
    },
    "awq_compare": {
        "directory": ("VLLM_SM70_AWQ_MOE_COMPARE_DENSE_DIR", None, "text"),
        "enable_file": ("VLLM_SM70_AWQ_MOE_COMPARE_DENSE_ENABLE_FILE", None, "text"),
        "layers": ("VLLM_SM70_AWQ_MOE_COMPARE_DENSE_LAYER_IDS", None, "text"),
        "steps": ("VLLM_SM70_AWQ_MOE_COMPARE_DENSE_STEPS", None, "text"),
        "max_dumps": (
            "VLLM_SM70_AWQ_MOE_COMPARE_DENSE_MAX_REPORTS",
            "128",
            "strict_integer",
        ),
    },
    "fp8_compare": {
        "enabled": (
            "VLLM_SM70_FP8_MOE_LEGACY_SINGLE_TOKEN_COMPACT_COMPARE",
            "0",
            "strict_flag",
        ),
        "max_dumps": (
            "VLLM_SM70_FP8_MOE_LEGACY_SINGLE_TOKEN_COMPACT_COMPARE_REPORTS",
            "16",
            "strict_integer",
        ),
        "strict_fail": (
            "VLLM_SM70_FP8_MOE_COMPACT_STRICT_COMPARE_FAIL",
            "0",
            "strict_flag",
        ),
    },
    "sampler_logits": {
        "directory": ("VLLM_SM70_DUMP_SAMPLER_LOGITS_DIR", None, "text"),
        "enable_file": ("VLLM_SM70_DUMP_SAMPLER_LOGITS_ENABLE_FILE", None, "text"),
        "max_dumps": ("VLLM_SM70_DUMP_SAMPLER_LOGITS_MAX_STEPS", "0", "strict_integer"),
    },
    "sample_sync": {
        "steps": ("VLLM_SM70_SYNC_SAMPLE_TENSORS_STEPS", None, "text"),
        "mode": ("VLLM_SM70_SYNC_SAMPLE_TENSORS_MODE", "stream", "text"),
    },
    "compile_inputs": {
        "directory": ("VLLM_SM70_DUMP_COMPILE_GRAPH_INPUT_DIR", None, "text"),
        "steps": ("VLLM_SM70_DUMP_COMPILE_GRAPH_INPUT_STEPS", None, "text"),
    },
}


@config
class TensorDiagnosticsConfig:
    top_token_margin: TensorDumpConfig = Field(default_factory=TensorDumpConfig)
    """LM-head top-token margin, probe and report budget."""
    top1_sync: TensorDumpConfig = Field(default_factory=TensorDumpConfig)
    """Top1 exchange diagnostic synchronization; does not change selected tokens."""
    awq_buffers: TensorDumpConfig = Field(default_factory=TensorDumpConfig)
    """AWQ observation labels; directory and layer input project from qwen_layer."""
    awq_compare: TensorDumpConfig = Field(default_factory=TensorDumpConfig)
    """AWQ reference comparison filters and report budget."""
    fp8_compare: TensorDumpConfig = Field(default_factory=TensorDumpConfig)
    """FP8 compact reference comparison and legacy warning."""
    mtp_step: TensorDumpConfig = Field(default_factory=TensorDumpConfig)
    """MTP step payload observations."""
    sample: TensorDumpConfig = Field(default_factory=TensorDumpConfig)
    """Sample hidden-state/logit observations."""
    sampler_logits: TensorDumpConfig = Field(default_factory=TensorDumpConfig)
    """Classic sampler logits before processing."""
    sample_sync: TensorDumpConfig = Field(default_factory=TensorDumpConfig)
    """Explicit diagnostic synchronization points."""
    compile_inputs: TensorDumpConfig = Field(default_factory=TensorDumpConfig)
    """Compile graph input and metadata observations."""
    runner_steps: frozenset[int] | None = Field(default=None, init=False)
    """The runner's original ordered graph-step filter combination."""
    qwen_layer: TensorDumpConfig = Field(default_factory=TensorDumpConfig)
    """Shared Qwen layer and MoE-runner dump selection."""
    gdn_core: TensorDumpConfig = Field(default_factory=TensorDumpConfig)
    """Eager GDN core tensor observations."""
    gdn_projection: TensorDumpConfig = Field(default_factory=TensorDumpConfig)
    """GDN projection observations retaining non-aliasing custom-op results."""
    gdn_graph: TensorDumpConfig = Field(default_factory=TensorDumpConfig)
    """GDN tensor/metadata capture copies and post-replay output."""
    gdn_compare: TensorDumpConfig = Field(default_factory=TensorDumpConfig)
    """Packed recurrence comparison observations; calculation stays in its provider."""

    def __post_init__(self) -> None:
        for channel in DUMP_BINDINGS:
            getattr(self, channel).resolve(channel)
        self.project_shared_fields()
        try:
            self.runner_steps = parse_int_filter(
                self.qwen_layer.steps or self.qwen_layer.counts or self.gdn_graph.steps,
                strict=True,
            )
        except ValueError:
            self.runner_steps = frozenset()

    def project_shared_fields(self) -> None:
        # Shared legacy inputs have one source, but observer filter dialects differ.
        for name in ("directory", "layers"):
            setattr(self.awq_buffers, name, getattr(self.qwen_layer, name))
            self.awq_buffers.sources[name] = self.qwen_layer.sources[name]
        self.awq_buffers.parse_filters("awq_buffers")
