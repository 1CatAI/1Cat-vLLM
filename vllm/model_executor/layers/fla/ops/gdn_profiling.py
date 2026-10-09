# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One GDN prefill timing budget per engine; no execution-time env reads."""

import os
import time

import torch

from vllm.config.gdn import GdnProfileConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


class GdnPrefillProfiler:
    def __init__(self, config: GdnProfileConfig):
        assert config.max_logs is not None and config.max_per_stage is not None
        self.config = config
        self.max_logs = config.max_logs
        self.max_per_stage = config.max_per_stage
        self.counts: dict[str, int] = {}

    def enabled(self) -> bool:
        return (
            bool(self.config.enabled)
            and not torch.compiler.is_compiling()
            and not (
                torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
            )
        )

    def start(self) -> float | None:
        if not self.enabled():
            return None
        torch.accelerator.synchronize()
        return time.perf_counter()

    def end(self, layer_name, stage, start, *, tokens=None, details="") -> None:
        if start is None:
            return
        torch.accelerator.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        total = self.counts.get("__total__", 0)
        if total >= self.max_logs:
            return
        key = f"{os.getpid()}:{layer_name}:{stage}"
        count = self.counts.get(key, 0)
        if count >= self.max_per_stage:
            return
        self.counts[key] = count + 1
        self.counts["__total__"] = total + 1
        logger.info(
            "SM70 GDN prefill profile: layer=%s stage=%s elapsed_ms=%.3f tokens=%s %s",
            layer_name,
            stage,
            elapsed_ms,
            tokens,
            details,
        )


def bind_gdn_profiler(vllm_config) -> GdnPrefillProfiler:
    """Share the runtime budget across layers, separate from serialized policy.

    This private runtime attachment is created only while binding layers. It is
    not a dataclass field and is absent from configuration JSON and graph hashes.
    """
    owner = getattr(vllm_config, "_gdn_prefill_profiler", None)
    if owner is None:
        observability = getattr(vllm_config, "observability_config", None)
        config = getattr(observability, "gdn_profile", None) or GdnProfileConfig()
        config.resolve()
        owner = GdnPrefillProfiler(config)
        vllm_config._gdn_prefill_profiler = owner
    return owner
