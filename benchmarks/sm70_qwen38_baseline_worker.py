# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only runtime provenance for the single-request reproduction driver."""

import os
import sys
from pathlib import Path

from benchmarks.sm70_qwen38_baseline import LIBRARY_ENVS, ROOT, digest


class BaselineWorker:
    def baseline_manifest(self):
        import torch
        from flash_attn_v100 import flash_attn_interface

        import vllm

        paths = {key: os.environ[name] for key, name in LIBRARY_ENVS.items()}
        paths["flashv100"] = flash_attn_interface.flash_attn_v100_cuda.__file__
        mapped = Path("/proc/self/maps").read_text()
        binaries = {
            key: {
                "path": str(Path(path).resolve()),
                "sha256": digest(Path(path)),
                "mapped": str(Path(path).resolve()) in mapped,
            }
            for key, path in paths.items()
        }
        native = {
            name: {"path": module.__file__, "sha256": digest(Path(module.__file__))}
            for name, module in list(sys.modules.items())
            if name.startswith("vllm.")
            and str(getattr(module, "__file__", "")).endswith(".so")
        }
        cfg = self.vllm_config
        return {
            "rank": int(self.rank),
            "pid": os.getpid(),
            "torch": str(torch.__version__),
            "cuda_runtime": str(torch.version.cuda),
            "source_root": str(ROOT),
            "vllm_file": vllm.__file__,
            "ssm_dtype": str(cfg.cache_config.mamba_ssm_cache_dtype),
            "kv_dtype": str(cfg.cache_config.cache_dtype),
            "prefix_cache": bool(cfg.cache_config.enable_prefix_caching),
            "mtp": cfg.speculative_config is not None,
            "max_model_len": int(cfg.model_config.max_model_len),
            "graph_mode": str(cfg.compilation_config.cudagraph_mode),
            "mamba_cache_mode": str(cfg.cache_config.mamba_cache_mode),
            "resolved_vllm_env": {
                key: value
                for key, value in os.environ.items()
                if key.startswith("VLLM_")
            },
            "compilation_config": str(cfg.compilation_config),
            "binaries": binaries,
            "native_dependencies": native,
            "qsa_specialization_version": (
                torch.ops._C_qsa_sm70.decode_specialization_version()
            ),
        }
