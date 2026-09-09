# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit-baseline and model-default contracts for the quality-repaired M1 lane."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXED_PROMPT = (
    "This fixed benchmark prompt is used to create a deterministic tokenized "
    "input for single-request decode measurement. "
)
LIBRARY_ENVS = {
    "hc": "VLLM_SM70_CUSTOM_AR_LIBRARY",
    "qpn": "VLLM_SM70_NVFP4_QPN_M1_LIBRARY",
    "qsa": "VLLM_SM70_QSA_TOPK_LIBRARY",
    "flashqla": "FLASH_QLA_SM70_PREBUILT_EXTENSION_PATH",
}
BASELINE_ENV = {
    name: "1"
    for name in (
        "VLLM_DISABLE_COMPILE_CACHE",
        "VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE",
        "VLLM_FLASH_V100_ROUTE_SUMMARY",
        "VLLM_PLE_CPU_OFFLOAD",
        "VLLM_PLE_DISK_OFFLOAD",
        "VLLM_QWEN3NEXT_ENABLE_SHARED_MOE_OVERLAP",
        "VLLM_SM70_FLASH_ATTN_V100",
        "VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH",
        "VLLM_SM70_FLA_RECURRENT_SCHEDULE",
        "VLLM_SM70_FUSED_SIGMOID_GATING_SCHED",
        "VLLM_SM70_GDN_CHUNK_O_SCHEDULE",
        "VLLM_SM70_GDN_DECODE_FLASHQLA",
        "VLLM_SM70_GDN_DELTA_H_SCHEDULE",
        "VLLM_SM70_GDN_KKT_SCHEDULE",
        "VLLM_SM70_GEMMA_RMS_NORM_COMPILE_NATIVE",
        "VLLM_SM70_GREEDY_TOKEN_FASTPATH",
        "VLLM_SM70_MOE_ADD_ALLREDUCE",
        "VLLM_SM70_NVFP4_MOE_GROUPED_PREFILL",
        "VLLM_SM70_NVFP4_QWEN38_MOE_FAST_PREFILL",
        "VLLM_SM70_NVFP4_QWEN38_MOE_FUSED_SWIGLU_PREFILL",
        "VLLM_SM70_NVFP4_QWEN38_MOE_INDEXED_PREFILL",
        "VLLM_SM70_NVFP4_QWEN38_MOE_QPN_M1_DECODE",
        "VLLM_SM70_NVFP4_TURBOMIND",
        "VLLM_SM70_QSA_GROUPED_PAGE4",
        "VLLM_SM70_QSA_INDEXER_CUBLAS",
        "VLLM_SM70_QSA_XQA_PAGE4",
        "VLLM_SM70_QWEN38_DUAL_COMPILE",
        "VLLM_SM70_QWEN38_FP16_GEMV",
        "VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16",
        "VLLM_SM70_QWEN38_FUSED_HC_FP16",
        "VLLM_SM70_QWEN38_HYBRID_PLE",
        "VLLM_SM70_QWEN38_ROUTER_TOPK",
        "VLLM_SM70_QWEN3NEXT_SHARED_GATE_FUSION",
        "VLLM_SM70_TP4_PUSH_ALLREDUCE",
        "VLLM_SM70_TP4_PUSH_ALLREDUCE_SUM2_M1",
        "VLLM_USE_AOT_COMPILE",
        "VLLM_USE_V2_MODEL_RUNNER",
    )
}
BASELINE_ENV.update(
    {
        name: "0"
        for name in (
            "VLLM_PLE_DISK_OFFLOAD_PROFILE",
            "VLLM_SM70_FP8_QPN8",
            "VLLM_SM70_GDN_QPN8_BA_SPLIT",
            "VLLM_SM70_GDN_RMSNORM_ONEPASS",
            "VLLM_SM70_LM_HEAD_TOP1",
            "VLLM_SM70_MTP_PROFILE",
            "VLLM_SM70_NVFP4_QPN2",
            "VLLM_SM70_NVFP4_QPN2_PREFILL",
            "VLLM_SM70_PROFILE_TRACE",
            "VLLM_SM70_QWEN4_EXP_ONLINE_QPN8",
            "VLLM_SM70_QWEN_GDN_INPUT_PROJECTION_OP",
            "VLLM_SM70_NVFP4_QWEN38_MOE_QPN_BATCH_DECODE",
        )
    }
)
BASELINE_ENV.update(
    {
        "VLLM_ENGINE_READY_TIMEOUT_S": "1200",
        "VLLM_LOGGING_LEVEL": "INFO",
        "VLLM_MQ_BROADCASTER_MAX_CHUNKS": "64",
        "VLLM_PLE_DISK_OFFLOAD_NUM_THREADS": "32",
        "VLLM_PLE_OFFLOAD_READY_TIMEOUT": "600",
        "VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS": "128",
        "VLLM_SM70_QSA_INDEXER_CUBLAS_MIN_ROWS": "512",
        "VLLM_SM70_QSA_INDEXER_CUBLAS_MIN_SCORE_ELEMENTS": "1048576",
        "VLLM_SM70_QSA_XQA_PAGE4_MIN_ROWS": "4096",
        "VLLM_SM70_QUANT_BACKEND": "turbomind",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }
)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def production_fingerprint() -> str:
    """Cover tracked Python and transitive local CUDA headers, including edits."""
    paths = (
        subprocess.check_output(
            [
                "git",
                "ls-files",
                "-z",
                "--",
                "vllm",
                "csrc",
                "flash_qla",
                "flash-attention-v100",
            ],
            cwd=ROOT,
        )
        .decode()
        .split("\0")
    )
    fingerprint = hashlib.sha256()
    for name in sorted(filter(None, paths)):
        path = ROOT / name
        if not path.is_file():
            raise ValueError(f"Missing tracked production source: {name}")
        fingerprint.update(name.encode() + b"\0" + digest(path).encode() + b"\0")
    return fingerprint.hexdigest()


def validate_bundle(bundle: Path) -> dict:
    bundle = bundle.resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    expected = {*LIBRARY_ENVS, "flashv100", "paged_kv"}
    if not manifest.get("complete") or set(manifest["libraries"]) != expected:
        raise ValueError("Runtime bundle is incomplete")
    sources = dict(manifest["flashv100_sources"])
    for entry in manifest["libraries"].values():
        path = Path(entry["path"]).resolve()
        if not path.is_relative_to(bundle) or digest(path) != entry["sha256"]:
            raise ValueError(f"Runtime binary is outside the bundle or changed: {path}")
        sources.update(entry.get("sources", {}))
    for name, expected_hash in sources.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or digest(path) != expected_hash:
            raise ValueError(f"Source changed since the runtime build: {name}")
    if manifest.get("production_fingerprint") != production_fingerprint():
        raise ValueError("Production source tree changed since the runtime build")
    return manifest


def configure_environment(
    bundle: Path, cache: Path, native_dir: Path | None, *, use_defaults: bool = False
) -> dict:
    manifest = validate_bundle(bundle)
    # Do not inherit another experiment's optional VLLM route switches.
    for name in list(os.environ):
        if name.startswith("VLLM_"):
            del os.environ[name]
    if not use_defaults:
        os.environ.update(BASELINE_ENV)
    for component, name in LIBRARY_ENVS.items():
        os.environ[name] = manifest["libraries"][component]["path"]
    if native_dir is not None:
        native_dir = native_dir.resolve()
        if not (native_dir / "_C.abi3.so").is_file():
            raise ValueError(f"Missing native vLLM extension: {native_dir}")
        os.environ["ONECAT_VLLM_EXTENSION_DIR"] = str(native_dir)
    else:
        os.environ.pop("ONECAT_VLLM_EXTENSION_DIR", None)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        map(
            str,
            (
                ROOT / "benchmarks/sm70_source_overlay",
                ROOT,
                Path(manifest["libraries"]["flashv100"]["path"]).parent,
                ROOT / "flash-attention-v100",
            ),
        )
    )
    for name, suffix in (
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
        ("TRITON_CACHE_DIR", "triton"),
        ("TORCH_EXTENSIONS_DIR", "extensions"),
        ("VLLM_CACHE_ROOT", "vllm"),
        ("XDG_CACHE_HOME", "xdg"),
    ):
        os.environ[name] = str(cache.resolve() / suffix)
    os.environ.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_MODULE_LOADING": "LAZY",
            "TORCH_CUDA_ARCH_LIST": "7.0",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "TRITON_CACHE_AUTOTUNING": "1",
            "OMP_NUM_THREADS": "1",
            "MALLOC_ARENA_MAX": "2",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return manifest


def engine_args(model: str, *, use_defaults: bool = False) -> dict:
    args = {
        "model": model,
        "tensor_parallel_size": 4,
        "dtype": "half",
        "quantization": "modelopt",
        "kv_cache_dtype": "float16",
        "max_model_len": 262144,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 1,
        "gpu_memory_utilization": 0.90,
        "seed": 0,
        "attention_backend": "FLASH_ATTN_V100",
        "language_model_only": True,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": True,
        "disable_log_stats": False,
        "distributed_executor_backend": "mp",
        "mamba_cache_mode": "align",
        "mamba_cache_dtype": "float16",
        "mamba_ssm_cache_dtype": "auto",
        "kernel_config": {
            "ir_op_priority": {
                "rms_norm": ["vllm_c", "native"],
                "fused_add_rms_norm": ["vllm_c", "native"],
            }
        },
        "compilation_config": {
            "mode": "VLLM_COMPILE",
            "cudagraph_mode": "PIECEWISE",
            "use_inductor_graph_partition": False,
            "inductor_compile_config": {
                "combo_kernels": True,
                "benchmark_combo_kernel": True,
            },
            "pass_config": {
                "fuse_norm_quant": True,
                "fuse_act_quant": True,
                "fuse_attn_quant": False,
                "fuse_allreduce_rms": False,
                "enable_sp": False,
                "fuse_gemm_comms": False,
                "fuse_rope_kvcache_cat_mla": False,
                "fuse_act_padding": False,
            },
        },
        "worker_extension_cls": "benchmarks.sm70_qwen38_baseline_worker.BaselineWorker",
    }
    if use_defaults:
        # Keep model, precision and capacity fixed; let ordinary configuration
        # select the backend, optimization flags and graph/kernel policy.
        for name in (
            "quantization",
            "attention_backend",
            "kernel_config",
            "compilation_config",
        ):
            args.pop(name)
    return args
