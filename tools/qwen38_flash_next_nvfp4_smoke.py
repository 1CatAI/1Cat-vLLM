#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal Qwen3.8-Flash-Next NVFP4 smoke test for 4x V100.

This intentionally tests loader compatibility and a minimal prefill/decode.  The
ModelOpt NVFP4 MoE backend is emulated on SM70, so this is not a performance
benchmark.
"""

from __future__ import annotations

import argparse
import os
import time


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--prompt", default="你好")
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=2)
    args = parser.parse_args()

    os.environ.setdefault("VLLM_PLE_CPU_OFFLOAD", "1")
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from vllm import LLM, SamplingParams

    log("constructing TP4 engine")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=4,
        dtype="half",
        trust_remote_code=True,
        language_model_only=True,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=1,
        # NVFP4 emulation transiently dequantizes a full expert tensor and
        # needs about 1.56 GiB beyond the steady-state model allocation.  Keep
        # only a smoke-sized KV cache so that workspace remains available.
        gpu_memory_utilization=0.98,
        kv_cache_memory_bytes=256 << 20,
        enforce_eager=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        moe_backend="emulation",
    )
    log(f"engine ready; generating {args.max_tokens} token(s)")
    outputs = llm.generate(
        [args.prompt], SamplingParams(temperature=0, max_tokens=args.max_tokens)
    )
    text = outputs[0].outputs[0].text
    token_ids = outputs[0].outputs[0].token_ids
    log(f"SMOKE_OK token_ids={token_ids!r} text={text!r}")


if __name__ == "__main__":
    main()
