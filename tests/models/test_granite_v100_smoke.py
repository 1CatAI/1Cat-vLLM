"""Smoke test: load cyankiwi/granite-4.1-8b-AWQ-INT4 on V100.

Probes whether the existing TurboMindAsymLinearKernel handles
compressed-tensors W4 gs=32 asymmetric (granite's quant). Run as a
script (not pytest), like:

    cd /tmp && PATH=$PATH:/usr/local/cuda-12.8/bin \
        /home/admin/venv/bin/python /home/admin/vllm-v100/tests/models/test_granite_v100_smoke.py
"""

import os
import sys
import traceback

os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRITON_CACHE_DIR", os.path.expanduser("~/.cache/triton"))
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR", os.path.expanduser("~/.cache/torchinductor")
)
os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "3000")

MODEL = "/home/admin/models/granite-4.1-8b-AWQ-INT4"


def main():
    from vllm import LLM, SamplingParams

    print(f"=== Granite 4.1-8B AWQ-INT4 smoke test ===", flush=True)
    print(f"model: {MODEL}", flush=True)

    try:
        llm = LLM(
            model=MODEL,
            quantization="compressed-tensors",
            dtype="half",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.80,
            max_model_len=2048,
            max_num_seqs=1,
            enforce_eager=True,
            attention_backend="TRITON_ATTN",
        )
    except Exception as e:
        print(f"\n!!! LOAD FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)

    print("\n=== Engine constructed; running 1 generation ===", flush=True)

    sp = SamplingParams(temperature=0.0, max_tokens=32)
    outputs = llm.generate(["Q: What is 2+3? A:"], sp)
    for o in outputs:
        print(f"PROMPT: {o.prompt!r}", flush=True)
        print(f"OUTPUT: {o.outputs[0].text!r}", flush=True)

    print("\n=== SMOKE TEST PASSED ===", flush=True)


if __name__ == "__main__":
    main()
