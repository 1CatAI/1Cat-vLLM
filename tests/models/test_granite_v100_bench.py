"""TP=2 benchmark for granite-4.1-8b-AWQ-INT4 on V100.

Two configs, one process each:
  ATTN_BACKEND=TRITON_ATTN python test_granite_v100_bench.py
  ATTN_BACKEND=FLASH_ATTN_V100 PR_CHUNKED_PREFILL_FIX=1 python test_granite_v100_bench.py

Measures:
  - Prefill latency (TTFT proxy: first generate() call wall time / prompt tokens)
  - Decode tok/s (steady-state, post-warmup)
  - Output coherence (print first 20 tokens for inspection)
"""

import os
import sys
import time
import traceback

os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRITON_CACHE_DIR", os.path.expanduser("~/.cache/triton"))
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR", os.path.expanduser("~/.cache/torchinductor")
)
os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "3000")

MODEL = "/home/admin/models/granite-4.1-8b-AWQ-INT4"
ATTN_BACKEND = os.environ.get("ATTN_BACKEND", "TRITON_ATTN")
PR_FIX = os.environ.get("PR_CHUNKED_PREFILL_FIX", "0") == "1"
DECODE_TOKENS = int(os.environ.get("DECODE_TOKENS", "128"))
TP = int(os.environ.get("TP", "2"))
BATCH = int(os.environ.get("BATCH", "1"))
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", str(max(BATCH, 1))))
MAX_NUM_BATCHED_TOKENS = int(
    os.environ.get("MAX_NUM_BATCHED_TOKENS", "4096")
)
ENFORCE_EAGER = os.environ.get("ENFORCE_EAGER", "1") == "1"


def main():
    from vllm import LLM, SamplingParams

    print(
        f"=== Granite TP={TP} bench: {ATTN_BACKEND} (PR_FIX={PR_FIX}) "
        f"BATCH={BATCH} max_num_seqs={MAX_NUM_SEQS} ===",
        flush=True,
    )

    kwargs = dict(
        model=MODEL,
        quantization="compressed-tensors",
        dtype="half",
        tensor_parallel_size=TP,
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        enforce_eager=ENFORCE_EAGER,
        attention_backend=ATTN_BACKEND,
    )
    if PR_FIX:
        # PR #25 operational note: disables chunked-prefill split for
        # FLASH_ATTN_V100 (which otherwise silently falls back).
        kwargs["compilation_config"] = {"compile_ranges_split_points": []}

    try:
        llm = LLM(**kwargs)
    except Exception as e:
        print(f"\n!!! LOAD FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)

    print("\n=== Engine constructed ===", flush=True)

    base_prompts = [
        "Write a short factual paragraph about the history of computing. "
        "Mention Babbage, Turing, and the first electronic computer. "
        "Keep it under 200 words.\n\nAnswer:",
        "Explain the difference between TCP and UDP in 100 words.\n\nAnswer:",
        "Describe how a transformer attention head works.\n\nAnswer:",
        "Summarize the plot of Hamlet in five sentences.\n\nAnswer:",
        "List the major causes of World War I in plain English.\n\nAnswer:",
        "What is the difference between mitosis and meiosis?\n\nAnswer:",
        "Describe how DNS resolution works for a typical website.\n\nAnswer:",
        "Explain entropy in thermodynamics for a curious teenager.\n\nAnswer:",
        "What is a kernel in operating-system terminology?\n\nAnswer:",
        "Describe the architecture of a modern GPU at a high level.\n\nAnswer:",
        "Explain why the sky appears blue during the day.\n\nAnswer:",
        "What is the role of hemoglobin in blood?\n\nAnswer:",
        "Describe public-key cryptography in simple terms.\n\nAnswer:",
        "Compare REST and GraphQL APIs in 100 words.\n\nAnswer:",
        "Explain what makes a programming language Turing-complete.\n\nAnswer:",
        "Describe the difference between SRAM and DRAM.\n\nAnswer:",
    ]
    prompts = [base_prompts[i % len(base_prompts)] for i in range(BATCH)]

    # Warmup: 1-token decode to amortize JIT / warm caches
    print("[warmup]", flush=True)
    llm.generate(["Hello world."], SamplingParams(temperature=0.0, max_tokens=4))

    # Bench: long decode at requested batch size
    sp = SamplingParams(temperature=0.0, max_tokens=DECODE_TOKENS,
                        ignore_eos=True)
    print(f"[bench] BATCH={BATCH}, generating {DECODE_TOKENS} tokens each",
          flush=True)
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sp)
    t1 = time.perf_counter()
    wall = t1 - t0

    n_prompt_total = sum(len(o.prompt_token_ids) for o in outputs)
    n_gen_total = sum(len(o.outputs[0].token_ids) for o in outputs)
    bang_count = sum(1 for o in outputs for tid in o.outputs[0].token_ids
                     if tid == 0)
    coherent = bang_count < n_gen_total * 0.1

    aggregate_decode_tok_s = n_gen_total / wall
    aggregate_overall_tok_s = (n_prompt_total + n_gen_total) / wall
    per_seq_decode_tok_s = aggregate_decode_tok_s / max(BATCH, 1)

    print(
        f"\n=== RESULTS ({ATTN_BACKEND}, PR_FIX={PR_FIX}, TP={TP}, "
        f"BATCH={BATCH}) ===",
        flush=True,
    )
    print(f"requests           : {len(outputs)}", flush=True)
    print(f"prompt_tokens_tot  : {n_prompt_total}", flush=True)
    print(f"gen_tokens_tot     : {n_gen_total}", flush=True)
    print(f"wall               : {wall:.2f}s", flush=True)
    print(
        f"AGG decode tok/s   : {aggregate_decode_tok_s:.2f}  "
        f"(sum across {BATCH} seqs)",
        flush=True,
    )
    print(
        f"AGG overall tok/s  : {aggregate_overall_tok_s:.2f}",
        flush=True,
    )
    print(
        f"per-seq decode tk/s: {per_seq_decode_tok_s:.2f}",
        flush=True,
    )
    print(f"\n--- output[0] preview (first 200 chars) ---", flush=True)
    print(repr(outputs[0].outputs[0].text[:200]), flush=True)
    print(
        f"\ncoherent           : {coherent}  "
        f"({bang_count}/{n_gen_total} tokens were id=0 '!')",
        flush=True,
    )


if __name__ == "__main__":
    main()
