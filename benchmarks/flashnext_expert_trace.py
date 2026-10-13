# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Collect per-step routed-expert traces for sizing a GPU expert cache.

Runs the served Flash-Next configuration with ``sm70_gguf.expert_trace_tokens``
and saves, per prompt, the decode routing (verify rows, in order) and the
prefill expert histogram of every MoE layer. Routing is replicated across TP
ranks, so rank 0's trace is the model's trace.

    python benchmarks/flashnext_expert_trace.py --serve-script claude-serve.sh \
        --model M.gguf --draft mtp --output trace.pt
"""

import argparse
import json
import os

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

import torch  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402

FILLER = (
    "The history of computing spans mechanical calculators, vacuum tubes, "
    "transistors and integrated circuits. Each generation traded generality, "
    "cost and speed in different ways, and the software that ran on them "
    "evolved from hand-written machine code to high-level languages. "
)

PROMPTS = [
    (
        "chat-cpu",
        (
            "Explain how a modern out-of-order CPU executes instructions: "
            "front end, renaming, reorder buffer, reservation stations, "
            "branch prediction."
        ),
    ),
    (
        "chat-history",
        (
            "Write a detailed essay on the causes and consequences of the "
            "Industrial Revolution in Britain."
        ),
    ),
    ("chat-zh", "请详细介绍一下量子计算的基本原理、主要技术路线以及目前面临的挑战。"),
    (
        "code-py",
        (
            "Write a Python implementation of a thread-safe LRU cache with TTL "
            "expiry, including unit tests."
        ),
    ),
    (
        "code-cpp",
        (
            "Implement a lock-free single-producer single-consumer ring buffer "
            "in C++17 and explain the memory ordering."
        ),
    ),
    (
        "math",
        (
            "Prove that there are infinitely many primes, then explain the prime "
            "number theorem and its significance."
        ),
    ),
]


def long_prompt(tokens: int) -> str:
    words = (FILLER * (tokens // 40 + 2)).split()
    return " ".join(words[: int(tokens * 0.74)]) + "\nSummarize the above in detail."


def reset(self):
    from vllm.model_executor.layers.quantization.gguf_expert_trace import current_trace

    trace = current_trace()
    if trace is None:
        return False
    trace.reset()
    return True


def snapshot(self):
    from vllm.model_executor.layers.quantization.gguf_expert_trace import current_trace

    trace = current_trace()
    return None if trace is None else trace.snapshot()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve-script", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--ring-tokens", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--long", default="8000,24000")
    args = parser.parse_args()

    with open(args.serve_script) as f:
        serve = f.read()
    kc = json.loads(serve.split("KC='")[1].split("'\n")[0].replace("'$HOT'", "8192"))
    kc.setdefault("sm70_gguf", {})["expert_trace_tokens"] = args.ring_tokens
    llm = LLM(
        model=args.model,
        quantization="gguf",
        tensor_parallel_size=4,
        dtype="half",
        kv_cache_dtype="float16",
        mamba_ssm_cache_dtype="float32",
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        max_num_batched_tokens=512,
        gpu_memory_utilization=0.95,
        enable_prefix_caching=True,
        mamba_cache_mode="align",
        language_model_only=True,
        compilation_config={
            "mode": 3,
            "cudagraph_mode": "FULL",
            "cudagraph_capture_sizes": [5],
            "inductor_compile_config": {"benchmark_combo_kernel": False},
        },
        kernel_config=kc,
        speculative_config={
            "method": "mtp",
            "model": args.draft,
            "num_speculative_tokens": 4,
            "draft_load_config": {"load_format": "safetensors"},
            "draft_sample_method": "greedy",
        },
    )
    prompts = list(PROMPTS)
    for n in (int(x) for x in args.long.split(",") if x):
        prompts.append((f"long-{n}", long_prompt(n)))
    sampling = SamplingParams(temperature=0, max_tokens=args.max_tokens)
    records = []
    for label, text in prompts:
        assert all(llm.collective_rpc(reset)), "expert tracing is not active"
        message = [{"role": "user", "content": text}]
        out = llm.chat(
            message,
            sampling,
            use_tqdm=False,
            chat_template_kwargs={"enable_thinking": False},
        )[0]
        trace = llm.collective_rpc(snapshot)[0]
        records.append(
            {
                "label": label,
                "prompt_tokens": len(out.prompt_token_ids),
                "output_tokens": len(out.outputs[0].token_ids),
                **trace,
            }
        )
        print(
            label,
            records[-1]["prompt_tokens"],
            records[-1]["output_tokens"],
            "decode rows",
            int(trace["counter"][3]),
            flush=True,
        )
    torch.save(records, args.output)
    print("saved", args.output, flush=True)


if __name__ == "__main__":
    main()
