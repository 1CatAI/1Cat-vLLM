# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 checkpoint parity: GGUF (llama.cpp / Unsloth) vs official.

Runs a few prompts greedily through the Inkling2 model with top-5 log-probs,
then compares two runs. The UD-Q8_K_XL GGUF holds the official weights, so it
should match the official checkpoint exactly; UD-Q4_K_XL within Q4 error.

    VLLM_ENABLE_V1_MULTIPROCESSING=0 python tools/inkling2/gguf_parity.py \
        run MODEL OUT.json          # MODEL: official dir or .gguf file
    python tools/inkling2/gguf_parity.py compare A.json B.json
"""

import json
import sys

PROMPTS = [
    "The capital of France is",
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
    "Translate to German: The weather is lovely today.\nGerman:",
    "Q: What is 17 * 23?\nA:",
    "東京タワーの高さは",
    "List three prime numbers greater than 50:",
]


def run(model: str, out: str) -> None:
    from vllm import LLM, SamplingParams

    kw = dict(
        model=model,
        dtype="float16",
        kv_cache_dtype="float16",
        max_model_len=4096,
        max_num_seqs=4,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        enable_prefix_caching=False,
        language_model_only=True,  # the GGUFs carry no vision tower
    )
    if model.endswith(".gguf"):
        kw["tokenizer"] = model
    llm = LLM(**kw)
    sp = SamplingParams(temperature=0, max_tokens=24, logprobs=5)
    res = []
    for o in llm.generate(PROMPTS, sp):
        c = o.outputs[0]
        res.append(
            {
                "prompt": o.prompt,
                "prompt_ids": list(o.prompt_token_ids),
                "ids": list(c.token_ids),
                "text": c.text,
                "logprobs": [
                    {str(k): v.logprob for k, v in s.items()} for s in c.logprobs
                ],
            }
        )
    with open(out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("wrote", out)


def compare(path_a: str, path_b: str) -> None:
    with open(path_a) as fa, open(path_b) as fb:
        a, b = json.load(fa), json.load(fb)
    print(f"{path_a} vs {path_b}")
    for x, y in zip(a, b):
        assert x["prompt"] == y["prompt"]
        n = 0
        while n < min(len(x["ids"]), len(y["ids"])) and x["ids"][n] == y["ids"][n]:
            n += 1
        diffs = [
            abs(x["logprobs"][i][k] - y["logprobs"][i][k])
            for i in range(n)
            for k in x["logprobs"][i].keys() & y["logprobs"][i].keys()
        ]
        print(
            f"  prompt ids equal={x['prompt_ids'] == y['prompt_ids']}  "
            f"greedy match {n}/{len(x['ids'])}  "
            f"max |dlogprob| (top-5, aligned) {max(diffs, default=float('nan')):.4f}  "
            f"| {x['text'][:40]!r}"
        )


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] not in ("run", "compare"):
        sys.exit(__doc__)
    (run if sys.argv[1] == "run" else compare)(sys.argv[2], sys.argv[3])
