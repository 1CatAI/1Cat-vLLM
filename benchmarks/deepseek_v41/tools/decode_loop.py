# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Long sync decode loop for CPU profiling (prints DECODE START, then 4 x 1500 tokens)."""

import os
import time
from pathlib import Path

os.environ.setdefault("VLLM_V41_ASYNC", "0")


def main():
    from benchmarks.deepseek_v41.main_smoke import create_llm
    from benchmarks.deepseek_v41.performance import make_prompt
    from vllm import SamplingParams

    llm = create_llm(
        Path("/home/ymzx/ds41-models"),
        8192,
        Path("/home/ymzx/ds41-dev/dspark-v41"),
        0.33,
        kv_cache_mib=256,
        decode_graph=True,
        draft_graph=True,
        custom_allreduce=True,
    )
    tok = llm.get_tokenizer()
    prompt = make_prompt(tok, 128)
    sp = SamplingParams(max_tokens=1500, temperature=0, ignore_eos=True)
    llm.generate([{"prompt_token_ids": prompt}], sp, use_tqdm=False)
    print("DECODE START", flush=True)
    for i in range(4):
        t = time.perf_counter()
        out = llm.generate([{"prompt_token_ids": prompt}], sp, use_tqdm=False)
        n = len(out[0].outputs[0].token_ids)
        print(
            "rep %d tokens %d %.2f s -> %.2f ms/token"
            % (i, n, time.perf_counter() - t, (time.perf_counter() - t) / n * 1000),
            flush=True,
        )
    print("DECODE END", flush=True)


if __name__ == "__main__":
    main()
