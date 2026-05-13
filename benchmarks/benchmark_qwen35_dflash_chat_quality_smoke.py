import argparse
from collections import Counter
import hashlib
import json
import time

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams


PROMPTS = [
    {
        "id": "qa_http",
        "prompt": (
            "请用中文详细解释 HTTP/2 和 HTTP/3 的核心区别，覆盖传输层、"
            "多路复用、丢包影响和部署成本，不要输出思考过程。"
        ),
    },
    {
        "id": "coding_lru",
        "prompt": (
            "请用 Python 3 写一个线程安全的 LRUCache，支持 get、put、"
            "delete、__len__，并给出最小可运行测试，不要输出思考过程。"
        ),
    },
    {
        "id": "reasoning_schedule",
        "prompt": (
            "A、B、C、D、E 五名工程师排 5 天值班：A 不能周一和周五；"
            "B 必须比 C 早；D 不能紧挨 B；E 只能周二到周四；"
            "如果 C 在周三之后，A 必须在 C 前。给出一个满足条件的排班并简要说明。"
        ),
    },
]


def max_char_run(text: str) -> int:
    best = cur = 0
    prev = None
    for ch in text:
        if ch == prev:
            cur += 1
        else:
            cur = 1
            prev = ch
        best = max(best, cur)
    return best


def repeat_ngram_ratio(text: str, n: int = 8) -> float:
    chars = [c for c in text if not c.isspace()]
    if len(chars) < n:
        return 0.0
    grams = ["".join(chars[i : i + n]) for i in range(len(chars) - n + 1)]
    counts = Counter(grams)
    repeated = sum(v - 1 for v in counts.values() if v > 1)
    return repeated / len(grams)


def punct_ratio(text: str) -> float:
    if not text:
        return 0.0
    punct = set("!！?？。。，,、；;：:\n\t \r")
    return sum(1 for c in text if c in punct) / len(text)


def build_chat_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/ymzx/models/Qwen3.5-9B-AWQ")
    parser.add_argument("--draft-model", default="/home/ymzx/models/Qwen3.5-9B-DFlash")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-output-tokens", type=int, default=192)
    parser.add_argument("--num-speculative-tokens", type=int, default=16)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=8589934592)
    parser.add_argument("--attention-backend", default="FLASH_ATTN_V100")
    parser.add_argument(
        "--mode", choices=("baseline", "dflash", "mtp"), default="dflash"
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    prompts = [build_chat_prompt(tokenizer, row["prompt"]) for row in PROMPTS]

    llm_kwargs = dict(
        model=args.model,
        quantization="awq",
        dtype="float16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=1,
        skip_mm_profiling=True,
        attention_backend=args.attention_backend,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        async_scheduling=False,
        compilation_config={
            "cudagraph_mode": "piecewise",
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32],
        },
    )
    if args.mode == "dflash":
        llm_kwargs["speculative_config"] = {
            "method": "dflash",
            "model": args.draft_model,
            "num_speculative_tokens": args.num_speculative_tokens,
        }
    elif args.mode == "mtp":
        llm_kwargs["speculative_config"] = {
            "method": "mtp",
            "num_speculative_tokens": args.num_speculative_tokens,
        }

    llm = LLM(**llm_kwargs)
    llm.generate(
        [prompts[0]],
        SamplingParams(max_tokens=32, temperature=0.0),
        use_tqdm=False,
    )

    sampling_params = SamplingParams(max_tokens=args.max_output_tokens, temperature=0.0)
    results = []
    for row, prompt in zip(PROMPTS, prompts):
        start = time.time()
        output = llm.generate([prompt], sampling_params, use_tqdm=False)[0].outputs[0]
        elapsed = time.time() - start
        text = output.text
        results.append(
            {
                "id": row["id"],
                "prompt_tokens": len(
                    tokenizer.encode(prompt, add_special_tokens=False)
                ),
                "output_tokens": len(output.token_ids),
                "elapsed_sec": elapsed,
                "output_toks_per_sec": (
                    len(output.token_ids) / elapsed if elapsed else None
                ),
                "max_char_run": max_char_run(text),
                "repeat_8gram_ratio": repeat_ngram_ratio(text, 8),
                "punct_ratio": punct_ratio(text),
                "looks_all_punct": punct_ratio(text) > 0.8,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "token_ids": output.token_ids,
                "text_prefix": text[:500],
            }
        )

    payload = {
        "mode": args.mode,
        "tensor_parallel_size": args.tensor_parallel_size,
        "num_speculative_tokens": (
            args.num_speculative_tokens if args.mode in ("dflash", "mtp") else 0
        ),
        "results": results,
    }
    if args.out is not None:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
