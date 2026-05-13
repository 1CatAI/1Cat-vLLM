import argparse
import json
import time

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.benchmarks.datasets import RandomDataset


def build_prompt_token_ids(
    tokenizer,
    input_tokens: int,
) -> list[int]:
    seed_text = (
        "请详细分析自回归模型、扩散模型和投机解码在推理系统中的差异、优缺点、"
        "工程实现难点，以及它们在长上下文和高吞吐服务中的适用场景。"
    )
    token_ids: list[int] = []
    while len(token_ids) < input_tokens:
        token_ids.extend(tokenizer.encode(seed_text, add_special_tokens=False))
    return token_ids[:input_tokens]


def build_random_prompt(
    tokenizer,
    input_tokens: int,
    output_tokens: int,
    seed: int,
) -> str:
    request = RandomDataset(
        random_seed=seed,
        dataset_path=None,
        disable_shuffle=False,
    ).sample(
        tokenizer=tokenizer,
        num_requests=1,
        prefix_len=0,
        input_len=input_tokens,
        output_len=output_tokens,
        range_ratio=0.0,
        request_id_prefix="local-random-",
        batchsize=1,
    )[0]
    return request.prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "dflash"), required=True)
    parser.add_argument(
        "--model",
        default="/home/test/下载/model/Qwen3.5-35B-A3B-AWQ",
    )
    parser.add_argument(
        "--draft-model",
        default="/home/test/下载/model/Qwen3.5-35B-A3B-DFlash",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--input-tokens", type=int, default=4096)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--warmup-output-tokens", type=int, default=16)
    parser.add_argument("--num-speculative-tokens", type=int, default=16)
    parser.add_argument(
        "--prompt-dataset",
        choices=("seed_text", "random"),
        default="seed_text",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attention-backend", type=str, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument(
        "--enforce-eager",
        choices=("true", "false"),
        default="false",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=None)
    parser.add_argument(
        "--disable-custom-all-reduce",
        choices=("true", "false"),
        default="false",
    )
    parser.add_argument(
        "--async-scheduling",
        choices=("true", "false"),
        default="false",
    )
    parser.add_argument("--compilation-config-json", type=str, default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if args.prompt_dataset == "random":
        prompt = build_random_prompt(
            tokenizer,
            args.input_tokens,
            args.output_tokens,
            args.seed,
        )
        prompt_token_ids = None
    else:
        prompt_token_ids = build_prompt_token_ids(tokenizer, args.input_tokens)
        prompt = {
            "prompt_token_ids": prompt_token_ids,
            "prompt": tokenizer.decode(prompt_token_ids, skip_special_tokens=False),
        }
    max_model_len = (
        args.max_model_len
        if args.max_model_len is not None
        else args.input_tokens + args.output_tokens + 256
    )
    max_num_batched_tokens = (
        args.max_num_batched_tokens
        if args.max_num_batched_tokens is not None
        else args.input_tokens + args.output_tokens + 256
    )

    llm_kwargs = dict(
        model=args.model,
        quantization="awq",
        dtype="float16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=1,
        skip_mm_profiling=True,
        enforce_eager=(args.enforce_eager == "true"),
        disable_custom_all_reduce=(args.disable_custom_all_reduce == "true"),
        async_scheduling=(args.async_scheduling == "true"),
    )
    if args.attention_backend is not None:
        llm_kwargs["attention_backend"] = args.attention_backend
    if args.kv_cache_memory_bytes is not None:
        llm_kwargs["kv_cache_memory_bytes"] = args.kv_cache_memory_bytes
    if args.compilation_config_json is not None:
        llm_kwargs["compilation_config"] = json.loads(args.compilation_config_json)
    if args.mode == "dflash":
        llm_kwargs["speculative_config"] = {
            "method": "dflash",
            "model": args.draft_model,
            "num_speculative_tokens": args.num_speculative_tokens,
        }

    llm = LLM(**llm_kwargs)

    warmup_params = SamplingParams(
        max_tokens=min(args.warmup_output_tokens, args.output_tokens),
        temperature=0.0,
    )
    llm.generate([prompt], warmup_params, use_tqdm=False)

    sampling_params = SamplingParams(
        max_tokens=args.output_tokens,
        temperature=0.0,
    )
    start = time.time()
    outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
    elapsed = time.time() - start

    output = outputs[0].outputs[0]
    result = {
        "mode": args.mode,
        "tensor_parallel_size": args.tensor_parallel_size,
        "input_tokens": (
            len(prompt_token_ids)
            if prompt_token_ids is not None
            else args.input_tokens
        ),
        "output_tokens": len(output.token_ids),
        "num_speculative_tokens": (
            args.num_speculative_tokens if args.mode == "dflash" else 0
        ),
        "elapsed_sec": elapsed,
        "output_toks_per_sec": len(output.token_ids) / elapsed if elapsed else None,
        "text_prefix": output.text[:200],
        "prompt_dataset": args.prompt_dataset,
        "token_ids": output.token_ids,
        "attention_backend": args.attention_backend,
        "enforce_eager": llm_kwargs["enforce_eager"],
        "async_scheduling": llm_kwargs["async_scheduling"],
        "disable_custom_all_reduce": llm_kwargs["disable_custom_all_reduce"],
        "max_model_len": llm_kwargs["max_model_len"],
        "max_num_batched_tokens": llm_kwargs["max_num_batched_tokens"],
        "compilation_config": llm_kwargs.get("compilation_config"),
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
