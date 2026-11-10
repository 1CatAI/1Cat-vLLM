# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-load V4.1 batch acceptance: round cost, stage timing, real text, quality.

Order matters: unhooked performance first, then hooked stage timing, then the
real-text set (unhooked, acceptance from arrival sizes), then traced quality
and state checks. Precision is judged against a saved baseline by
v41_compare.py; this script only records.
"""

import argparse
import json
import time
from pathlib import Path

REAL_TEXT = [
    ("chat-zh", "请用三段话介绍一下长城的历史，以及它在今天的意义。"),
    ("chat-en", "Explain in a few paragraphs how vaccines train the immune system."),
    (
        "advice-zh",
        "我是刚毕业的程序员，想在三年内成长为后端架构师，请给我一份具体的学习计划。",
    ),
    (
        "code-py",
        "用Python实现一个线程安全的LRU缓存类，支持get、put和容量上限，并附上简单的单元测试。",
    ),
    (
        "code-explain",
        "Explain what the following does and point out any bugs:\n\n"
        "def merge(a, b):\n    i = j = 0\n    out = []\n    while i < len(a) and j < len(b):\n"
        "        if a[i] < b[j]:\n            out.append(a[i]); i += 1\n        else:\n"
        "            out.append(b[j]); j += 1\n    return out\n",
    ),
    (
        "math-zh",
        "一个水池有两个进水管，甲管单独注满需要6小时，乙管单独注满需要9小时。"
        "两管同时打开，多少小时能注满？请写出详细的推理过程。",
    ),
    (
        "math-en",
        "Prove that the square root of 2 is irrational. Be rigorous but concise.",
    ),
    ("story-zh", "写一个关于一只在城市里迷路的小猫找到回家路的短篇故事，约400字。"),
    (
        "translate",
        "把下面这段话翻译成英文：人工智能正在改变我们工作和生活的方式，"
        "但我们也需要认真思考它带来的伦理和社会问题。",
    ),
    (
        "json-zh",
        "为一个在线书店设计商品数据的JSON示例，包含书名、作者、价格、库存、"
        "分类和评分字段，给出三本书的数据。",
    ),
]
LONG_SOURCES = ["README.md", "docs/design/architecture/runtime_reference.md"]


def chat(tokenizer, text):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def long_prompt(tokenizer, src_root, length):
    material = "\n\n".join(
        (src_root / name).read_text(errors="ignore") for name in LONG_SOURCES
    )
    head = tokenizer.encode("下面是一份技术文档：\n\n", add_special_tokens=False)
    tail_text = "\n\n请用中文分点总结这份文档的主要内容和关键设计。"
    tail = chat(tokenizer, "__X__")
    marker = tokenizer.decode(tail)
    before_text, after_text = marker.split("__X__")
    before = tokenizer.encode(before_text, add_special_tokens=False)
    after = tokenizer.encode(tail_text + after_text, add_special_tokens=False)
    body = tokenizer.encode(material, add_special_tokens=False)
    size = length - len(before) - len(head) - len(after)
    while len(body) < size:
        body = body + body
    return before + head + body[:size] + after


def run_request(llm, prompt, maximum, logprobs):
    from vllm import SamplingParams

    engine = llm.llm_engine
    begin = time.monotonic()
    params = SamplingParams(
        max_tokens=maximum, temperature=0, seed=41, logprobs=0 if logprobs else None
    )
    engine.add_request(
        f"e2e-{time.monotonic_ns()}", {"prompt_token_ids": prompt}, params
    )
    arrivals, previous, final = [], 0, None
    while engine.has_unfinished_requests():
        for output in engine.step():
            count = len(output.outputs[0].token_ids)
            if count > previous:
                arrivals.append(
                    dict(seconds=time.monotonic() - begin, tokens=count - previous)
                )
                previous = count
            if output.finished:
                final = output
    completion = final.outputs[0]
    lps = None
    if logprobs and completion.logprobs is not None:
        lps = [
            float(entry[tok].logprob) if entry and tok in entry else None
            for tok, entry in zip(completion.token_ids, completion.logprobs)
        ]
    first, last = arrivals[0]["seconds"], arrivals[-1]["seconds"]
    return dict(
        prompt_tokens=len(prompt),
        token_ids=list(completion.token_ids),
        text=completion.text,
        finish_reason=completion.finish_reason,
        ttft_seconds=first,
        tpot_seconds=(last - first) / (previous - 1) if previous > 1 else None,
        arrivals=arrivals,
        logprobs=lps,
    )


def _validate(validate_commits, ranks, expected_end):
    """Per-commit invariants always; under async scheduling the request's
    final in-flight speculative step may commit up to one round past the end."""
    import os

    if os.environ.get("VLLM_V41_ASYNC", "0") != "1":
        return validate_commits(ranks, expected_end)
    validate_commits(ranks, None)
    for rank in ranks:
        if not rank["commits"]:
            raise AssertionError("Request has no state commits")
        end = rank["commits"][-1]["hash_history"][3]
        if not expected_end <= end <= expected_end + 6:
            raise AssertionError((rank["rank"], expected_end, end))


def acceptance(arrivals):
    # Arrival 0 is the prefill token; the final arrival may be truncated.
    sizes = [a["tokens"] for a in arrivals[1:-1]]
    if not sizes:
        return None
    hits = [sum(1 for s in sizes if s - 1 >= i) / len(sizes) for i in range(1, 6)]
    return dict(
        rounds=len(sizes),
        mean_accepted=sum(s - 1 for s in sizes) / len(sizes),
        position_rates=hits,
    )


def main(args):
    import profile_decode as pd

    from benchmarks.deepseek_v41 import trace
    from benchmarks.deepseek_v41.acceptance import check_quality, validate_commits
    from benchmarks.deepseek_v41.main_smoke import (
        _memory_report,
        create_llm,
        trim_allocator,
    )
    from benchmarks.deepseek_v41.performance import make_prompt
    from benchmarks.deepseek_v41.resources import ResourceMonitor

    monitor = ResourceMonitor()
    report = dict(
        status="running",
        label=args.label,
        perf={},
        stages=None,
        real_text=[],
        quality=[],
        errors=[],
    )

    def save():
        report["resources"] = monitor.report()
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
        tmp.replace(args.output)

    llm = None
    try:
        begin = time.monotonic()
        llm = create_llm(
            args.model_root,
            args.chunk,
            args.draft,
            args.offload,
            kv_cache_mib=args.kv_mib,
            decode_graph=True,
            draft_graph=True,
            custom_allreduce=True,
        )
        report["load_seconds"] = time.monotonic() - begin
        llm.collective_rpc(trim_allocator)
        tok = llm.get_tokenizer()
        save()

        # 1. Round cost on the counting load (unhooked).
        for length in args.lengths:
            prompt = make_prompt(tok, length)
            for _ in range(2):
                run_request(llm, prompt, 128, False)
            report["perf"][str(length)] = [
                run_request(llm, prompt, 128, False) for _ in range(3)
            ]
            save()

        # 2. Per-stage GPU timing (hooked), 128 input.
        prompt = make_prompt(tok, args.lengths[0])
        llm.collective_rpc(pd.install)
        stage_runs = []
        for _ in range(2):
            run_request(llm, prompt, 128, False)
            stage_runs.append(llm.collective_rpc(pd.drain))
        llm.collective_rpc(pd.uninstall)
        report["stages"] = stage_runs
        save()

        # 3. Real text: acceptance from arrival sizes, logprobs for precision.
        quick_names = (
            "chat-zh",
            "chat-en",
            "code-py",
            "math-zh",
            "story-zh",
            "json-zh",
        )
        cases = [
            (n, chat(tok, q))
            for n, q in REAL_TEXT
            if not args.quick or n in quick_names
        ]
        for length in args.long_lengths:
            cases.append((f"longdoc-{length}", long_prompt(tok, args.src_root, length)))
        for name, prompt in cases:
            try:
                result = run_request(llm, prompt, args.real_tokens, True)
            except Exception as error:  # logprobs unsupported under spec decode
                report["errors"].append(f"{name}: logprobs failed: {error!r}")
                result = run_request(llm, prompt, args.real_tokens, False)
            result["name"] = name
            result["acceptance"] = acceptance(result["arrivals"])
            report["real_text"].append(result)
            save()

        # 4. Quality and committed-state checks (traced).
        llm.collective_rpc(trace.install)
        prompts = {
            "math": "只输出计算结果：17乘以23等于多少？",
            "chat": "用一句话解释为什么月亮会有阴晴圆缺。",
            "code": "只输出Python代码，写一个返回两个整数之和的add函数。",
            "json": "只输出有效JSON，内容是城市北京、温度20，键为city和temperature。",
        }
        for name, question in prompts.items():
            prompt = chat(tok, question)
            result = run_request(llm, prompt, 96, False)
            ranks = llm.collective_rpc(trace.drain)
            entry = dict(name=name, text=result["text"], ok=True)
            try:
                _validate(
                    validate_commits, ranks, len(prompt) + len(result["token_ids"]) - 1
                )
                check_quality(name, result["text"])
            except AssertionError as error:
                entry.update(ok=False, error=repr(error)[:2000])
            report["quality"].append(entry)
            save()
        if args.context_16k:
            source = tok.encode(
                "这是一段用于验证模型的测试上下文。" * 2048, add_special_tokens=False
            )
            question = tok.encode(
                "\n请用中文简短回答：2+2等于几？\n答：", add_special_tokens=False
            )
            prompt = source[: 16384 - 32 - len(question)] + question
            result = run_request(llm, prompt, 32, False)
            ranks = llm.collective_rpc(trace.drain)
            entry = dict(
                name="context-16k",
                text=result["text"],
                ok=True,
                ttft_seconds=result["ttft_seconds"],
            )
            try:
                _validate(
                    validate_commits, ranks, len(prompt) + len(result["token_ids"]) - 1
                )
            except AssertionError as error:
                entry.update(ok=False, error=repr(error)[:2000])
            report["quality"].append(entry)
            save()
        report["memory_by_rank"] = llm.collective_rpc(_memory_report)
        report["status"] = "completed"
        save()
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        save()
        raise
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
        monitor.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-root", type=Path, default=Path("/home/ymzx/ds41-models"))
    p.add_argument("--draft", type=Path, default=Path("/home/ymzx/ds41-dev/dspark-v41"))
    p.add_argument("--src-root", type=Path, default=Path("/home/ymzx/ds41-dev/src"))
    p.add_argument("--lengths", type=int, nargs="+", default=[128, 2048])
    p.add_argument("--long-lengths", type=int, nargs="*", default=[2048, 8192])
    p.add_argument("--real-tokens", type=int, default=256)
    p.add_argument("--context-16k", action="store_true")
    p.add_argument("--offload", type=float, default=0.30)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--kv-mib", type=int, default=256)
    p.add_argument(
        "--quick",
        action="store_true",
        help="dev mode: 128-token perf, 6 short real-text cases, no long inputs",
    )
    p.add_argument("--label", default="")
    p.add_argument("--output", type=Path, required=True)
    parsed = p.parse_args()
    if parsed.quick:
        parsed.lengths = [128]
        parsed.long_lengths = []
        parsed.context_16k = False
    main(parsed)
