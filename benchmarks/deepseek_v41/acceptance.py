# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-request V4.1 qualification, matched greedy paths and an optional soak.

The capped draft supplies an M1 target correctness reference while retaining
the same weights. It is not a no-speculative performance baseline; run without
--draft separately for that baseline. Forced proposals test state selectors,
not the trained draft's acceptance rate.
"""

import argparse
import ast
import json
import time
from pathlib import Path

from benchmarks.deepseek_v41 import trace
from benchmarks.deepseek_v41.main_smoke import (
    _install_finite_checks,
    _memory_report,
    create_llm,
    trim_allocator,
)
from benchmarks.deepseek_v41.resources import ResourceMonitor


def validate_commits(records, expected_end=None):
    """All recurrent sources must stop at the same accepted input boundary."""
    for rank in records:
        for entry in rank["commits"]:
            selected = entry["rows"] if entry["prefill"] else entry["accepted_inputs"]
            end = entry["start"] + selected
            if (
                not 0 < selected <= entry["rows"]
                or entry["hash_history"][3] != end
                or entry["hash_metadata"][2] != 0
                or entry["c2_positions"] != [end] * 3
            ):
                raise AssertionError((rank["rank"], entry))
        if expected_end is not None:
            if not rank["commits"]:
                raise AssertionError("Request has no state commits")
            last = rank["commits"][-1]
            if last["hash_history"][3] != expected_end:
                raise AssertionError((rank["rank"], expected_end, last))


def request(
    llm,
    prompt,
    maximum,
    *,
    seed=41,
    temperature=0,
    ignore_eos=False,
    skip_special_tokens=True,
):
    from vllm import SamplingParams

    engine = llm.llm_engine
    begin = time.monotonic()
    engine.add_request(
        f"v41-{time.monotonic_ns()}",
        {"prompt_token_ids": prompt},
        SamplingParams(
            max_tokens=maximum,
            temperature=temperature,
            seed=seed,
            ignore_eos=ignore_eos,
            skip_special_tokens=skip_special_tokens,
        ),
    )
    arrivals, previous, final = [], 0, None
    while engine.has_unfinished_requests():
        for output in engine.step():
            now = time.monotonic() - begin
            count = len(output.outputs[0].token_ids)
            if count > previous:
                arrivals.append(dict(seconds=now, tokens=count - previous))
                previous = count
            if output.finished:
                final = output
    elapsed = time.monotonic() - begin
    if final is None or not arrivals:
        raise RuntimeError("V4.1 request produced no completion")
    completion = final.outputs[0]
    first, last = arrivals[0]["seconds"], arrivals[-1]["seconds"]
    return dict(
        started_monotonic=begin,
        prompt_tokens=len(prompt),
        token_ids=list(completion.token_ids),
        text=completion.text,
        finish_reason=completion.finish_reason,
        wall_seconds=elapsed,
        ttft_seconds=first,
        tpot_seconds=(last - first) / (previous - 1) if previous > 1 else None,
        output_tokens_per_second=previous / elapsed,
        arrivals=arrivals,
    )


def _unfence(text):
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1]).strip()
    return text


def check_quality(name, text):
    text = _unfence(text)
    if name == "math":
        assert text == "391", text
    elif name == "chat":
        assert len(text) > 5 and "月" in text, text
    elif name == "json":
        assert json.loads(text) == {"city": "北京", "temperature": 20}, text
    elif name == "code":
        functions = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)]
        function = next(n for n in functions if n.name == "add")
        returned = function.body[-1]
        assert len(function.args.args) == 2 and isinstance(returned, ast.Return), text
        value = returned.value
        assert isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add), text
        assert isinstance(value.left, ast.Name) and isinstance(value.right, ast.Name)
        assert {value.left.id, value.right.id} == {a.arg for a in function.args.args}


def run(args):
    begin = time.monotonic()
    resources = ResourceMonitor()
    llm = None
    report = dict(
        status="running",
        profile=(
            "fp16-tensorcore-dp4a-dspark5-graph"
            if args.decode_graph
            else "fp16-dense-tail-fp32-draft-sum"
        ),
        dense_prefill_min_m=1,
        chunk=args.chunk,
        soak_requested_seconds=args.soak_seconds,
        dspark=args.draft is not None,
        offload_fraction=args.offload,
        correctness_checks=True,
        target_graph=args.decode_graph,
        draft_graph=args.draft_graph,
        custom_allreduce=args.custom_ar,
        cases=[],
    )

    def save():
        report["resources"] = resources.report()
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(args.output)

    def execute(name, prompt, maximum, **kwargs):
        result = request(llm, prompt, maximum, **kwargs)
        result["name"] = name
        result["trace_by_rank"] = llm.collective_rpc(trace.drain)
        validate_commits(
            result["trace_by_rank"], len(prompt) + len(result["token_ids"]) - 1
        )
        if args.decode_graph:
            for rank in result["trace_by_rank"]:
                small = [row for row in rank["target_forwards"] if row["rows"] <= 6]
                assert all(row["mode"] == "FULL" for row in small), (name, rank)
        if args.custom_ar:
            assert all(
                rank["custom_allreduce_enabled"] for rank in result["trace_by_rank"]
            )
        result["memory_by_rank"] = llm.collective_rpc(_memory_report)
        report["cases"].append(result)
        save()
        print(
            json.dumps(
                {
                    k: v
                    for k, v in result.items()
                    if k not in ("trace_by_rank", "memory_by_rank", "arrivals")
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return result

    try:
        llm = create_llm(
            args.model_root,
            args.chunk,
            args.draft,
            args.offload,
            decode_graph=args.decode_graph,
            draft_graph=args.draft_graph,
            custom_allreduce=args.custom_ar,
        )
        report["load_seconds"] = time.monotonic() - begin
        llm.collective_rpc(trim_allocator)
        llm.collective_rpc(_install_finite_checks)
        llm.collective_rpc(trace.install)
        tokenizer = llm.get_tokenizer()
        if args.context_checks:
            question = tokenizer.encode(
                "\n请用中文简短回答：2+2等于几？\n答：", add_special_tokens=False
            )
            source = tokenizer.encode(
                "这是一段用于验证模型的测试上下文。" * 2048, add_special_tokens=False
            )
            for context in (64, 2048, 16384):
                maximum = 8 if context == 64 else 32
                length = context - maximum
                prompt = source[: length - len(question)] + question
                result = execute(
                    f"context-{context}", prompt, maximum, ignore_eos=context != 64
                )
                if context != 64 and len(result["token_ids"]) != maximum:
                    raise AssertionError("Context-bound probe ended too early")
        prompts = {
            "math": "只输出计算结果：17乘以23等于多少？",
            "chat": "用一句话解释为什么月亮会有阴晴圆缺。",
            "code": "只输出Python代码，写一个返回两个整数之和的add函数。",
            "json": "只输出有效JSON，内容是城市北京、温度20，键为city和temperature。",
        }
        for name, question in prompts.items():
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if args.draft is not None:
                llm.collective_rpc(trace.set_draft_cap, args=(0,))
                baseline = execute(name + "-target-m1", prompt, 96)
                llm.collective_rpc(trace.set_draft_cap, args=(None,))
                actual = execute(name + "-dspark5", prompt, 96)
                if actual["token_ids"] != baseline["token_ids"]:
                    raise AssertionError(f"Greedy M1/M6 divergence in {name}")
            else:
                actual = execute(name + "-no-spec", prompt, 96)
            check_quality(name, actual["text"])

        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
        )
        from vllm.tool_parsers.deepseekv41_tool_parser import DeepSeekV41ToolParser

        question = "查询北京的天气。必须调用get_weather工具，city参数设为北京。"
        tools = [
            dict(
                type="function",
                function=dict(
                    name="get_weather",
                    description="查询指定城市的天气",
                    parameters=dict(
                        type="object",
                        properties=dict(city=dict(type="string")),
                        required=["city"],
                    ),
                ),
            )
        ]
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tool_result = execute("tool-format", prompt, 128, skip_special_tokens=False)
        tool_request = ChatCompletionRequest(
            model="ds41", messages=[dict(role="user", content=question)], tools=tools
        )
        parsed = DeepSeekV41ToolParser(
            tokenizer, tool_request.tools
        ).extract_tool_calls(tool_result["text"], tool_request)
        assert parsed.tools_called and len(parsed.tool_calls) == 1, tool_result["text"]
        call = parsed.tool_calls[0].function
        assert call.name == "get_weather" and json.loads(call.arguments) == {
            "city": "北京"
        }

        if args.draft is not None and args.force_selectors:
            filler = tokenizer.encode(
                "甲乙丙丁春夏秋冬。" * 64, add_special_tokens=False
            )
            question = tokenizer.encode(
                "\n请依次列出从一到十的中文数字：一、", add_special_tokens=False
            )
            for length in (127, 128):
                prompt = filler[: length - len(question)] + question
                llm.collective_rpc(trace.set_draft_cap, args=(0,))
                baseline = execute(f"selector-reference-{length}", prompt, 16)
                if len(baseline["token_ids"]) < 7:
                    raise AssertionError("Selector reference ended before seven tokens")
                llm.collective_rpc(trace.set_draft_cap, args=(None,))
                for rejected in range(6):
                    llm.collective_rpc(
                        trace.force_draft, args=(baseline["token_ids"], rejected)
                    )
                    actual = execute(
                        f"selector-{length}-{rejected}", prompt, rejected + 4
                    )
                    if actual["token_ids"] != baseline["token_ids"][: rejected + 4]:
                        raise AssertionError(
                            "Forced proposal changed the greedy output"
                        )
                    for rank in actual["trace_by_rank"]:
                        decode = [c for c in rank["commits"] if not c["prefill"]]
                        if not decode or decode[0]["accepted_inputs"] != rejected + 1:
                            raise AssertionError((rank["rank"], rejected, decode))
                        if len(decode) < 2:
                            raise AssertionError(
                                "No continuation after forced rejection"
                            )
                llm.collective_rpc(trace.force_draft, args=(None, None))

        soak_begin = time.monotonic()
        deadline = soak_begin + args.soak_seconds
        iteration = 0
        while time.monotonic() < deadline:
            question = f"只输出结果：{iteration % 17 + 1}+2等于多少？"
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            actual = execute(
                f"soak-{iteration}",
                prompt,
                32,
                seed=41 + iteration,
                temperature=0.8 if iteration % 4 == 0 else 0,
            )
            assert actual["text"].strip() == str(iteration % 17 + 3), actual["text"]
            iteration += 1
        if resources.report()["host_minimum_available_bytes"] < 8 * 1024**3:
            raise MemoryError("Acceptance breached the 8 GiB host reserve")
        report.update(
            status="completed",
            soak_requests=iteration,
            soak_elapsed_seconds=time.monotonic() - soak_begin,
        )
        save()
    except BaseException as exception:
        report.update(status="failed", error=repr(exception))
        save()
        raise
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
        resources.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--draft", type=Path)
    parser.add_argument("--chunk", type=int, choices=(512, 1024, 2048), default=1024)
    parser.add_argument(
        "--offload", type=float, choices=(0.20, 0.25, 0.30), default=0.30
    )
    parser.add_argument("--force-selectors", action="store_true")
    parser.add_argument("--decode-graph", action="store_true")
    parser.add_argument("--draft-graph", action="store_true")
    parser.add_argument("--custom-ar", action="store_true")
    parser.add_argument("--context-checks", action="store_true")
    parser.add_argument("--soak-seconds", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
