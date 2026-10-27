# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Long-context check: needle retrieval, TTFT and decode speed at large prompt lengths."""

import json
import os
import sys
import time
from pathlib import Path


def main(out_path, lengths):
    from v41_e2e import long_prompt

    from benchmarks.deepseek_v41.main_smoke import create_llm
    from vllm import SamplingParams

    llm = create_llm(
        Path("/home/ymzx/ds41-models"),
        8192,
        Path("/home/ymzx/ds41-dev/dspark-v41"),
        0.33,
        kv_cache_mib=int(os.environ.get("KV_MIB", "512")),
        decode_graph=True,
        draft_graph=True,
        custom_allreduce=True,
    )
    tok = llm.get_tokenizer()
    engine = llm.llm_engine
    report = []
    for n in lengths:
        needle = "【重要】本次测试的秘密口令是：翡翠-7381-QX。请记住它。"
        question = "\n\n问题：上文中提到的秘密口令是什么？请只回答口令本身。"
        body = long_prompt(tok, Path("/home/ymzx/ds41-dev/src"), n)
        text = tok.decode(body)
        depth = len(text) // 3
        text = text[:depth] + "\n" + needle + "\n" + text[depth:]
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": text + question}],
            tokenize=True,
            add_generation_prompt=True,
        )
        engine.add_request(
            f"long-{n}",
            {"prompt_token_ids": prompt},
            SamplingParams(max_tokens=48, temperature=0),
        )
        t = time.perf_counter()
        first = None
        out = None
        stamps = []
        while engine.has_unfinished_requests():
            for o in engine.step():
                now = time.perf_counter()
                if o.outputs and o.outputs[0].token_ids:
                    if first is None:
                        first = now - t
                    stamps.append((now, len(o.outputs[0].token_ids)))
                if o.finished:
                    out = o.outputs[0]
        answer = out.text
        decode = (
            (stamps[-1][1] - stamps[0][1]) / (stamps[-1][0] - stamps[0][0])
            if len(stamps) > 1
            else 0
        )
        entry = dict(
            prompt_tokens=len(prompt),
            ttft=round(first, 2),
            decode_tok_s=round(decode, 1),
            found="7381" in answer,
            answer=answer[:80],
        )
        # Follow-up turn: prefix-cache hit at long context + a long decode.
        follow = tok.encode(
            "\n\n请用中文写一段约 300 字的说明，介绍这份文档的主要内容。",
            add_special_tokens=False,
        )
        p2 = prompt + list(out.token_ids) + follow
        engine.add_request(
            f"long2-{n}",
            {"prompt_token_ids": p2},
            SamplingParams(max_tokens=300, temperature=0, ignore_eos=True),
        )
        t = time.perf_counter()
        first2 = None
        stamps = []
        while engine.has_unfinished_requests():
            for o in engine.step():
                now = time.perf_counter()
                if o.outputs and o.outputs[0].token_ids:
                    if first2 is None:
                        first2 = now - t
                    stamps.append((now, len(o.outputs[0].token_ids)))
        if len(stamps) > 2:
            entry["turn2_ttft"] = round(first2, 2)
            entry["turn2_decode_tok_s"] = round(
                (stamps[-1][1] - stamps[0][1]) / (stamps[-1][0] - stamps[0][0]), 1
            )
            entry["turn2_ms_per_round"] = round(
                (stamps[-1][0] - stamps[0][0]) * 1000 / max(1, len(stamps) - 1), 1
            )
        report.append(entry)
        print("LONG", json.dumps(entry, ensure_ascii=False), flush=True)
        Path(out_path).write_text(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in sys.argv[2:]])
