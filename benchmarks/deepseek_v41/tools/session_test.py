# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""G7 session test: turn 2 continues turn 1 (raw token concat); TTFT and output tokens."""

import json
import os
import sys
import time
from pathlib import Path


def main(out_path):
    from v41_e2e import chat, long_prompt

    from benchmarks.deepseek_v41.main_smoke import create_llm
    from vllm import SamplingParams

    llm = create_llm(
        Path("/home/ymzx/ds41-models"),
        8192,
        Path("/home/ymzx/ds41-dev/dspark-v41"),
        0.33,
        kv_cache_mib=288,
        decode_graph=True,
        draft_graph=True,
        custom_allreduce=True,
    )
    tok = llm.get_tokenizer()
    engine = llm.llm_engine

    def run(prompt, name, n):
        engine.add_request(
            name,
            {"prompt_token_ids": prompt},
            SamplingParams(max_tokens=n, temperature=0),
        )
        t = time.perf_counter()
        first = None
        ids = []
        while engine.has_unfinished_requests():
            for o in engine.step():
                if o.outputs and o.outputs[0].token_ids and first is None:
                    first = time.perf_counter() - t
                if o.finished:
                    ids = list(o.outputs[0].token_ids)
        return first, ids

    report = {"prefix": os.environ.get("VLLM_V41_SESSION_PREFIX", "0")}
    doc = long_prompt(tok, Path("/home/ymzx/ds41-dev/src"), 6000)
    for name, p1 in (("short", chat(tok, "请用三句话介绍一下长城。")), ("long", doc)):
        t1, o1 = run(p1, name + "-1", 120)
        follow = tok.encode(
            "\n\n用户：请把上面的内容再简要总结成一句话。\n\n助手：",
            add_special_tokens=False,
        )
        p2 = p1 + o1 + follow
        t2, o2 = run(p2, name + "-2", 64)
        t3, o3 = run(p2 + o2 + follow, name + "-3", 32)
        report[name] = dict(
            turn1=[len(p1), round(t1, 2)],
            turn2=[len(p2), round(t2, 2)],
            turn3=[len(p2 + o2 + follow), round(t3, 2)],
            out2=o2,
            out3=o3,
        )
        print(
            name,
            report[name]["turn1"],
            report[name]["turn2"],
            report[name]["turn3"],
            flush=True,
        )
    Path(out_path).write_text(json.dumps(report))


if __name__ == "__main__":
    main(sys.argv[1])
