# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Freeze the small quality regression set from the official dataset files.

Input names: sanitized-mbpp.json and gsm8k-test.jsonl. Keep the generated JSON
and source hashes fixed between baseline and candidate; no GPU work occurs.
"""

import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset-dir", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
root = args.dataset_dir
mbpp = json.loads((root / "sanitized-mbpp.json").read_text())[:12]
gsm = [json.loads(s) for s in (root / "gsm8k-test.jsonl").read_text().splitlines()[:12]]
cases = []
for r in mbpp:
    cases.append(
        {
            "id": f"mbpp-{r['task_id']}",
            "category": "mbpp",
            "prompt": "Write a Python function that satisfies the task and tests. "
            "Return runnable Python code only.\n\nTask: "
            + r["prompt"]
            + "\n\nTests:\n"
            + "\n".join(r["test_list"]),
            "tests": r["test_list"],
            "setup": r.get("test_setup_code", ""),
        }
    )
for i, r in enumerate(gsm):
    cases.append(
        {
            "id": f"gsm8k-{i}",
            "category": "gsm8k",
            "prompt": "Solve this problem. End your final answer with #### "
            "followed by the numerical answer.\n" + r["question"],
            "answer": r["answer"].split("####")[-1].strip(),
        }
    )
for i, (prompt, expected) in enumerate(
    [
        ("计算17乘以23，给出算式和答案。", ["391"]),
        ("中国的首都是哪个城市？", ["北京"]),
        ("苹果是水果还是动物？请简洁回答。", ["水果"]),
        ("标准大气压下，纯水的沸点是多少摄氏度？", ["100"]),
        ("原样抄写下面一行，不要改变字符：MAPLE-8261", ["MAPLE-8261"]),
        ("小明比小红高，小红比小李高。谁最高？", ["小明"]),
        ("数列2、4、6、8的下一项是什么？", ["10"]),
        ("把英文 GPU 翻译成中文。", ["图形", "图像"]),
    ]
):
    cases.append(
        {
            "id": f"zh-{i}",
            "category": "chinese",
            "prompt": prompt,
            "any_expected": expected,
        }
    )
for n in (8192, 32768, 131072, 258048):
    cases.append(
        {
            "id": f"needle-{n}",
            "category": "needle",
            "target_tokens": n,
            "depth": 0.5,
            "answer": f"CEDAR-{n}-7319",
        }
    )
report = {
    "source_sha256": {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("gsm8k-test.jsonl", "sanitized-mbpp.json")
    },
    "dataset_urls": {
        "mbpp": "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json",
        "gsm8k": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
    },
    "cases": cases,
    "sampling": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": 4096,
        "thinking": True,
        "seed": "4201 + case index",
        "ignore_eos": False,
    },
}
report["storage_policy"] = {"ngram": "disk mmap", "pinned_full_table": False}
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print("frozen cases", len(cases))
