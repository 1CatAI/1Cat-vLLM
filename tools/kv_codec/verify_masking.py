# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU parity gate for masks/debug attention extracted from immutable source."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch

from vllm.v1.attention.backends.flash_v100 import masking, reference

BASE_SHA = "c4f6245f841466782752a8c3283e4727565cf17a"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    original = subprocess.check_output(
        ["git", "show", f"{BASE_SHA}:vllm/v1/attention/backends/flash_attn_v100.py"],
        cwd=root,
        text=True,
    )
    names = {
        "_cdiv_int",
        "_build_bfla_block_mask_for_seq",
        "_build_ddtree_visibility_mask",
        "_torch_attention_reference",
    }
    tree = ast.Module(
        body=[n for n in ast.parse(original).body if getattr(n, "name", None) in names],
        type_ignores=[],
    )
    baseline = {"torch": torch}
    exec(compile(tree, "immutable-baseline", "exec"), baseline)
    settings = {
        "VLLM_FLASH_V100_BFLA_POOL": "mean",
        "VLLM_FLASH_V100_BFLA_THRESHOLD": 0.1,
        "VLLM_FLASH_V100_BFLA_KEEP_MASS": 0.75,
        "VLLM_FLASH_V100_BFLA_KEEP_RATIO": 0.5,
        "VLLM_FLASH_V100_BFLA_MIN_KEEP_BLOCKS": 1,
        "VLLM_FLASH_V100_BFLA_LOCAL_BLOCKS": 1,
        "VLLM_FLASH_V100_BFLA_SPEC_STRIDE": 3,
        "VLLM_FLASH_V100_BFLA_SPEC_SEED": 19,
        "VLLM_FLASH_V100_BFLA_SPEC_PROB": 0.2,
    }
    config = SimpleNamespace(**settings)
    baseline["envs"] = config
    saved_envs = masking.envs
    masking.envs = config
    generator = torch.Generator().manual_seed(1048)
    cases = []

    def compare(name, function, **kwargs):
        expected = baseline[name](**kwargs)
        actual = function(**kwargs)
        if expected is None:
            assert actual is None, name
        else:
            assert torch.equal(expected, actual), name
        cases.append({"function": name, "bitwise_equal": True})

    try:
        with torch.no_grad():
            q = torch.randn((1, 130, 6, 8), generator=generator, dtype=torch.float16)
            cache = torch.randn((8, 64, 2, 8), generator=generator, dtype=torch.float16)
            page_row = torch.tensor([7, 1, 4, 0], dtype=torch.int32)
            for pool in ("mean", "center", "maxabs", "flat64"):
                config.VLLM_FLASH_V100_BFLA_POOL = pool
                for mass, ratio, stride, probability in (
                    (0.0, 0.0, 0, 0.0),
                    (0.75, 0.5, 3, 0.2),
                    (1.0, 0.0, 0, 0.0),
                    (0.01, 1.0, 2, 1.0),
                ):
                    config.VLLM_FLASH_V100_BFLA_KEEP_MASS = mass
                    config.VLLM_FLASH_V100_BFLA_KEEP_RATIO = ratio
                    config.VLLM_FLASH_V100_BFLA_SPEC_STRIDE = stride
                    config.VLLM_FLASH_V100_BFLA_SPEC_PROB = probability
                    compare(
                        "_build_bfla_block_mask_for_seq",
                        masking._build_bfla_block_mask_for_seq,
                        q_seq=q,
                        key_cache=cache,
                        block_table_row=page_row,
                        seq_len=250,
                        block_size=64,
                        mask_block_n=64,
                        softmax_scale=8**-0.5,
                    )
                # Invalid pool geometry and payloads must retain rejection behavior.
                for invalid in ({"mask_block_n": 0}, {"q_seq": q.float()}):
                    options = dict(
                        q_seq=q,
                        key_cache=cache,
                        block_table_row=page_row,
                        seq_len=250,
                        block_size=64,
                        mask_block_n=64,
                        softmax_scale=8**-0.5,
                    )
                    options.update(invalid)
                    compare(
                        "_build_bfla_block_mask_for_seq",
                        masking._build_bfla_block_mask_for_seq,
                        **options,
                    )
            for q_len, seq_len, prefix_len in ((0, 0, 0), (1, 9, 8), (6, 14, 8)):
                for tree_len in (0, 3, 5):
                    for parents in (None, torch.tensor([0, 0, 0, 1, 2, 3])):
                        for window in ((-1, -1), (3, 0), (2, 2)):
                            compare(
                                "_build_ddtree_visibility_mask",
                                masking._build_ddtree_visibility_mask,
                                q_len=q_len,
                                seq_len=seq_len,
                                prefix_len=prefix_len,
                                tree_len=tree_len,
                                parent_row=parents,
                                device=torch.device("cpu"),
                                window_size=window,
                            )
            for q_len, kv_len, q_heads, kv_heads in (
                (1, 8, 6, 1),
                (4, 8, 12, 2),
                (4, 4, 2, 2),
            ):
                tensors = {
                    "query": torch.randn(
                        (q_len, q_heads, 8), generator=generator
                    ).half(),
                    "key": torch.randn(
                        (kv_len, kv_heads, 8), generator=generator
                    ).half(),
                    "value": torch.randn(
                        (kv_len, kv_heads, 8), generator=generator
                    ).half(),
                }
                for causal in (False, True):
                    for window in ((-1, -1), (2, 0), (1, 1)):
                        compare(
                            "_torch_attention_reference",
                            reference._torch_attention_reference,
                            **tensors,
                            causal=causal,
                            window_size=window,
                            softmax_scale=8**-0.5,
                        )
    finally:
        masking.envs = saved_envs
    result = {
        "base_sha": BASE_SHA,
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "working_tree_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff"], cwd=root)
        ).hexdigest(),
        "source_sha256": {
            Path(module.__file__).name: hashlib.sha256(
                Path(module.__file__).read_bytes()
            ).hexdigest()
            for module in (masking, reference)
        },
        "torch": torch.__version__,
        "device": "cpu",
        "seed": 1048,
        "cases": len(cases),
        "all_bitwise_equal": all(case["bitwise_equal"] for case in cases),
        "functions": {
            name: sum(case["function"] == name for case in cases)
            for name in sorted(names - {"_cdiv_int"})
        },
        "scope": (
            "CPU masks/reference extraction; no native attention/model/performance gate"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
