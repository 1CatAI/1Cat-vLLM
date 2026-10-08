# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare storage budgets with the immutable pre-refactor source on CPU."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
BASELINE = "c4f6245f841466782752a8c3283e4727565cf17a"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    from vllm.v1 import kv_cache_interface as candidate

    path = args.out / "baseline_kv_cache_interface.py"
    path.write_bytes(
        subprocess.check_output(
            ["git", "show", f"{BASELINE}:vllm/v1/kv_cache_interface.py"], cwd=ROOT
        )
    )
    spec = importlib.util.spec_from_file_location("baseline_kv_cache_interface", path)
    assert spec is not None and spec.loader is not None
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    count = 0
    for kind in (
        "AttentionSpec",
        "FullAttentionSpec",
        "SlidingWindowSpec",
        "TQFullAttentionSpec",
        "MLAAttentionSpec",
        "SlidingWindowMLASpec",
        "CircularBufferSpec",
    ):
        for dtype, mode in (
            (torch.float16, 0),
            (torch.bfloat16, 0),
            (torch.float32, 0),
            (torch.uint8, 1),
            (torch.int8, 2),
            (torch.uint8, 3),
            (torch.uint8, 4),
        ):
            for block in (16, 784, 2048):
                for heads in (1, 2, 4):
                    for dim in (64, 128, 256):
                        for padding in (None, 1 << 26):
                            kwargs = dict(
                                block_size=block,
                                num_kv_heads=heads,
                                head_size=dim,
                                dtype=dtype,
                                page_size_padded=padding,
                            )
                            if kind in ("SlidingWindowSpec", "SlidingWindowMLASpec"):
                                kwargs["sliding_window"] = 8192
                            old = getattr(baseline, kind)(
                                **kwargs, kv_quant_mode=baseline.KVQuantMode(mode)
                            )
                            new = getattr(candidate, kind)(
                                **kwargs, kv_quant_mode=candidate.KVQuantMode(mode)
                            )
                            assert old.page_size_bytes == new.page_size_bytes, (
                                kind,
                                kwargs,
                                mode,
                            )
                            assert old.real_page_size_bytes == new.real_page_size_bytes
                            count += 1
    paths = (
        "vllm/v1/kv_cache_codec.py",
        "vllm/v1/kv_cache_interface.py",
        "vllm/v1/attention/backends/triton_attn.py",
    )
    result = {
        "baseline": BASELINE,
        "cases": count,
        "page_and_payload_sizes_equal": True,
        "scope": "CPU accounting; does not qualify model or GPU paths",
        "torch": torch.__version__,
        "source_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in paths
        },
    }
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
