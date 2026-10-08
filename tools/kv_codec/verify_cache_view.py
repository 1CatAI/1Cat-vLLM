# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare extracted page views, admission and aliasing with immutable source."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch

from vllm.v1.attention.backends.flash_v100 import cache_view

BASE_SHA = "c4f6245f841466782752a8c3283e4727565cf17a"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    original = subprocess.check_output(
        ["git", "show", f"{BASE_SHA}:vllm/v1/attention/backends/flash_attn_v100.py"],
        cwd=root,
        text=True,
    )
    names = {
        "_split_paged_kv_cache",
        "_same_storage",
        "_contiguous_paged_start_block",
        "_contiguous_paged_kv_view",
        "_contiguous_paged_kv_bhmd",
        "_as_flash_v100_metadata",
    }
    tree = ast.Module(
        body=[n for n in ast.parse(original).body if getattr(n, "name", None) in names],
        type_ignores=[],
    )
    # Metadata casting is type-only; both arms use fresh metadata/cache objects.
    baseline = {
        "torch": torch,
        "cast": lambda _, value: value,
        "FlashAttnV100Metadata": SimpleNamespace,
    }
    exec(compile(tree, "immutable-baseline", "exec"), baseline)
    count = 0

    def compare(name, **kwargs):
        nonlocal count
        outputs = []
        states = []
        for fn in (baseline[name], getattr(cache_view, name)):
            metadata = SimpleNamespace()
            if "attn_metadata" in kwargs:
                kwargs["attn_metadata"] = metadata
            # Exercise the memoized hit/rejection as well as the first call.
            outputs.append((fn(**kwargs), fn(**kwargs)))
            states.append(getattr(metadata, "flash_v100_contig_dense_cache", None))
        assert states[0] == states[1], (name, kwargs)
        for expected, actual in zip(*outputs):
            if (
                expected is None
                or isinstance(expected, (int, bool))
                or isinstance(expected, tuple)
                and isinstance(expected[0], int)
            ):
                assert expected == actual
            else:
                assert len(expected) == len(actual)
                for old, new in zip(expected, actual):
                    assert torch.equal(old, new), name
                    assert old.shape == new.shape and old.stride() == new.stride()
                    assert old.storage_offset() == new.storage_offset()
                    # Keep zero-copy view/copy behavior, not just tensor values.
                    for value in kwargs.values():
                        if isinstance(value, torch.Tensor):
                            assert cache_view._same_storage(old, value) == (
                                cache_view._same_storage(new, value)
                            )
        count += 1

    for dtype in (torch.float16, torch.uint8):
        storage = (
            torch.arange(6 * 2 * 4 * 2 * 8, device=args.device)
            .reshape(6, 2, 4, 2, 8)
            .to(dtype)
        )
        for cache in (storage, storage.transpose(0, 1), list(storage.unbind(1))):
            compare("_split_paged_kv_cache", kv_cache=cache)
        for separate in (False, True):
            key, value = storage.unbind(1)
            if separate:
                key, value = key.contiguous(), value.contiguous()
            for blocks, length in (
                ([1, 2, 3], 9),
                ([1, 3, 2], 9),
                ([-1, 0, 1], 9),
                ([5, 6, 7], 9),
                ([1], 9),
                ([1], 0),
            ):
                kwargs = dict(
                    key_cache=key,
                    block_table_row=torch.tensor(
                        blocks, dtype=torch.int32, device=args.device
                    ),
                    seq_len=length,
                    block_size=4,
                    attn_metadata=None,
                    seq_idx=0,
                )
                compare("_contiguous_paged_start_block", **kwargs)
                kwargs["value_cache"] = value
                compare("_contiguous_paged_kv_bhmd", **kwargs)
                for allow_copy in (False, True):
                    compare(
                        "_contiguous_paged_kv_view", **kwargs, allow_copy=allow_copy
                    )
    result = dict(
        base_sha=BASE_SHA,
        source_sha=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        cache_view_sha256=hashlib.sha256(
            Path(cache_view.__file__).read_bytes()
        ).hexdigest(),
        torch=torch.__version__,
        device=args.device,
        cases=count,
        bitwise_and_aliasing_equal=True,
        scope="page views only; not attention/model or performance admission",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
