# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare existing mapped-gather and pinned-host DMA QSA staging paths.

The benchmark retains the original pinned tensor and switches only the
existing HostQSAKV host-view branch. It does not modify serving or create a
new operator. Exact bytes/scales are required for every staged valid page.
Foreground timings include four HC-shaped GEMMs concurrent with one stage
copy and its completion wait. They are a synthetic interference screen.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm.models.qwen4_exp.nvidia.ops.host_kv import HostQSAKV
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor


class CopyCalls(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.completed = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if str(func) == "vllm_sm70_qsa_device.stage_host_run.default" and result:
            self.completed += 1
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pages", type=int, default=1785)
    parser.add_argument("--native-copy", action="store_true")
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=("contiguous", "32_page_runs", "random_pages"),
        default=["contiguous", "32_page_runs", "random_pages"],
    )
    args = parser.parse_args()
    if not 1 <= args.pages <= 4096:
        parser.error("pages must be between 1 and 4096")
    torch.manual_seed(9683)
    device = torch.device("cuda", 0)
    torch.accelerator.set_device_index(device)
    # Last full chunk's prefix in the matched service trace is 1785 pages.
    page, blocks, used = 16, 4096, args.pages
    x = torch.randn(7344, 10240, device=device, dtype=torch.float16) * 0.01
    weight = torch.randn(88, 10240, device=device, dtype=torch.float16) * 0.01
    result = torch.empty(7344, 88, device=device, dtype=torch.float16)
    side = torch.cuda.Stream(priority=-1)
    main_stream = torch.cuda.current_stream()
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cases": [],
    }
    for dtype in (torch.uint8, torch.float16):
        host = torch.empty((blocks, 2, page, 1, 256), dtype=dtype, pin_memory=True)
        host.copy_(torch.randint(0, 100, host.shape, dtype=torch.int32).to(dtype))
        mapped = get_accelerator_view_from_cpu_tensor(host)
        state = HostQSAKV(
            blocks,
            page,
            256,
            device,
            history=mapped,
            hot_tokens=2048,
            rows=8,
            width=2051,
        )
        native_copy = getattr(state, "_stage_host_run", None)
        if args.native_copy:
            assert native_copy is not None, "candidate native helper was not loaded"
        state.host_scales.fill_(0.03125)
        for layout in args.layouts:
            if layout == "contiguous":
                pages = torch.arange(used, dtype=torch.int32)
            elif layout == "32_page_runs":
                runs = torch.randperm(blocks // 32)[: (used + 31) // 32]
                pages = (
                    (runs[:, None] * 32 + torch.arange(32)[None, :])
                    .flatten()[:used]
                    .int()
                )
            else:
                pages = torch.randperm(blocks)[:used].int()
            table = pages.to(device).view(1, -1)
            lengths = torch.tensor([used * page - 3], dtype=torch.int32, device=device)
            plan = state.stage_plan(table, lengths, 131072)
            expected = host[pages.long()]
            expected_scales = (
                state.host_scales.view(-1, page, 2)[pages.long()].flatten(0, 1)
                if state.fp8
                else state.host_scales
            )

            def stage(state=state, table=table, lengths=lengths, plan=plan):
                return state.stage_history(table, lengths, 131072, plan)

            def run(concurrent, stage=stage):
                if concurrent:
                    side.wait_stream(main_stream)
                    with torch.cuda.stream(side):
                        stage()
                    for _ in range(4):
                        torch.mm(x, weight.T, out=result)
                    main_stream.wait_stream(side)
                else:
                    stage()

            for mode in ("gather", "dma", "gather_repeat"):
                state.host = host if mode == "dma" and not args.native_copy else None
                state._stage_host_run = (
                    native_copy if mode == "dma" and args.native_copy else None
                )
                with CopyCalls() as calls:
                    actual = stage()
                torch.accelerator.synchronize()
                assert torch.equal(actual[0].cpu(), expected)
                assert torch.equal(actual[1].cpu(), expected_scales)
                for concurrent in (False, True):
                    run(concurrent)
                    torch.accelerator.synchronize()
                    samples = []
                    for _ in range(3):
                        begin, end = (
                            torch.cuda.Event(enable_timing=True) for _ in range(2)
                        )
                        begin.record()
                        for _ in range(5):
                            run(concurrent)
                        end.record()
                        end.synchronize()
                        samples.append(begin.elapsed_time(end) / 5)
                    row = {
                        "dtype": str(dtype),
                        "pages": used,
                        "layout": layout,
                        "runs": len(plan[0]),
                        "mode": mode,
                        "concurrent_hc": concurrent,
                        "median_ms": statistics.median(samples),
                        "samples_ms": samples,
                        "exact_bytes_and_scales": True,
                        "native_copy": args.native_copy,
                        "native_copy_completions": calls.completed,
                    }
                    report["cases"].append(row)
                    print(json.dumps(row), flush=True)
        del state, host, mapped
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
