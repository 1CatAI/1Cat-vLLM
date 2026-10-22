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
import hashlib
import importlib
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


class MultiRunScreen(HostQSAKV):
    """Benchmark-only admission screen using the normal native operation."""

    max_runs = 64

    def _stage_range(self, plan, history, scales, first, last):
        runs = [r for r in plan[0] if r[1] < last and r[1] + r[2] > first]
        if self._stage_host_run is not None and len(runs) <= self.max_runs:
            for source, target, count in runs:
                lo, hi = max(first, target), min(last, target + count)
                if lo < hi and not self._stage_host_run(
                    self.history,
                    self.scales,
                    history,
                    scales,
                    source + lo - target,
                    lo,
                    hi - lo,
                ):
                    break
            else:
                return
        super()._stage_range(plan, history, scales, first, last)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pages", type=int, default=1785)
    parser.add_argument("--prefix-pages", type=int)
    parser.add_argument("--native-copy", action="store_true")
    parser.add_argument("--native-run-screen", type=int)
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=(
            "contiguous",
            "2_runs",
            "4_runs",
            "8_runs",
            "serving_three_runs",
            "32_page_runs",
            "random_pages",
        ),
        default=["contiguous", "32_page_runs", "random_pages"],
    )
    args = parser.parse_args()
    if not 1 <= args.pages <= 4096:
        parser.error("pages must be between 1 and 4096")
    if args.prefix_pages is not None and not 1 <= args.prefix_pages <= args.pages:
        parser.error("prefix-pages must be positive and no larger than pages")
    if args.native_run_screen is not None and (
        not args.native_copy or args.native_run_screen < 1
    ):
        parser.error("native-run-screen requires native-copy and a positive limit")
    torch.manual_seed(9683)
    device = torch.device("cuda", 0)
    torch.accelerator.set_device_index(device)
    # Last full chunk's prefix in the matched service trace is 1785 pages.
    page, blocks, used = 16, 4096, args.pages
    copied = args.prefix_pages or used
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
    if args.native_copy:
        native_module = importlib.import_module("vllm._sm70_qsa_device_C")
        assert native_module.__file__ is not None
        native_path = Path(native_module.__file__).resolve()
        report["native_artifact"] = {
            "path": str(native_path),
            "sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
        }
    for dtype in (torch.uint8, torch.float16):
        host = torch.empty((blocks, 2, page, 1, 256), dtype=dtype, pin_memory=True)
        host.copy_(torch.randint(0, 100, host.shape, dtype=torch.int32).to(dtype))
        mapped = get_accelerator_view_from_cpu_tensor(host)
        state_class = MultiRunScreen if args.native_run_screen else HostQSAKV
        state = state_class(
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
        if isinstance(state, MultiRunScreen):
            state.max_runs = args.native_run_screen
        if args.native_copy:
            assert native_copy is not None, "candidate native helper was not loaded"
        state.host_scales.fill_(0.03125)
        for layout in args.layouts:
            if layout == "contiguous":
                pages = torch.arange(used, dtype=torch.int32)
            elif layout == "serving_three_runs":
                # Measured final-chunk prefix, translated by its first
                # physical page: [1989, 51], [3672, 459], [4182, 1275].
                if used != 1785:
                    parser.error("serving_three_runs requires exactly 1785 pages")
                pages = torch.cat(
                    [
                        torch.arange(51),
                        torch.arange(1683, 2142),
                        torch.arange(2193, 3468),
                    ]
                ).int()
            elif layout in ("2_runs", "4_runs", "8_runs"):
                count = int(layout.split("_", 1)[0])
                width = (used + count - 1) // count
                if used + 16 * count > blocks:
                    parser.error("fragmented run layout exceeds the source capacity")
                pages = (
                    (
                        torch.arange(count)[:, None] * (width + 16)
                        + torch.arange(width)[None, :]
                    )
                    .flatten()[:used]
                    .int()
                )
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
            expected = host[pages[:copied].long()]
            expected_scales = (
                state.host_scales.view(-1, page, 2)[pages[:copied].long()].flatten(0, 1)
                if state.fp8
                else state.host_scales
            )

            def stage(state=state, table=table, lengths=lengths, plan=plan):
                return state.stage_history(table, lengths, 131072, plan, upto=copied)

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
                assert torch.equal(actual[0][:copied].cpu(), expected)
                staged_scales = actual[1][: copied * page] if state.fp8 else actual[1]
                assert torch.equal(staged_scales.cpu(), expected_scales)
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
                        "copied_pages": copied,
                        "layout": layout,
                        "runs": len(plan[0]),
                        "copied_runs": sum(r[1] < copied for r in plan[0]),
                        "mode": mode,
                        "concurrent_hc": concurrent,
                        "median_ms": statistics.median(samples),
                        "samples_ms": samples,
                        "exact_bytes_and_scales": True,
                        "native_copy": args.native_copy,
                        "native_run_screen": args.native_run_screen,
                        "native_copy_completions": calls.completed,
                    }
                    report["cases"].append(row)
                    print(json.dumps(row), flush=True)
        del state, host, mapped
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
