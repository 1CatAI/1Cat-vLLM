# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure each GPU's host link: pinned H2D DMA and in-kernel UVA reads.

Reports GB/s for large transfers and the time to move one expert-sized
(~0.47 MiB) block, both alone and with all GPUs copying at once (the
decode case, where every TP rank fetches its own slice of the same expert).
"""

import argparse
import threading
import time

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor


@triton.jit
def _uva_read(src, dst, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(dst + offsets, tl.load(src + offsets, mask=mask, other=0), mask=mask)


def measure(device, nbytes, repeats, barrier=None):
    torch.accelerator.set_device_index(device)
    host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    host.fill_(1)
    dev = torch.empty(nbytes, dtype=torch.uint8, device=device)
    view = get_accelerator_view_from_cpu_tensor(host)
    stream = torch.cuda.Stream(device)
    out = {}
    for name in ("dma", "uva"):
        with torch.cuda.stream(stream):
            for _ in range(3):
                if name == "dma":
                    dev.copy_(host, non_blocking=True)
                else:
                    _uva_read[(triton.cdiv(nbytes // 4, 1024),)](
                        view.view(torch.int32),
                        dev.view(torch.int32),
                        nbytes // 4,
                        BLOCK=1024,
                    )
            stream.synchronize()
            if barrier is not None:
                barrier.wait()
            start = time.perf_counter()
            for _ in range(repeats):
                if name == "dma":
                    dev.copy_(host, non_blocking=True)
                else:
                    _uva_read[(triton.cdiv(nbytes // 4, 1024),)](
                        view.view(torch.int32),
                        dev.view(torch.int32),
                        nbytes // 4,
                        BLOCK=1024,
                    )
            stream.synchronize()
            elapsed = (time.perf_counter() - start) / repeats
        out[name] = (nbytes / elapsed / 1e9, elapsed * 1e6)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="490240,67108864")
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    gpus = torch.accelerator.device_count()
    for nbytes in (int(x) for x in args.sizes.split(",")):
        for gpu in range(gpus):
            result = measure(gpu, nbytes, args.repeats)
            print(
                f"alone  gpu{gpu} {nbytes:>10} B  "
                + "  ".join(
                    f"{k}={gbps:6.2f} GB/s {us:8.1f} us"
                    for k, (gbps, us) in result.items()
                ),
                flush=True,
            )
        barrier = threading.Barrier(gpus)
        results = [None] * gpus

        def run(index, results=results, nbytes=nbytes, barrier=barrier):
            results[index] = measure(index, nbytes, args.repeats, barrier)

        threads = [threading.Thread(target=run, args=(i,)) for i in range(gpus)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        for gpu, result in enumerate(results):
            print(
                f"concur gpu{gpu} {nbytes:>10} B  "
                + "  ".join(
                    f"{k}={gbps:6.2f} GB/s {us:8.1f} us"
                    for k, (gbps, us) in result.items()
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
