# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import torch
import tilelang


def profile(func, inputs, wait: int = 50, warmup: int = 50, rep: int = 100):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=rep),
        # on_trace_ready=torch.profiler.tensorboard_trace_handler('./tb'),
    ) as prof:
        for idx in range(wait + warmup + rep):
            func(*inputs)
            prof.step()
    # print(prof.key_averages().table(sort_by="cpu_time", row_limit=10))
    result = {x.key: x.device_time * 1e-3 for x in prof.key_averages()}
    result["total"] = tilelang.profiler.do_bench(
        lambda: func(*inputs), warmup=warmup, rep=rep
    )
    return result
