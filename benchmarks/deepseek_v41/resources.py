# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sample node/driver usage separately from per-worker allocator peaks."""

import threading
import time
from collections import deque

from benchmarks.deepseek_v41.shared_host_probe import cgroup_memory, memory_available


class ResourceMonitor:
    def __init__(self):
        from vllm.third_party import pynvml

        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(4)]
        self.gpus = [
            dict(
                uuid=pynvml.nvmlDeviceGetUUID(handle),
                total_bytes=pynvml.nvmlDeviceGetMemoryInfo(handle).total,
                peak_used_bytes=0,
                minimum_free_bytes=pynvml.nvmlDeviceGetMemoryInfo(handle).total,
            )
            for handle in self.handles
        ]
        self.minimum_available = memory_available()
        self.peak_cgroup = 0
        self.samples: deque[dict] = deque(maxlen=20000)
        self.samples_seen = 0
        self.stopped, self.lock = threading.Event(), threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stopped.is_set():
            info = [self.nvml.nvmlDeviceGetMemoryInfo(h) for h in self.handles]
            usage = []
            for handle in self.handles:
                try:
                    utilization = self.nvml.nvmlDeviceGetUtilizationRates(handle)
                    usage.append(
                        dict(
                            gpu_utilization_pct=utilization.gpu,
                            memory_controller_utilization_pct=utilization.memory,
                            power_watts=self.nvml.nvmlDeviceGetPowerUsage(handle)
                            / 1000,
                            sm_clock_mhz=self.nvml.nvmlDeviceGetClockInfo(
                                handle, self.nvml.NVML_CLOCK_SM
                            ),
                            memory_clock_mhz=self.nvml.nvmlDeviceGetClockInfo(
                                handle, self.nvml.NVML_CLOCK_MEM
                            ),
                            throttle_reasons=self.nvml.nvmlDeviceGetCurrentClocksThrottleReasons(
                                handle
                            ),
                        )
                    )
                except self.nvml.NVMLError as error:
                    usage.append(dict(unavailable=str(error)))
            available, current = memory_available(), cgroup_memory()["current"]
            with self.lock:
                self.samples.append(dict(monotonic=time.monotonic(), gpu=usage))
                self.samples_seen += 1
                for entry, value in zip(self.gpus, info):
                    entry["peak_used_bytes"] = max(entry["peak_used_bytes"], value.used)
                    entry["minimum_free_bytes"] = min(
                        entry["minimum_free_bytes"], value.free
                    )
                self.minimum_available = min(self.minimum_available, available)
                self.peak_cgroup = max(self.peak_cgroup, current)
            self.stopped.wait(0.25)

    def report(self):
        with self.lock:
            return dict(
                sample_interval_seconds=0.25,
                gpu=[dict(entry) for entry in self.gpus],
                host_minimum_available_bytes=self.minimum_available,
                cgroup_peak_observed_bytes=self.peak_cgroup,
                gpu_usage_samples=list(self.samples),
                gpu_usage_samples_dropped=self.samples_seen - len(self.samples),
            )

    def close(self):
        self.stopped.set()
        self.thread.join(2)
        self.nvml.nvmlShutdown()
