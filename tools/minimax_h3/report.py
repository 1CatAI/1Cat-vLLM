# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Export measured H3 rank throughput, phase times and NVML curves."""

import argparse
import csv
import json
from pathlib import Path


def export_report(run_dir):
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    if run.get("timing_valid") is False:
        raise ValueError(
            "Cannot report excluded timing: "
            + run.get("timing_exclusion_reason", "run marked invalid")
        )
    ranks = sorted(run["ranks"], key=lambda rank: rank["rank"])
    denoise_seconds = max(rank["stage_seconds"]["denoise"] for rank in ranks)
    rows = []
    for rank in ranks:
        rows.append(
            {
                "rank": rank["rank"],
                "gpu": run["gpus"][rank["rank"]],
                "useful_flops": rank["useful_denoise_flops"],
                "denoise_seconds": denoise_seconds,
                "effective_tflops": rank["useful_denoise_flops"]
                / denoise_seconds
                / 1e12,
                "peak_allocated_gib": rank["peak_allocated_bytes"] / 1024**3,
                "dit_calls": rank["dit_calls"],
            }
        )
    with (run_dir / "rank-performance.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (run_dir / "phase-times.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["rank", "phase", "seconds"])
        for rank in ranks:
            for phase, seconds in rank["stage_seconds"].items():
                writer.writerow([rank["rank"], phase, seconds])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    samples = [
        json.loads(line)
        for line in (run_dir / "nvml.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if not samples:
        raise ValueError("no NVML samples were recorded")
    start = min(sample["timestamp"] for sample in samples)
    fields = [
        ("gpu_util_percent", "GPU utilization (%)", 1),
        ("memory_used_bytes", "Device memory (GiB)", 1024**3),
        ("power_watts", "Power (W)", 1),
        ("sm_clock_mhz", "SM clock (MHz)", 1),
        ("temperature_c", "Temperature (C)", 1),
        ("memory_util_percent", "Memory controller utilization (%)", 1),
    ]
    figure, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    for axis, (key, label, divisor) in zip(axes.flat, fields):
        for gpu in run["gpus"]:
            points = [
                sample
                for sample in samples
                if sample["gpu"] == gpu and sample.get(key) is not None
            ]
            axis.plot(
                [point["timestamp"] - start for point in points],
                [point[key] / divisor for point in points],
                label=f"GPU {gpu}",
                linewidth=1,
            )
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(ncol=4, fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("Seconds from first NVML sample")
    figure.suptitle(
        f"H3 {run['config']['attention_backend']} "
        f"({run.get('mode', 'native request')}) — "
        "NVML diagnostics (Tensor Core activity requires Nsight Compute)"
    )
    figure.tight_layout()
    for extension in ("png", "svg"):
        figure.savefig(run_dir / f"nvml-curves.{extension}", dpi=160)
    plt.close(figure)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(export_report(args.run_dir), indent=2))


if __name__ == "__main__":
    main()
