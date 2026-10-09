# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare HCX schedules with fixed loaded weights and compiled operators.

Use an installed wheel and a completed reference JSON from
benchmark_flashnext_acceptance.py with HCX enabled, local_schedule false,
separate output projection, TP4 and MTP4. Set
VLLM_ALLOW_INSECURE_SERIALIZATION=1 for the trusted offline worker callbacks.
GPU ownership locks must cover the entire benchmark.

Compare original, recaptured reference, candidate and original graphs again.
Time C1 only when all controls are exact; otherwise compare native schedules
on saved real M5 inputs. Always shut down the owned engine on exit.
"""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import torch
import vllm._C as core

import vllm
from vllm import LLM, SamplingParams
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

# Import the installed vLLM before making repository benchmark helpers visible.
sys.path.append(str(Path(__file__).resolve().parents[1]))
from benchmarks.benchmark_flashnext_acceptance import (
    digest,
    natural_row,
    observed_cohort,
    summarize,
)
from benchmarks.sm70_teacher_conditions import teacher_conditions


def capture_variant(worker, label, enabled):
    from vllm.config import CUDAGraphMode
    from vllm.models.qwen4_exp.nvidia.sm70_hcx import get_hcx_runtime

    runner = worker.model_runner
    manager = runner.cudagraph_manager
    runtime = get_hcx_runtime(runner.device)
    torch.accelerator.synchronize()
    if not hasattr(worker, "_hcx_schedule_graphs"):
        assert runtime.local_schedule is False
        worker._hcx_schedule_graphs = {"original": dict(manager.graphs)}
        worker._hcx_schedule_flags = {"original": False}
    original = worker._hcx_schedule_graphs["original"]
    descriptions = manager._capture_descs
    selected = [d for d in descriptions[CUDAGraphMode.FULL] if d.num_tokens == 5]
    assert selected and all(d in original for d in selected)
    calls = []
    original_run = runtime.run

    def observed(*args, **kwargs):
        calls.append((runtime.local_schedule, torch.cuda.is_current_stream_capturing()))
        return original_run(*args, **kwargs)

    runtime.local_schedule = enabled
    runtime.run = observed
    manager.graphs = {}
    manager._capture_descs = {CUDAGraphMode.FULL: selected}
    connector = runner._ple_offload_connector
    if connector is not None:
        connector.signal_dummy_outputs(runner.max_num_tokens)
    try:
        manager.capture(
            runner.model,
            runner.model_state,
            runner.input_buffers,
            runner.intermediate_tensors,
            runner.block_tables,
            runner.attn_groups,
            runner.kv_cache_config,
            has_lora=False,
            use_aux_hidden_state_outputs=runner.use_aux_hidden_state_outputs,
        )
        torch.accelerator.synchronize()
        assert set(manager.graphs) == set(selected)
        assert all(flag is enabled for flag, _ in calls)
        captured_calls = sum(capturing for _, capturing in calls)
        # The prepared-module count includes a final mix-only boundary.
        # Record actual invocations and require matched counts across arms.
        assert captured_calls > 0 and captured_calls % len(selected) == 0
        graphs = dict(original)
        graphs.update(manager.graphs)
        worker._hcx_schedule_graphs[label] = graphs
        worker._hcx_schedule_flags[label] = enabled
        manager.graphs = graphs
        return dict(
            rank=worker.rank,
            label=label,
            local_schedule=enabled,
            captured_hcx_calls=captured_calls,
            descriptions=[str(d) for d in selected],
            original_graph_ids=[id(original[d]) for d in selected],
            selected_graph_ids=[id(graphs[d]) for d in selected],
        )
    finally:
        manager._capture_descs = descriptions
        runtime.run = original_run
        if connector is not None:
            connector.release_outputs()


def select_variant(worker, label):
    from vllm.models.qwen4_exp.nvidia.sm70_hcx import get_hcx_runtime

    runner = worker.model_runner
    runtime = get_hcx_runtime(runner.device)
    torch.accelerator.synchronize()
    if not hasattr(worker, "_hcx_schedule_graphs"):
        assert label == "original" and runtime.local_schedule is False
        worker._hcx_schedule_graphs = {
            "original": dict(runner.cudagraph_manager.graphs)
        }
        worker._hcx_schedule_flags = {"original": False}
    enabled = worker._hcx_schedule_flags[label]
    runner.cudagraph_manager.graphs = worker._hcx_schedule_graphs[label]
    runtime.local_schedule = enabled
    return dict(rank=worker.rank, label=label, local_schedule=enabled)


def enable_snapshots(worker):
    from vllm.models.qwen4_exp.nvidia.sm70_hcx import get_hcx_runtime

    runtime = get_hcx_runtime(worker.model_runner.device)
    runtime.diagnostic = True
    return dict(rank=worker.rank, diagnostic=True)


def compare_snapshots(worker, directory):
    from vllm.models.qwen4_exp.nvidia.hyperconnection import _PARTIAL_MODULES
    from vllm.models.qwen4_exp.nvidia.sm70_hcx import get_hcx_runtime

    runtime = get_hcx_runtime(worker.model_runner.device)
    saved_schedule, saved_diagnostic = runtime.local_schedule, runtime.diagnostic
    runtime.diagnostic = False
    rows = []
    first_failure = None
    root = Path(directory) / f"rank-{worker.rank}"
    root.mkdir(parents=True, exist_ok=True)
    try:
        with torch.inference_mode():
            for name, snapshot in sorted(runtime.snapshots.items()):
                module = _PARTIAL_MODULES[name]
                assert snapshot["partial"].shape[0] == 5
                values = []
                for enabled in (False, True):
                    runtime.local_schedule = enabled
                    outputs = runtime.run(
                        snapshot["partial"],
                        snapshot["hidden"],
                        snapshot["injection"],
                        module.hc_norm.weight,
                        module.config.rms_norm_eps,
                        module._hcx_down,
                        module._hcx_up,
                        secondary=snapshot.get("secondary"),
                    )
                    values.append(tuple(x.clone() for x in outputs))
                counts, errors, finite = [], [], []
                for left, right in zip(*values, strict=True):
                    counts.append(
                        int(
                            torch.count_nonzero(
                                left.view(torch.int16) != right.view(torch.int16)
                            ).item()
                        )
                    )
                    finite.append(
                        bool(
                            torch.isfinite(left).all().item()
                            and torch.isfinite(right).all().item()
                        )
                    )
                    errors.append(
                        float((left.float() - right.float()).abs().max().item())
                    )
                rows.append(
                    dict(
                        name=name,
                        bit_mismatches=counts,
                        max_errors=errors,
                        finite=finite,
                    )
                )
                if first_failure is None and (any(counts) or not all(finite)):
                    first_failure = str(root / "first-failure.pt")
                    torch.save(
                        dict(
                            name=name,
                            inputs={k: v.detach().cpu() for k, v in snapshot.items()},
                            norm=module.hc_norm.weight.detach().cpu(),
                            eps=module.config.rms_norm_eps,
                            down=module._hcx_down.cpu(),
                            up=module._hcx_up.cpu(),
                            reference=[x.cpu() for x in values[0]],
                            candidate=[x.cpu() for x in values[1]],
                        ),
                        first_failure,
                    )
        assert rows
        torch.accelerator.synchronize()
        return dict(
            rank=worker.rank,
            modules=len(rows),
            rows=rows,
            first_failure=first_failure,
            exact_modules=sum(
                not any(x["bit_mismatches"]) and all(x["finite"]) for x in rows
            ),
        )
    finally:
        runtime.local_schedule, runtime.diagnostic = saved_schedule, saved_diagnostic


def local_quality_checks(root, phases):
    result = {}
    for label, left, right in (
        ("recapture_control", "original", "recaptured_reference"),
        ("schedule", "recaptured_reference", "candidate"),
        ("roundtrip_control", "original", "original_again"),
    ):
        a, b = phases[left], phases[right]
        assert len(a["teacher"]) == len(b["teacher"]) == 64
        exact = 0
        for x, y in zip(a["teacher"], b["teacher"], strict=True):
            assert x == y
            va = torch.load(root / left / (x["key"] + ".pt"), weights_only=True)[
                "logits"
            ]
            vb = torch.load(root / right / (y["key"] + ".pt"), weights_only=True)[
                "logits"
            ]
            exact += bool(torch.equal(va, vb))
        result[label] = dict(
            teacher_exact=exact,
            natural_exact=sum(
                x["output_token_ids"] == y["output_token_ids"]
                for x, y in zip(a["natural"], b["natural"], strict=True)
            ),
            acceptance_exact=sum(
                x["acceptance"] == y["acceptance"]
                for x, y in zip(a["natural"], b["natural"], strict=True)
            ),
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Pass only these trusted benchmark functions over offline RPC.
    # Catch a missing serialization opt-in before the expensive model load.
    for function in (
        capture_variant,
        select_variant,
        enable_snapshots,
        compare_snapshots,
    ):
        assert callable(MsgpackDecoder().decode(MsgpackEncoder().encode(function)))
    reference = json.loads(args.reference.read_text())
    assert reference["complete"]
    assert (
        hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest()
        == reference["core_sha256"]
    )
    assert "/site-packages/vllm/" in str(Path(vllm.__file__))
    # Match the acceptance benchmark's matmul precision contract.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    config = copy.deepcopy(reference["config"])
    assert config["kernel_config"]["sm70_hcx_local_schedule"] is False
    assert config["tensor_parallel_size"] == 4
    assert config["speculative_config"]["num_speculative_tokens"] == 4
    assert config["kernel_config"]["sm70_hcx"] is True
    assert config["kernel_config"]["sm70_hcx_output_projection"] is False
    report = dict(
        complete=False,
        scope="same-process quality localization; C1 timing only if controls pass",
        reference=str(args.reference),
        core_sha256=reference["core_sha256"],
        version=vllm.__version__,
        origin=vllm.__file__,
        config=config,
        captures={},
        phases={},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n"
        )

    save()
    # EngineArgs mutates nested speculative configuration during initialization.
    # Keep the recorded comparison contract independent from that mutable input.
    llm = LLM(**copy.deepcopy(config))
    try:
        assert config == reference["config"]
        report["worker_routes"] = llm.collective_rpc(
            "get_sm70_acceleration_report", timeout=30
        )
        assert all(
            w["decode_cudagraph_mode"] == "FULL" for w in report["worker_routes"]
        )
        llm.generate(
            {"prompt_token_ids": reference["rows"][0]["prompt_token_ids"]},
            SamplingParams(temperature=0, max_tokens=16),
            use_tqdm=False,
        )
        for phase, graph_name, enabled, recapture in (
            ("original", "original", False, False),
            ("recaptured_reference", "recaptured_reference", False, True),
            ("candidate", "candidate", True, True),
            ("original_again", "original", False, False),
        ):
            if recapture:
                report["captures"][phase] = llm.collective_rpc(
                    capture_variant, args=(graph_name, enabled), timeout=120
                )
                if phase == "candidate":
                    for left, right in zip(
                        report["captures"]["recaptured_reference"],
                        report["captures"]["candidate"],
                        strict=True,
                    ):
                        assert left["captured_hcx_calls"] == right["captured_hcx_calls"]
                        assert left["descriptions"] == right["descriptions"]
            selection = llm.collective_rpc(
                select_variant, args=(graph_name,), timeout=30
            )
            assert all(w["local_schedule"] is enabled for w in selection)
            root = args.output.parent / phase
            root.mkdir(exist_ok=True)
            row = dict(selection=selection, natural=[], teacher=[])
            report["phases"][phase] = row
            save()
            params = SamplingParams(**reference["sampling"], seed=20261005)
            for prompt in reference["rows"]:
                row["natural"].append(
                    natural_row(llm, prompt, prompt["prompt_token_ids"], params)
                )
                save()
            for key, prefix, forced in teacher_conditions(reference, 8):
                llm.collective_rpc(
                    "start_teacher_capture", args=(str(root), key), timeout=30
                )
                try:
                    llm.generate(
                        {"prompt_token_ids": prefix},
                        SamplingParams(
                            temperature=0, max_tokens=6, allowed_token_ids=[forced]
                        ),
                        use_tqdm=False,
                    )
                finally:
                    workers = llm.collective_rpc("stop_teacher_capture", timeout=30)
                assert all(w["captured"] == 1 for w in workers)
                payload = torch.load(root / f"{key}.pt", weights_only=True)
                assert payload["position"].item() == len(prefix)
                assert payload["input_ids"].item() == forced
                row["teacher"].append(
                    dict(
                        key=key,
                        prefix_sha256=digest(prefix),
                        position=len(prefix),
                        forced=forced,
                    )
                )
                save()
            print(
                json.dumps(
                    dict(
                        phase=phase,
                        natural=len(row["natural"]),
                        teacher=len(row["teacher"]),
                    )
                ),
                flush=True,
            )
        report["quality_phases_complete"] = True
        report["local_checks"] = local_quality_checks(
            args.output.parent, report["phases"]
        )
        save()
        if any(
            row != dict(teacher_exact=64, natural_exact=8, acceptance_exact=8)
            for row in report["local_checks"].values()
        ):
            report["operator_check_scope"] = (
                "Both native schedules on each module's same saved M5 inputs "
                "from a 32-token real-model continuation; no speed claim"
            )
            llm.collective_rpc(enable_snapshots, timeout=30)
            report["captures"]["operator_reference"] = llm.collective_rpc(
                capture_variant, args=("operator_reference", False), timeout=120
            )
            llm.collective_rpc(select_variant, args=("operator_reference",), timeout=30)
            llm.generate(
                {"prompt_token_ids": reference["rows"][1]["prompt_token_ids"]},
                SamplingParams(temperature=0, max_tokens=32, seed=20261005),
                use_tqdm=False,
            )
            report["operator_checks"] = llm.collective_rpc(
                compare_snapshots,
                args=(str(args.output.parent / "operators"),),
                timeout=180,
            )
        else:
            fixed_ids = reference["rows"][1]["prompt_token_ids"]
            fixed_ids = (fixed_ids * (8192 // len(fixed_ids) + 1))[:8192]
            probe_params = SamplingParams(
                temperature=0, top_p=1, top_k=-1, max_tokens=256, ignore_eos=True
            )
            report["probes"] = []
            for label in ("recaptured_reference", "candidate"):
                llm.collective_rpc(select_variant, args=(label,), timeout=30)
                llm.generate(
                    {"prompt_token_ids": fixed_ids}, probe_params, use_tqdm=False
                )
            for label in (
                "recaptured_reference",
                "candidate",
                "candidate",
                "recaptured_reference",
                "candidate",
                "recaptured_reference",
                "recaptured_reference",
                "candidate",
            ):
                llm.collective_rpc(select_variant, args=(label,), timeout=30)
                steps, outputs = observed_cohort(llm, fixed_ids, probe_params)
                report["probes"].append(
                    dict(
                        arm=label,
                        steps=steps,
                        summary=summarize(steps, 1),
                        output_token_ids=[
                            list(o.outputs[0].token_ids) for o in outputs
                        ],
                    )
                )
                save()
        report["complete"] = True
        save()
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
