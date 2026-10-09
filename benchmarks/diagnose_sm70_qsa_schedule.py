# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Localize QSA changes with one set of loaded weights and graph ablations.

Graph ablations use the production dispatch and normally installed operators.
No research extension or replacement attention function is loaded.
The draft graphs and prefill implementation remain the original ones.
Capture handling follows benchmarks/diagnose_sm70_hcx_schedule.py.
"""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import vllm._C as core

import vllm
from vllm import LLM, SamplingParams
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

# Installed vLLM is imported first; these are offline benchmark helpers only.
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))
from benchmark_flashnext_acceptance import (
    digest,
    natural_row,
    observed_cohort,
    summarize,
)

from benchmarks.sm70_teacher_conditions import teacher_conditions


def configure(worker, mode):
    import torch

    from vllm.models.qwen4_exp.nvidia.ops.device_kv_attention import (
        initialize_device_history_attention,
    )
    from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAAttention

    assert mode in ("control", "scorer", "native", "both")
    runner = worker.model_runner
    torch.accelerator.synchronize()
    if not hasattr(worker, "_qsa_structure"):
        owners = [
            x for x in runner.model.modules() if isinstance(x, Qwen4ExpQSAAttention)
        ]
        assert len(owners) == 12
        assert all(x.host_kv.device_reference for x in owners)
        assert all(x.host_kv.device_history_workspace is None for x in owners)
        assert all(x.indexer.shared_key_scoring is False for x in owners)
        state = dict(
            owners=owners,
            graphs={"original": dict(runner.cudagraph_manager.graphs)},
            modes={"original": "control"},
            mode="control",
            workspaces=[],
        )
        for owner in owners:
            initialize_device_history_attention(owner.host_kv, True)
            assert owner.host_kv.device_history_workspace is not None, (
                owner.host_kv.device_history_reason
            )
            state["workspaces"].append(owner.host_kv.device_history_workspace)
        worker._qsa_structure = state
    state = worker._qsa_structure
    state["mode"] = mode
    for owner, workspace in zip(state["owners"], state["workspaces"], strict=True):
        owner.indexer.shared_key_scoring = mode in ("scorer", "both")
        owner.host_kv.device_history_workspace = (
            workspace if mode in ("native", "both") else None
        )
    return dict(
        rank=worker.rank,
        mode=mode,
        owners=len(state["owners"]),
        native_owners=sum(
            x.host_kv.device_history_workspace is not None for x in state["owners"]
        ),
        scorer_owners=sum(x.indexer.shared_key_scoring for x in state["owners"]),
        dispatch="production helpers; no operator replacement",
    )


def capture(worker, label, mode):
    from vllm.config import CUDAGraphMode

    configure(worker, mode)
    state = worker._qsa_structure
    runner = worker.model_runner
    manager = runner.cudagraph_manager
    descriptions = manager._capture_descs
    selected = [d for d in descriptions[CUDAGraphMode.FULL] if 0 < d.num_tokens <= 20]
    assert {5, 20}.issubset({d.num_tokens for d in selected})
    assert all(d in state["graphs"]["original"] for d in selected)
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
        graphs = dict(state["graphs"]["original"])
        graphs.update(manager.graphs)
        state["graphs"][label] = graphs
        state["modes"][label] = mode
        manager.graphs = graphs
        return dict(
            rank=worker.rank,
            label=label,
            mode=mode,
            descriptions=[str(d) for d in selected],
            rows=sorted({d.num_tokens for d in selected}),
        )
    finally:
        manager._capture_descs = descriptions
        if connector is not None:
            connector.release_outputs()


def select(worker, label):
    configure(worker, worker._qsa_structure["modes"][label])
    worker.model_runner.cudagraph_manager.graphs = worker._qsa_structure["graphs"][
        label
    ]
    return dict(rank=worker.rank, label=label, mode=worker._qsa_structure["mode"])


def start_interleave(worker, candidate, width, reverse=False):
    """Alternate captured graphs inside one request, without redoing prefill."""
    manager = worker.model_runner.cudagraph_manager
    if not hasattr(worker, "_qsa_interleave"):
        original = manager.run_fullgraph

        def replay(desc):
            state = worker._qsa_interleave
            if (
                state["enabled"]
                and desc.num_tokens == state["width"] * 5
                and desc.num_reqs == state["width"]
                and desc.uniform_token_count == 5
            ):
                index = len(state["rows"])
                label = "recaptured_control"
                if index >= 8:
                    label = state["plan"][((index - 8) // 8) % 4]
                manager.graphs = worker._qsa_structure["graphs"][label]
                state["rows"].append(
                    dict(
                        step=worker._graph_parity_recorder.step,
                        index=index,
                        label=label,
                        warmup=index < 8,
                    )
                )
            return original(desc)

        manager.run_fullgraph = replay
    plan = ["recaptured_control", candidate, candidate, "recaptured_control"]
    if reverse:
        plan = [candidate, "recaptured_control", "recaptured_control", candidate]
    worker._qsa_interleave = dict(enabled=True, width=width, plan=plan, rows=[])
    return dict(rank=worker.rank, width=width, plan=plan)


def stop_interleave(worker):
    state = worker._qsa_interleave
    state["enabled"] = False
    return dict(rank=worker.rank, rows=state["rows"], plan=state["plan"])


def interleaved_probes(llm, reference, report, save):
    fixed = reference["rows"][1]["prompt_token_ids"]
    report["interleaved"] = []
    for width, input_len in ((1, 8192), (4, 128)):
        ids = (fixed * (input_len // len(fixed) + 1))[:input_len]
        params = SamplingParams(
            temperature=0, top_p=1, top_k=-1, max_tokens=600, ignore_eos=True
        )
        llm.collective_rpc(select, args=("recaptured_control",), timeout=30)
        steps, outputs = observed_cohort(llm, ids, params, width)
        expected = [list(o.outputs[0].token_ids) for o in outputs]
        report["interleaved"].append(
            dict(
                width=width,
                arm="pure_control",
                summary=summarize(steps, width),
                steps=steps,
                output_token_ids=expected,
            )
        )
        save()
        for candidate in ("scorer", "native", "both"):
            for reverse in (False, True):
                llm.collective_rpc(select, args=("recaptured_control",), timeout=30)
                llm.collective_rpc(
                    "start_graph_parity_observer", args=(False, True), timeout=30
                )
                llm.collective_rpc(
                    start_interleave, args=(candidate, width, reverse), timeout=30
                )
                try:
                    steps, outputs = observed_cohort(llm, ids, params, width)
                finally:
                    labels = llm.collective_rpc(stop_interleave, timeout=30)
                    workers = llm.collective_rpc(
                        "read_graph_parity_observer", args=(True,), timeout=30
                    )
                actual = [list(o.outputs[0].token_ids) for o in outputs]
                sequences = [[r["label"] for r in w["rows"]] for w in labels]
                assert all(s == sequences[0] for s in sequences)
                assert len(sequences[0]) >= 72
                entry = dict(
                    width=width,
                    candidate=candidate,
                    reverse=reverse,
                    summary=summarize(steps, width),
                    steps=steps,
                    output_token_ids=actual,
                    output_exact=actual == expected,
                    labels=labels,
                    workers=workers,
                )
                report["interleaved"].append(entry)
                save()
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in entry.items()
                            if k
                            not in ("steps", "output_token_ids", "labels", "workers")
                        }
                    ),
                    flush=True,
                )


def quality(root, phases, left, right):
    a, b = phases[left], phases[right]
    records = []
    assert a["teacher"] == b["teacher"]
    for row in a["teacher"]:
        x = torch.load(root / left / (row["key"] + ".pt"), weights_only=True)[
            "logits"
        ].float()
        y = torch.load(root / right / (row["key"] + ".pt"), weights_only=True)[
            "logits"
        ].float()
        p, q = x.log_softmax(-1), y.log_softmax(-1)
        records.append(
            dict(
                key=row["key"],
                exact=torch.equal(x, y),
                top1_same=torch.equal(x.argmax(-1), y.argmax(-1)),
                kl=float((p.exp() * (p - q)).sum(-1).mean()),
                max_absolute=float((x - y).abs().max()),
                relative_l2=float((x - y).norm() / x.norm().clamp_min(1e-20)),
            )
        )
    delta = np.array(
        [
            y["acceptance"]["draft_acceptance_rate"]
            - x["acceptance"]["draft_acceptance_rate"]
            for x, y in zip(a["natural"], b["natural"], strict=True)
        ]
    )
    indices = np.random.default_rng(20261005).integers(0, 8, (20000, 8))
    return dict(
        teacher=records,
        teacher_exact=sum(r["exact"] for r in records),
        teacher_top1_same=sum(r["top1_same"] for r in records),
        kl_mean=float(np.mean([r["kl"] for r in records])),
        kl_max=max(r["kl"] for r in records),
        natural_exact=sum(
            x["output_token_ids"] == y["output_token_ids"]
            for x, y in zip(a["natural"], b["natural"], strict=True)
        ),
        acceptance_exact=sum(
            x["acceptance"] == y["acceptance"]
            for x, y in zip(a["natural"], b["natural"], strict=True)
        ),
        acceptance_difference=float(delta.mean()),
        acceptance_difference_95ci=np.quantile(
            delta[indices].mean(1), [0.025, 0.975]
        ).tolist(),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--interleaved-only",
        action="store_true",
        help="Diagnose timing drift with symmetric eight-round blocks and GPU events.",
    )
    args = p.parse_args()
    for fn in (configure, capture, select, start_interleave, stop_interleave):
        assert callable(MsgpackDecoder().decode(MsgpackEncoder().encode(fn)))
    reference = json.loads(args.reference.read_text())
    assert reference["complete"] and "/site-packages/vllm/" in vllm.__file__
    core_sha256 = hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest()
    config = copy.deepcopy(reference["config"])
    assert config["kernel_config"]["sm70_qsa_device_history"] is False
    assert config["kernel_config"]["sm70_qsa_shared_key"] is False
    assert config["kernel_config"]["qsa_host_kv_device_reference"] is True
    assert config["tensor_parallel_size"] == 4
    assert config["speculative_config"]["num_speculative_tokens"] == 4
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        complete=False,
        scope=(
            "Same-process target QSA ablation; installed native operators; "
            "production capability dispatch"
        ),
        config=config,
        origin=vllm.__file__,
        version=vllm.__version__,
        core_sha256=core_sha256,
        captures={},
        phases={},
        checks={},
        probes=[],
    )

    def save():
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n"
        )

    save()
    llm = LLM(**copy.deepcopy(config))
    try:
        report["routes"] = llm.collective_rpc(
            "get_sm70_acceleration_report", timeout=30
        )
        llm.collective_rpc(configure, args=("control",), timeout=30)
        llm.generate(
            {"prompt_token_ids": reference["rows"][0]["prompt_token_ids"]},
            SamplingParams(temperature=0, max_tokens=16),
            use_tqdm=False,
        )
        if args.interleaved_only:
            for label, mode in (
                ("recaptured_control", "control"),
                ("scorer", "scorer"),
                ("native", "native"),
                ("both", "both"),
            ):
                report["captures"][label] = llm.collective_rpc(
                    capture, args=(label, mode), timeout=180
                )
                save()
            interleaved_probes(llm, reference, report, save)
            report["complete"] = True
            save()
            return
        for label, mode in [
            ("original", "control"),
            ("recaptured_control", "control"),
            ("both", "both"),
            ("original_again", "control"),
        ]:
            graph_label = "original" if label == "original_again" else label
            if label not in ("original", "original_again"):
                report["captures"][label] = llm.collective_rpc(
                    capture, args=(label, mode), timeout=180
                )
            selection = llm.collective_rpc(select, args=(graph_label,), timeout=30)
            root = args.output.parent / label
            root.mkdir(exist_ok=True)
            phase = dict(selection=selection, natural=[], teacher=[], completions=[])
            report["phases"][label] = phase
            save()
            params = SamplingParams(**reference["sampling"], seed=20261005)
            for prompt in reference["rows"]:
                phase["natural"].append(
                    natural_row(llm, prompt, prompt["prompt_token_ids"], params)
                )
                save()
            for prompt in (
                "请用一句话解释什么是张量并行。",
                "What is 17 + 25? Give only the number.",
                "写一句简短的睡前晚安祝福。",
            ):
                tokenizer = llm.get_tokenizer()
                tokens = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                result = llm.generate(
                    {"prompt_token_ids": tokens},
                    SamplingParams(temperature=0, max_tokens=512),
                    use_tqdm=False,
                )[0].outputs[0]
                phase["completions"].append(
                    dict(
                        prompt=prompt,
                        text=result.text,
                        token_ids=result.token_ids,
                        finish_reason=result.finish_reason,
                    )
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
                payload = torch.load(root / (key + ".pt"), weights_only=True)
                assert (
                    payload["position"].item() == len(prefix)
                    and payload["input_ids"].item() == forced
                )
                phase["teacher"].append(
                    dict(
                        key=key,
                        prefix_sha256=digest(prefix),
                        position=len(prefix),
                        forced=forced,
                    )
                )
                save()
            if label != "original":
                left = (
                    "original"
                    if label in ("recaptured_control", "original_again")
                    else "recaptured_control"
                )
                report["checks"][label] = quality(
                    args.output.parent, report["phases"], left, label
                )
                check = report["checks"][label]
                print(
                    json.dumps(
                        dict(
                            phase=label,
                            **{k: v for k, v in check.items() if k != "teacher"},
                        )
                    ),
                    flush=True,
                )
                save()
                if label in ("recaptured_control", "original_again") and (
                    check["teacher_exact"],
                    check["natural_exact"],
                    check["acceptance_exact"],
                ) != (64, 8, 8):
                    report["stop_reason"] = (
                        "Same-process control did not reproduce; no timing claim"
                    )
                    save()
                    return
        fixed = reference["rows"][1]["prompt_token_ids"]
        for mode in ("scorer", "native"):
            report["captures"][mode] = llm.collective_rpc(
                capture, args=(mode, mode), timeout=180
            )
            save()

        for width, input_len, output_len in ((1, 8192, 256), (4, 128, 600)):
            ids = (fixed * (input_len // len(fixed) + 1))[:input_len]
            params = SamplingParams(
                temperature=0, top_p=1, top_k=-1, max_tokens=output_len, ignore_eos=True
            )
            for candidate in ("scorer", "native", "both"):
                for arm in ("recaptured_control", candidate):
                    llm.collective_rpc(select, args=(arm,), timeout=30)
                    observed_cohort(llm, ids, params, width)
                for arm in (
                    "recaptured_control",
                    candidate,
                    candidate,
                    "recaptured_control",
                ):
                    llm.collective_rpc(select, args=(arm,), timeout=30)
                    steps, outputs = observed_cohort(llm, ids, params, width)
                    probe = dict(
                        width=width,
                        candidate=candidate,
                        arm=arm,
                        steps=steps,
                        summary=summarize(steps, width),
                        output_token_ids=[
                            list(o.outputs[0].token_ids) for o in outputs
                        ],
                    )
                    report["probes"].append(probe)
                    save()
                    print(
                        json.dumps(
                            {
                                k: v
                                for k, v in probe.items()
                                if k not in ("steps", "output_token_ids")
                            }
                        ),
                        flush=True,
                    )
        report["complete"] = True
        save()
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
