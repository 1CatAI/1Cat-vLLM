# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research complete post-convolution GDN core plus gated RMSNorm screen.

M5 records all sequential recurrent states; runtime metadata/scatter admission
is excluded. This numerical oracle is independent FP32 PyTorch arithmetic.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from benchmarks.kernels.sm70_chain_screen_utils import graph_kernel_geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).parents[1] / "csrc/sm70_gdn_head_chain_screen.cu"
    extension = load(
        "sm70_gdn_head_chain_screen",
        [str(source)],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70"],
        verbose=True,
    )
    if args.build_only:
        args.output.write_text(json.dumps({"library": extension.__file__}) + "\n")
        return
    import vllm._C  # noqa: F401

    from flash_qla.ops.gated_delta_rule.chunk.sm70.fused_fwd import (
        gdn_decode_mixed_qkv_global_state_sm70,
    )
    from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update_mixed_qkv,
    )

    report = {
        "research_only": True,
        "scope": (
            "post-convolution recurrent core and gated norm; no projection/conv/scatter"
        ),
        "widths": [],
    }
    torch.manual_seed(4201)

    def screen(m):
        qkv = torch.randn(m, 2560, device="cuda", dtype=torch.float16) * 0.1
        a, b = [
            torch.randn(m, 12, device="cuda", dtype=torch.float16) for _ in range(2)
        ]
        al = torch.randn(12, device="cuda", dtype=torch.float32) * 0.1
        bias = torch.randn(12, device="cuda", dtype=torch.float16) * 0.1
        z = torch.randn(m, 12, 128, device="cuda", dtype=torch.float16)
        weight = torch.randn(128, device="cuda", dtype=torch.float16) * 0.1 + 1
        state = torch.randn(12, 128, 128, device="cuda", dtype=torch.float32) * 0.01
        final_states = torch.empty(m, 12, 128, 128, device="cuda", dtype=torch.float32)
        output = torch.empty(m, 12, 128, device="cuda", dtype=torch.float16)
        reference_state = state.clone()
        expected, states = [], []
        for row in range(m):
            # Use the kernel's sqrt(sum + epsilon) normalization explicitly.
            q0 = qkv[row, :512].float().view(4, 128)
            k0 = qkv[row, 512:1024].float().view(4, 128)
            q = (
                q0 * (q0.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
            ).repeat_interleave(3, dim=0)
            k = (
                k0 * (k0.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
            ).repeat_interleave(3, dim=0)
            decay = torch.exp(
                -al.exp() * torch.nn.functional.softplus(a[row].float() + bias.float())
            )
            kv = (reference_state * k[:, None, :]).sum(-1)
            delta = (qkv[row, 1024:].float().view(12, 128) - decay[:, None] * kv) * b[
                row
            ].float().sigmoid()[:, None]
            reference_state = (
                reference_state * decay[:, None, None]
                + delta[:, :, None] * k[:, None, :]
            )
            states.append(reference_state.clone())
            core = ((reference_state * q[:, None, :]).sum(-1) / 128**0.5).half().float()
            y = (
                core
                * (core.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
                * weight.float()
                * torch.nn.functional.silu(z[row].float())
            )
            expected.append(y.half())

        def candidate():
            extension.run(
                qkv, a, b, al, bias, z, weight, state, final_states, output, 1e-6
            )

        candidate()
        expected = torch.stack(expected)
        errors = {
            "output_max_abs": float((expected.float() - output.float()).abs().max()),
            "output_rel_l2": float(
                (expected.float() - output.float()).norm() / expected.float().norm()
            ),
            "state_max_abs": float((torch.stack(states) - final_states).abs().max()),
        }
        native_states = torch.empty(m, 12, 128, 128, device="cuda", dtype=torch.float32)
        native_states[0].copy_(state)
        indices = torch.arange(m, device="cuda", dtype=torch.int32).view(1, m)
        cu = torch.tensor([0, m], device="cuda", dtype=torch.int32)
        accepted = torch.ones(1, device="cuda", dtype=torch.int32)
        native_output = torch.empty_like(output)
        core = torch.empty_like(output)

        def control():
            if m == 1:
                gdn_decode_mixed_qkv_global_state_sm70(
                    qkv,
                    a,
                    b,
                    al,
                    bias,
                    native_states,
                    indices.flatten(),
                    core,
                    use_qk_l2norm_in_kernel=True,
                )
            else:
                fused_sigmoid_gating_delta_rule_update_mixed_qkv(
                    A_log=al,
                    a=a,
                    b=b,
                    dt_bias=bias,
                    mixed_qkv=qkv,
                    num_q_heads=4,
                    num_v_heads=12,
                    head_k_dim=128,
                    head_v_dim=128,
                    initial_state=native_states,
                    cu_seqlens=cu,
                    ssm_state_indices=indices,
                    num_accepted_tokens=accepted,
                    use_qk_l2norm_in_kernel=True,
                    out=core.unsqueeze(0),
                )
            torch.ops._C.sm70_rmsnorm_gated_exact_out(
                native_output.view(m * 12, 128),
                core.view(m * 12, 128),
                z.view(m * 12, 128),
                weight,
                1e-6,
                True,
            )

        control()
        torch.cuda.synchronize()
        errors["native_output_max_abs"] = float(
            (native_output.float() - output.float()).abs().max()
        )
        errors["native_state_max_abs"] = float(
            (native_states - final_states).abs().max()
        )
        graphs = {}
        for name, fn in (("control", control), ("candidate", candidate)):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(36):
                    fn()
            graphs[name] = graph
        samples = {name: [] for name in graphs}
        for rep in range(7):
            for name in list(graphs)[:: 1 if rep % 2 == 0 else -1]:
                for _ in range(10):
                    graphs[name].replay()
                start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                start.record()
                for _ in range(100):
                    graphs[name].replay()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) / 100)
        counts = {
            name: graph_kernel_geometry(graph, args.output, f"M{m}.{name}")
            for name, graph in graphs.items()
        }
        report["widths"].append(
            dict(
                m=m,
                errors=errors,
                measured_kernel_counts=counts,
                samples_ms=samples,
                medians_ms={n: statistics.median(v) for n, v in samples.items()},
                control_scope=(
                    "native M1 FlashQLA or M5 mixed-QKV verifier plus gated norm"
                ),
            )
        )

    for m in (1, 5):
        screen(m)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
