# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research-only gated norm epilogue, retaining the BV2 delta geometry.

Each head's first value tile waits for its own output payload, then normalizes
eight rows. Other value tiles never wait. A signalling-NaN payload sentinel is
initialized by the existing convolution kernel, not by an additional launch.
This is restricted to valid, finite, single-request M8 inputs. It is not a
production synchronization protocol or a serving performance claim.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

from vllm.model_executor.layers.fla.ops import fused_sigmoid_gating as gdn
from vllm.model_executor.layers.mamba.gdn import sm70_preprocess


def candidate_module(directory):
    source = Path(gdn.__file__).read_text()
    begin = source.index("@triton.heuristics(")
    end = source.index("\ndef fused_sigmoid_gating_delta_rule_update(", begin)
    kernel = (
        source[begin:end]
        .replace(
            "def fused_sigmoid_gating_delta_rule_update_kernel(",
            "def norm_epilogue_kernel(",
            1,
        )
        .replace(
            "    A_log,\n",
            "    NormWeight,\n    NormGate,\n    Normalized,\n    A_log,\n",
            1,
        )
    )
    kernel += """
    if i_v == 0:
        row = tl.arange(0, 8)
        col = tl.arange(0, 128)
        offset = ((bos + row[:, None]) * HV + i_hv) * V + col[None, :]
        bits = tl.inline_asm_elementwise("ld.global.cg.b16 $0, [$1];",
            constraints="=h,l", args=[o + offset], dtype=tl.uint16,
            is_pure=False, pack=1)
        pending = tl.sum((bits == 0x7c01).to(tl.int32))
        while pending != 0:
            bits = tl.inline_asm_elementwise("ld.global.cg.b16 $0, [$1];",
                constraints="=h,l", args=[o + offset], dtype=tl.uint16,
                is_pure=False, pack=1)
            pending = tl.sum((bits == 0x7c01).to(tl.int32))
        value = bits.to(tl.float16, bitcast=True).to(tl.float32)
        variance = tl.sum(value * value, 1) / 128.0
        inv = tl.rsqrt(variance + 1e-6)
        weight = tl.load(NormWeight + col).to(tl.float32)
        gate = tl.load(NormGate + offset).to(tl.float32)
        result = ((value * inv[:, None]) * weight[None, :]) * (
            gate * tl.sigmoid(gate))
        tl.store(Normalized + offset, result)
"""
    prep = Path(sm70_preprocess.__file__).read_text()
    prep = prep[: prep.index("\ndef conv_gate_zero(")]
    prep = prep.replace("def _conv_gate_zero_kernel(", "def sentinel_conv_kernel(", 1)
    marker = "        0,\n        (token[:, None] < TOKENS)"
    assert prep.count(marker) == 1
    prep = prep.replace(
        marker,
        "        tl.full((), 0x7c01, tl.uint16).to(tl.float16, bitcast=True),\n"
        "        (token[:, None] < TOKENS)",
    )
    path = directory / "gdn_norm_epilogue_generated.py"
    text = (
        "from vllm.triton_utils import triton, tl\n"
        "from vllm.model_executor.layers.fla.ops.op import exp\n\n"
        + kernel
        + "\n"
        + prep
    )
    path.write_text(text)
    spec = importlib.util.spec_from_file_location("gdn_norm_epilogue_generated", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (directory / "source-hashes.json").write_text(
        json.dumps(
            {
                "delta_source": hashlib.sha256(source.encode()).hexdigest(),
                "generated": hashlib.sha256(text.encode()).hexdigest(),
            },
            indent=2,
        )
    )
    return module


class LaunchAdapter:
    def __init__(self, kernel, norm, gate, normalized):
        self.kernel, self.norm, self.gate = kernel, norm, gate
        self.normalized = normalized
        self.compiled = None

    def __getitem__(self, grid):
        def invoke(**kwargs):
            # The replay helper provides its own candidate-only arguments.
            for key in list(kwargs):
                if key.startswith(("replay_", "save_")) or key in (
                    "REPLAY_PREFIX",
                    "SAVE_HISTORY",
                    "CACHE_FACTORS",
                ):
                    del kwargs[key]
            kwargs.update(
                NormWeight=self.norm, NormGate=self.gate, Normalized=self.normalized
            )
            self.compiled = self.kernel[grid](**kwargs)

        return invoke


def prepare(
    module, q, history, conv, indices, accepted, cu, a_log, a, b, bias, core, sentinel
):
    tokens = q.shape[0]
    g = torch.empty((1, tokens, 12), device=q.device, dtype=torch.float32)
    beta = torch.empty_like(g)
    kernel = (
        module.sentinel_conv_kernel
        if sentinel
        else sm70_preprocess._conv_gate_zero_kernel
    )
    kernel[(1, 40)](
        q,
        conv,
        history,
        indices,
        accepted,
        cu,
        g,
        beta,
        a_log,
        a,
        b,
        bias,
        core,
        q.stride(0),
        *conv.stride(),
        *history.stride(),
        history.shape[0],
        tokens,
        tokens + 2,
        8,
        16,
        64,
        num_warps=2,
    )
    return q, g.view(8, 12), beta.view(8, 12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.manual_seed(123)
    assert torch.cuda.get_device_capability() == (7, 0)
    module = candidate_module(args.out)
    # Reuse the exact real-weight layer setup; change only its core and norm.
    source_path = Path(__file__).with_name("benchmark_sm70_gdn_rollback_layer.py")
    text = source_path.read_text().replace("a" + "log", "a_log")
    text = text.replace("import rmsnorm_fn", "import layer_norm_fwd")
    text = text.replace(
        "    candidate = candidate_module(args.out).rollback_replay_kernel",
        "    candidate = LaunchAdapter(\n"
        "        module.norm_epilogue_kernel, norm, z, normalized)",
    )
    text = text.replace("    def prepare():", "    def prepare(use_replay=False):")
    start = text.index("        transformed = causal_conv1d_update(")
    stop = text.index("\n    history.copy_(history_seed)", start)
    text = (
        text[:start]
        + """        mixed, g, beta = epilogue_prepare(
            module, q, history, conv, conv_indices, accepted, cu,
            a_log, a, b, bias, core.view(8, 12, 128), use_replay)
        return mixed, g, beta, res
"""
        + text[stop:]
    )
    # No replay/cache preparation: both paths write the same eight snapshots.
    start = text.index(
        "    history.copy_(history_seed)\n    mixed, g, beta, _ = prepare()"
    )
    stop = text.index("    def run(use_replay):", start)
    text = text[:start] + "    snapshot_seed = snapshots.clone()\n\n" + text[stop:]
    text = text.replace(
        "mixed, g, beta, res = prepare()", "mixed, g, beta, res = prepare(use_replay)"
    )
    start = text.index("        if use_replay:\n            launch(")
    stop = text.index("        ops.fp8_qpn8_gemm_sm70_out(", start)
    text = (
        text[:start]
        + """        launch(candidate if use_replay else reference_kernel,
               mixed, g, beta, snapshots, indices, core,
               metadata=(cu, accepted))
        if not use_replay:
            layer_norm_fwd(core.view(96, 128), norm, None,
                z=z.reshape(96, 128), eps=1e-6,
                norm_before_gate=True, is_rms_norm=True,
                out=normalized.view(96, 128))
"""
        + text[stop:]
    )
    # Use a stable destination in both arms, without a reference-only copy.
    text = text.replace(
        "    candidate = LaunchAdapter",
        "    normalized = x.new_empty(8, 1536)\n    candidate = LaunchAdapter",
    )
    text = text.replace(
        "    torch.testing.assert_close(down, expected, rtol=0, atol=0)",
        "    torch.testing.assert_close(down, expected, rtol=0.01, atol=0.005)\n"
        "    layer_max_diff = float((down - expected).abs().max())\n"
        "    assert torch.isfinite(down).all()\n"
        "    assert candidate.compiled.metadata.num_warps == 1\n"
        "    print('CANDIDATE_REGISTERS', candidate.compiled.n_regs, flush=True)",
    )
    text = text.replace(
        "    expected = down.clone()",
        "    expected = down.clone()\n"
        "    expected_states = snapshots.clone()\n"
        "    expected_core = core.clone()",
    )
    text = text.replace(
        "    compact.copy_(initial)\n    run(True)",
        "    snapshots.copy_(snapshot_seed)\n    run(True)\n"
        "    assert torch.equal(snapshots.view(torch.int32),\n"
        "                       expected_states.view(torch.int32))\n"
        "    assert torch.equal(core.view(torch.int16),\n"
        "                       expected_core.view(torch.int16))",
    )
    text = text.replace(
        "final_layer_output_bitwise=True,",
        "layer_max_diff=layer_max_diff,\n        numerical_gate_passed=False,\n"
        "        original_bv=2, candidate_bv=2,\n"
        "        candidate_registers=candidate.compiled.n_regs,",
    )
    text = text.replace('"layer-result.json"', '"norm-epilogue-layer-result.json"')
    text = text.replace('if __name__ == "__main__":\n    main()', "")
    # Use this screen's parsed parameters while preserving the original loader.
    globals_dict = {
        "__file__": str(source_path),
        "__name__": "layer_epilogue_screen",
        "LaunchAdapter": LaunchAdapter,
        "module": module,
        "epilogue_prepare": prepare,
    }
    exec(
        compile(text, str(args.out / "layer-screen-generated.py"), "exec"), globals_dict
    )
    (args.out / "layer-screen-generated.py").write_text(text)
    globals_dict["main"]()


if __name__ == "__main__":
    main()
