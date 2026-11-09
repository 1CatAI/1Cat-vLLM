# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare oracle building blocks with a local official V4.1 reference.

Run on CPU with --reference-dir pointing to the reviewed official inference
sources. This validates components only; it does not generate model logits.
"""

import argparse
import ast
import hashlib
import json
import resource
import runpy
from pathlib import Path
from types import SimpleNamespace

import gguf
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from benchmarks.deepseek_v41.oracle_primitives import (
    EngramHashState,
    GGUFWeights,
    engram_update,
    rms_norm,
    route,
)


def reference_method(path: Path, class_name: str, method_name: str):
    """Load one reviewed method without importing vision or GPU-only modules."""
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    scope = {"torch": torch, "F": F, "linear": F.linear}
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(module, str(path), "exec"), scope)
    return scope[method_name]


def compare(root: Path, reference_dir: Path) -> dict:
    torch.set_num_threads(4)
    torch.manual_seed(41)
    config = json.loads((root / "base/config.json").read_text())["text_config"]
    args = SimpleNamespace(
        **{key: value for key, value in config.items() if key.startswith("engram_")},
        max_batch_size=2,
        max_seq_len=2056,
    )
    args.engram_pad_id = args.engram_pad_token_id
    official = runpy.run_path(str(reference_dir / "engram.py"))
    layout = official["EngramLayout"].from_args(args)
    tokenizer = AutoTokenizer.from_pretrained(root / "base", local_files_only=True)
    expected = official["NgramHashState"](args, layout, tokenizer)
    weights = GGUFWeights(sorted((root / "gguf").glob("*.gguf")))
    token_map = weights.integer("engram_token_map.weight")
    primes = weights.integer("engram_primes.weight")
    offsets = weights.integer("engram_offsets.weight")
    multipliers = weights.integer("engram_multipliers.weight")
    for name, actual in (
        ("token_map", token_map),
        ("primes", primes.reshape_as(expected.primes)),
        ("offsets", offsets),
        ("multipliers", multipliers),
    ):
        torch.testing.assert_close(actual, getattr(expected, name), rtol=0, atol=0)

    def new_hash():
        return EngramHashState(
            token_map, primes, offsets, multipliers, args.engram_pad_token_id
        )

    hash_cases = []
    for length in (1, 2, 3, 4, 64, 2048):
        ids = torch.randint(len(token_map), (2, length + 3))
        mask = torch.ones_like(ids, dtype=torch.bool)
        mask[0, ::7] = False
        if length > 1:
            mask[1, length - 1] = False
        reference = expected(ids, 0, mask)
        full = new_hash()(ids, 0, mask)
        torch.testing.assert_close(full, reference, rtol=0, atol=0)
        # Split at prefill/decode and within the 4-gram lookback window.
        for split in sorted({1, min(3, length), length}):
            state = new_hash()
            chunks = [state(ids[:, :split], 0, mask[:, :split])]
            for position in range(split, ids.shape[1]):
                chunks.append(
                    state(
                        ids[:, position : position + 1],
                        position,
                        mask[:, position : position + 1],
                    )
                )
            torch.testing.assert_close(
                torch.cat(chunks, dim=1), reference, rtol=0, atol=0
            )
        hash_cases.append({"prefill_tokens": length, "decode_tokens": 3})

    model = reference_dir / "model.py"
    norm = reference_method(model, "RMSNorm", "forward")
    x = torch.randn(2, 7, 16) * 1e-12
    norm_weight = torch.randn(16)
    torch.testing.assert_close(
        rms_norm(x, norm_weight, 1e-20),
        norm(SimpleNamespace(weight=norm_weight, eps=1e-20), x),
        rtol=1e-6,
        atol=1e-7,
    )
    router = reference_method(model, "Gate", "forward")
    router_state = SimpleNamespace(
        weight=torch.randn(384, 16),
        bias=torch.randn(384),
        bias_vl=None,
        topk=6,
        score_func="sqrtsoftplus",
        gate_temp=1.0,
        norm_topk_prob=True,
        route_scale=1.5,
    )
    x = torch.randn(13, 16)
    actual_weights, actual_ids = route(x, router_state.weight, router_state.bias)
    expected_weights, expected_ids = router(router_state, x)
    torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=0, atol=0)

    update = reference_method(model, "Engram", "forward")
    gate_errors = []
    for factor in (0.0, 1e-12, 1.0, 1e6):
        hidden = torch.randn(2, 5, 4, 16) * factor
        kv = torch.randn(2, 5, 80) * factor
        q, k = torch.randn(4, 16), torch.randn(4, 16)
        mask = torch.tensor([[True, False, True, False, True]]).expand(2, -1)
        state = SimpleNamespace(
            wkv=lambda value: value,
            embed=lambda value, kv=kv: kv.unsqueeze(-2),
            hc_mult=4,
            dim=16,
            q_weight=q,
            k_weight=k,
            eps=1e-20,
            clamp_value=1e-6,
        )
        reference = update(state, hidden, torch.zeros(2, 5, 24), mask)
        actual = engram_update(hidden, kv, q, k, mask)
        torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-7)
        assert torch.isfinite(actual).all()
        assert torch.equal(actual[~mask], hidden[~mask])
        gate_errors.append(float((actual - reference).abs().max()))

    # Real IQ4_NL table rows plus chunked Q5_K projection for both Engram layers.
    real_rows = []
    ids = torch.tensor([[0, 2]])
    hashes = new_hash()(ids, 0)
    for index, layer in enumerate(args.engram_layer_ids):
        name = f"blk.{layer}.engram_embd.weight"
        row_ids = hashes[:, :, index]
        embeddings = weights.rows(name, row_ids)
        tensor = weights.tensors[name]
        for position in range(2):
            for head in range(24):
                row = int(row_ids[0, position, head])
                raw = tensor.data[row : row + 1]
                independent = torch.from_numpy(
                    gguf.quants.dequantize(raw, tensor.tensor_type).copy()
                ).flatten()
                torch.testing.assert_close(
                    embeddings[0, position, head], independent, rtol=0, atol=0
                )
        kv = weights.linear(f"blk.{layer}.engram_wkv.weight", embeddings.flatten(-2))
        assert kv.shape == (1, 2, 25600) and torch.isfinite(kv).all()
        real_rows.append({"layer": layer, "lookup_rows": row_ids.numel()})
    return {
        "status": "component_checks_passed",
        "scope": "M1 building blocks only; no full-model logits or text validation",
        "reference_sha256": {
            name: hashlib.sha256((reference_dir / name).read_bytes()).hexdigest()
            for name in ("model.py", "engram.py")
        },
        "checkpoint_hash_auxiliaries_match_official": True,
        "hash_cases": hash_cases,
        "rms_norm_and_router_match_official": True,
        "engram_gate_max_absolute_errors": gate_errors,
        "real_engram_row_and_projection_checks": real_rows,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with torch.inference_mode():
        result = compare(args.model_root, args.reference_dir)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
