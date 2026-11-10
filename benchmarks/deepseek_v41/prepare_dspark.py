# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write a small DSpark-only config/index and symlink original weight shards."""

import argparse
import json
from pathlib import Path

from benchmarks.deepseek_v41.checkpoint_audit import safetensors_names


def prepare(root: Path, destination: Path):
    original = json.loads((root / "base/config.json").read_text())
    config = dict(original["text_config"])
    if original["model_type"] != "deepseek_v41":
        raise ValueError("Expected the audited DeepSeek V4.1 checkpoint")
    config.update(
        architectures=["DSparkDeepseekV41ForCausalLM"],
        dtype="float16",
        num_hidden_layers=3,
        n_routed_experts=config["dspark_n_routed_experts"],
        num_experts_per_tok=config["dspark_num_experts_per_tok"],
        num_nextn_predict_layers=0,
        compress_ratios=[0, 0, 0],
        kv_source_layer_ids=[],
        index_source_layer_ids=[],
        engram_layer_ids=[],
        max_position_embeddings=16384,
        rope_scaling=None,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        quantization_config=original["quantization_config"],
        bos_token_id=original["bos_token_id"],
        eos_token_id=original["eos_token_id"],
        pad_token_id=original["pad_token_id"],
    )
    index = json.loads((root / "base/model.safetensors.index.json").read_text())
    expected = {k: v for k, v in index["weight_map"].items() if k.startswith("mtp.")}
    actual, total = {}, 0
    shards = sorted((root / "base/dspark").glob("*.safetensors"))
    for path in shards:
        for name, record in safetensors_names(path).items():
            if name in actual:
                raise ValueError(f"Duplicate draft tensor {name}")
            actual[name] = path.name
            start, end = record["data_offsets"]
            total += end - start
    if actual != expected or len(actual) != 2401:
        raise ValueError("DSpark shard coverage does not match the original index")
    destination.mkdir(parents=True, exist_ok=True)
    for source in shards:
        link = destination / source.name
        if link.is_symlink() and link.resolve() == source.resolve():
            continue
        if link.exists() or link.is_symlink():
            raise FileExistsError(link)
        link.symlink_to(source.resolve())
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (destination / "model.safetensors.index.json").write_text(
        json.dumps(dict(metadata=dict(total_size=total), weight_map=actual), indent=2)
        + "\n"
    )
    return dict(tensors=len(actual), payload_bytes=total, weight_conversion=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.model_root, args.destination)))


if __name__ == "__main__":
    main()
