# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the V4.1 adapter against real GGUF headers without loading weights."""

import argparse
import hashlib
import inspect
import json
from collections import Counter
from pathlib import Path

import gguf

from vllm.model_executor.model_loader.gguf_adapters.deepseek_v41 import (
    DeepseekV41Adapter,
)
from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config


def audit(root: Path):
    config = DeepseekV41Config.from_pretrained(root / "base", local_files_only=True)
    adapter = DeepseekV41Adapter(config.get_text_config(), tp_size=4)
    tensors, files = {}, []
    for path in sorted((root / "gguf").glob("*.gguf")):
        reader = gguf.GGUFReader(path, mode="r")
        architecture = reader.get_field("general.architecture")
        if architecture and architecture.contents() == "deepseek41-engram":
            continue
        files.append({"path": str(path), "tensors": len(reader.tensors)})
        for tensor in reader.tensors:
            if tensor.name in tensors:
                raise ValueError(f"Duplicate backbone tensor: {tensor.name}")
            tensors[tensor.name] = tensor
    mapping = adapter.build_name_map(tensors)
    manifest = [
        (
            raw,
            list(map(int, tensors[raw].shape)),
            int(tensors[raw].tensor_type),
            mapping[raw],
        )
        for raw in sorted(mapping)
    ]
    return {
        "status": "passed",
        "model_type": config.model_type,
        "text_model_type": config.text_config.model_type,
        "backbone_files": files,
        "tensor_count": len(mapping),
        "unique_target_count": len(set(mapping.values())),
        "expert_tensor_count": sum("_exps.weight" in raw for raw in mapping),
        "source_types": dict(Counter(t.tensor_type.name for t in tensors.values())),
        "mapping_sha256": hashlib.sha256(json.dumps(manifest).encode()).hexdigest(),
        "adapter_sha256": hashlib.sha256(
            Path(inspect.getfile(DeepseekV41Adapter)).read_bytes()
        ).hexdigest(),
        "scope": "Header and mapping validation only; no model forward executed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.model_root)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
