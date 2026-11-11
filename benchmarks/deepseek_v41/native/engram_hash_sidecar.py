# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the V4.1 Engram hash tables exactly as the official reference code
(checkpoint inference/engram.py) does, in the target's parameter layout:
token_map [vocab] int32, primes/offsets [layers, (max_ngram-1)*heads] int64,
multipliers [layers, max_ngram] int64. Usage: MODEL_DIR OUT.safetensors"""

import json
import sys

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

model_dir, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, model_dir + "/inference")
from engram import (  # noqa: E402
    EngramLayout,
    build_compressed_token_map,
    compute_hash_multipliers,
)

cfg = json.load(open(model_dir + "/config.json"))
text = cfg.get("text_config", cfg)


class Args:
    engram_layer_ids = text["engram_layer_ids"]
    engram_num_embeddings = text["engram_num_embeddings"]
    engram_max_ngram_size = text["engram_max_ngram_size"]
    engram_vocab_size = text["engram_vocab_size"]
    engram_n_heads = text["engram_n_heads"]
    engram_head_dim = text["engram_head_dim"]


layout = EngramLayout.from_args(Args)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
token_map, size = build_compressed_token_map(tokenizer)
assert size == text["engram_compressed_vocab_size"], (
    size,
    text["engram_compressed_vocab_size"],
)
flat = [[p for per in layer for p in per] for layer in layout.primes]
offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]
multipliers = compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, size)
vocab = text["vocab_size"]
full = torch.zeros(vocab, dtype=torch.int32)
full[: len(token_map)] = torch.tensor(token_map, dtype=torch.int32)
tables = {
    "model.engram_token_map": full,
    "model.engram_primes": torch.tensor(flat, dtype=torch.int64),
    "model.engram_offsets": torch.tensor(np.array(offsets), dtype=torch.int64),
    "model.engram_multipliers": multipliers.to(torch.int64),
}
for layer, rows in enumerate(layout.num_embeddings):
    end = int(
        tables["model.engram_offsets"][layer, -1]
        + tables["model.engram_primes"][layer, -1]
    )
    assert end == rows, (layer, end, rows)
for k, v in tables.items():
    print(k, tuple(v.shape), v.dtype)
print("tokenizer", len(tokenizer), "compressed", size)
save_file(tables, out)
print("SIDECAR_OK", out)
