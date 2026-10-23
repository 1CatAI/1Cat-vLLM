# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model geometry used by graph qualification, without platform decisions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GraphModelContract:
    speculative: bool
    max_model_len: int | None
    num_heads: int | None
    num_kv_heads: int | None
    head_dim: int | None
    index_topk: int | None
    compress_ratios: tuple[int, ...]
    cache_dtype: str | None
    attention_backend: str | None

    def gqa_shape(self, ratios):
        return bool(
            isinstance(self.num_heads, int)
            and isinstance(self.num_kv_heads, int)
            and self.num_kv_heads > 0
            and self.num_heads % self.num_kv_heads == 0
            and self.num_heads // self.num_kv_heads in ratios
            and self.head_dim == 256
        )

    def compressed_context_buckets(self):
        if not isinstance(self.index_topk, int) or not self.compress_ratios:
            return ()
        short = self.index_topk * min(self.compress_ratios)
        if short <= 0 or self.max_model_len <= short:
            return ()
        return tuple(
            short * m for m in (1, 2, 8, 32, 64) if short * m < self.max_model_len
        )


def model_graph_contract(cfg):
    model = cfg.model_config
    text = getattr(model, "hf_text_config", None)
    hf = getattr(model, "hf_config", None)
    ratios = getattr(hf, "compress_ratios", None)
    backend = getattr(getattr(cfg, "attention_config", None), "backend", None)
    return GraphModelContract(
        cfg.speculative_config is not None,
        getattr(model, "max_model_len", None),
        getattr(text, "num_attention_heads", None),
        getattr(text, "num_key_value_heads", None),
        getattr(text, "head_dim", None),
        getattr(hf, "index_topk", None),
        tuple(r for r in ratios if isinstance(r, int) and r > 0)
        if isinstance(ratios, (list, tuple))
        else (),
        getattr(getattr(cfg, "cache_config", None), "cache_dtype", None),
        getattr(backend, "name", backend),
    )
