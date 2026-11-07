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
        buckets = [m * short for m in (1, 2) if m * short < self.max_model_len]
        # A decode round costs time with the width of its bucket, not with the
        # context, so a context just above a bucket pays for the whole gap to
        # the next one. From 8 * short on, the buckets are a step apart that
        # starts at 8 * short and doubles each time the bucket passes four
        # times the bucket where the step last changed.
        bucket = step = 8 * short
        widen_at = 4 * bucket
        while bucket < self.max_model_len:
            buckets.append(bucket)
            if bucket >= widen_at:
                step *= 2
                widen_at *= 4
            bucket += step
        # Above the largest bucket a decode graph would be bound by the model's
        # maximum length, which keeps the indexer off its cuBLAS decode route;
        # a last bucket at the maximum length keeps it on.
        return (*buckets, self.max_model_len)


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
