# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The expert cache resolve check catches a dispatched call-shape mismatch."""

from pathlib import Path

from tools.pre_commit.check_expert_cache_resolve import violations

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "vllm" / "model_executor" / "layers" / "quantization"

CACHE = """\
class ExpertCache:
    def resolve(self, layer, ids{weights}):
        return ids


def create_cache(banks, num_experts, config, device) -> ExpertCache:
    return ExpertCache()
"""

STORAGE = """\
class SharedOriginalExpertCache:
    def resolve(self, layer, ids, weights=None):
        return ids


def _prepare_shared_cache(banks, config, device):
    cache = SharedOriginalExpertCache(banks, config, device)
    for method in banks:
        method.expert_cache = cache


def _prepare_expert_cache(banks, config, device):
    from vllm.model_executor.layers.quantization import gguf_expert_cache

    cache = gguf_expert_cache.create_cache(banks, 8, config, device)
    for method in banks:
        method.expert_cache = cache
"""

MOE = """\
def forward(self, tokens, topk_weights):
    # The prefill split and the decode fast path share one call shape.
    if tokens <= 6:
        return self.expert_cache.resolve(self.cache_layer, tokens, topk_weights)
    return self.expert_cache.resolve(self.cache_layer, tokens, topk_weights)
"""

UNRESOLVABLE = """\
def _prepare_unknown_cache(banks):
    cache = build_cache(banks)
    for method in banks:
        method.expert_cache = cache
"""


def _write(directory, cache, storage=STORAGE, moe=MOE):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "gguf_expert_cache.py").write_text(cache)
    (directory / "gguf_expert_storage.py").write_text(storage)
    (directory / "gguf_moe.py").write_text(moe)


def test_repository_expert_caches_are_call_compatible():
    assert violations(PACKAGE) == []


def test_missing_weight_parameter_is_rejected(tmp_path):
    _write(tmp_path, CACHE.format(weights=""))
    messages = violations(tmp_path)
    assert len(messages) == 2
    assert all("ExpertCache.resolve" in message for message in messages)
    assert all("passes 3" in message for message in messages)


def test_declared_weight_parameter_satisfies_the_dispatch(tmp_path):
    _write(tmp_path, CACHE.format(weights=", weights=None"))
    assert violations(tmp_path) == []


def test_shared_storage_route_is_checked(tmp_path):
    _write(tmp_path, CACHE.format(weights=", weights=None"), storage=STORAGE)
    assert violations(tmp_path) == []
    _write(
        tmp_path,
        CACHE.format(weights=", weights=None"),
        storage=STORAGE.replace("weights=None", "weights"),
    )
    assert violations(tmp_path) == []


def test_unresolvable_binding_is_reported(tmp_path):
    _write(tmp_path, CACHE.format(weights=", weights=None"), storage=UNRESOLVABLE)
    messages = violations(tmp_path)
    assert len(messages) == 1
    assert "'build_cache' is bound to 'expert_cache'" in messages[0]
