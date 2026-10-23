# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from tests.config.test_flash_v100_lifecycle import prepare
from vllm.config import set_current_vllm_config
from vllm.config.flash_v100 import FlashV100Diagnostics, FlashV100Options
from vllm.runtime_resources import runtime_resources_for
from vllm.v1.attention.backends.flash_v100 import runtime

pytestmark = pytest.mark.cpu_test


def options(**kwargs):
    return FlashV100Options(
        fa2_d256_prefill=True,
        prefill_d256_gqa_arch_128k_experimental=True,
        prefill_d256_gqa_v37=False,
        **kwargs,
    )


@pytest.mark.parametrize(
    "raw,algorithm,block,serial,present",
    [
        (None, 0, 0, 1, 0),
        ("", 0, -1, 1, 1),
        ("0", 0, -1, 0, 1),
        ("01", 1, -1, 1, 1),
        ("  +8192", 8192, 8192, 1, 1),
        ("8192\n", 8192, -1, 1, 1),
        ("131072", 131072, 131072, 1, 1),
        ("139264", 139264, -1, 1, 1),
        ("-1suffix", -1, -1, 1, 1),
        ("\u00a08192", 0, -1, 1, 1),
        ("999999999999999999999999", -1, -1, 1, 1),
    ],
)
def test_legacy_native_dialects(monkeypatch, raw, algorithm, block, serial, present):
    for name in (
        "PREFIX_QK_CUBLAS_ALGO_RUNTIME",
        "VLLM_FLASH_V100_PREFILL_SCORE_BLOCK_TOKENS",
        "PREFIX_TORCH_SERIAL_TAIL",
        "PREFIX_TORCH_EXACT_TAIL",
        "PREFIX_TORCH_DUMP_TAIL",
        "PREFIX_TORCH_DIRECT_TAIL",
    ):
        if raw is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, raw)
    policy = prepare(options())[0]
    assert policy.options.prefill_native_effective == (
        present,
        algorithm,
        block,
        serial,
        present,
        present,
        present,
    )
    snapshot = policy.explain()
    monkeypatch.setattr("os.getenv", lambda *a: pytest.fail("runtime env read"))
    policy.resolve()
    assert policy.explain() == snapshot


def test_typed_overrides_presence_and_bad_score_without_rewriting_env(monkeypatch):
    monkeypatch.setenv("PREFIX_TORCH_EXACT_TAIL", "")
    monkeypatch.setenv("VLLM_FLASH_V100_PREFILL_SCORE_BLOCK_TOKENS", "invalid")
    monkeypatch.setenv("PREFIX_TORCH_DUMP_TAIL", "0")
    policy = prepare(
        options(prefill_exact_tail=False, prefill_score_block_tokens=16384),
        trace=FlashV100Diagnostics(prefill_dump_tail=False),
    )[0]
    values = policy.options.prefill_native_effective
    assert values[2] == 16384 and values[4:6] == (0, 0)
    assert policy.options.sources["prefill_score_block_tokens"] == "typed"
    assert policy.options.legacy_inputs["prefill_score_block_tokens"] == "invalid"


def test_prefill_hash_tracks_only_effective_computation():
    normal = prepare(options())[0]
    dump = prepare(options(), trace=FlashV100Diagnostics(prefill_dump_tail=True))[0]
    exact = prepare(options(prefill_exact_tail=True))[0]
    assert normal.compute_hash() == dump.compute_hash()
    assert normal.compute_hash() != exact.compute_hash()
    off = prepare(FlashV100Options(prefill_d256_gqa_arch_128k_experimental=False))[0]
    unused = prepare(
        FlashV100Options(
            prefill_d256_gqa_arch_128k_experimental=False,
            prefill_exact_tail=True,
            prefill_score_block_tokens=16384,
        )
    )[0]
    assert off.compute_hash() == unused.compute_hash()


def engine(**kwargs):
    return SimpleNamespace(
        attention_config=SimpleNamespace(flash_v100=prepare(options(**kwargs))[0])
    )


def test_missing_optional_operator_falls_back_but_old_binary_fails_at_binding():
    with set_current_vllm_config(engine()):
        runtime.prepare_prefill_runtime(SimpleNamespace())
        assert runtime.bind_prefill_operation(None, 8000) is None
        with pytest.raises(RuntimeError, match="policy ABI 1"):
            runtime.prepare_prefill_runtime(
                SimpleNamespace(sm70_d256_gqa_architecture_fwd=lambda: None)
            )


def test_two_owners_bind_once_and_close_independently(monkeypatch):
    native = SimpleNamespace(sm70_d256_gqa_architecture_fwd=object())
    created = []

    class Owner:
        def __init__(self, options, native):
            self.values = options.prefill_native_effective
            self.operations = {8000: object(), 8192: object()}
            self.closed = False
            created.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(runtime, "PrefillRuntime", Owner)
    engines = [engine(prefill_serial_tail=value) for value in (False, True)]
    for e in engines:
        with set_current_vllm_config(e):
            runtime.prepare_prefill_runtime(native)
    monkeypatch.setattr("os.getenv", lambda *a: pytest.fail("execution env read"))
    for index in (1, 0, 1, 0):
        with set_current_vllm_config(engines[index]):
            runtime.prepare_prefill_runtime(native)
            assert (
                runtime.bind_prefill_operation(object(), 8000)
                is created[index].operations[8000]
            )
    assert len(created) == 2
    assert [owner.values[3] for owner in created] == [0, 1]
    from vllm.runtime_resources import release_runtime_resources

    release_runtime_resources(engines[0])
    assert created[0].closed and not created[1].closed
    assert runtime_resources_for(engines[1])["sm70_prefill"] is created[1]


def test_missing_precision_qualification_keeps_dense_fallback(monkeypatch):
    from vllm.v1.attention.backends.flash_v100 import ops

    for name in (
        "_flash_attn_func",
        "_flash_attn_decode_paged",
        "_flash_attn_prefill_paged",
    ):
        monkeypatch.setattr(ops, name, object())
    monkeypatch.setattr(ops, "prepare_attention_runtime", lambda: None)
    monkeypatch.setattr(ops, "get_sm70_splitd_d256_ops", lambda: None)
    monkeypatch.setattr(ops, "_sm70_gqa_has_fp32_accumulation", lambda: False)
    monkeypatch.setattr(ops, "bind_attention_operation", lambda operation: operation)
    monkeypatch.setattr(
        ops,
        "prepare_prefill_runtime",
        lambda *a: pytest.fail("unqualified FA2 tried policy binding"),
    )
    with set_current_vllm_config(engine()):
        assert len(ops.get_flash_ops()) == 9
