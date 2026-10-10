# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing
import os

import pytest

from vllm.config import KernelConfig
from vllm.config.sm70_native import NATIVE_FIELDS, Sm70NativeConfig


@pytest.mark.parametrize("first,second", [(1, 8), (8, 1)])
def test_marlin_inputs_cross_worker_boundary_without_reinterpretation(
    monkeypatch, first, second
):
    engines = []
    for value in (first, second):
        monkeypatch.setenv("SM70_MARLIN_DENSE_SPLIT_K", str(value))
        cfg = KernelConfig()
        cfg.capture_provider_inputs()
        recv, send = multiprocessing.Pipe(duplex=False)
        send.send(cfg)
        engines.append(recv.recv())
        send.close()
        recv.close()
    # Capturing an unused provider must not perturb another engine's hash.
    assert engines[0].compute_hash() == engines[1].compute_hash()

    def forbidden(*args):
        raise AssertionError("worker read an initialization input")

    monkeypatch.setattr(os, "getenv", forbidden)
    index = next(
        i for i, row in enumerate(NATIVE_FIELDS) if row[0] == "marlin_dense_split_k"
    )
    for cfg, expected in zip(engines, (first, second)):
        cfg.sm70_marlin.resolve("marlin")
        assert cfg.sm70_marlin.values[index] == str(expected)
    assert engines[0].compute_hash() != engines[1].compute_hash()


def test_explicit_marlin_override_and_dormant_error(monkeypatch):
    monkeypatch.setenv("SM70_MARLIN_DENSE_SPLIT_K", "invalid")
    native = Sm70NativeConfig(marlin_dense_split_k=4)
    native.resolve("marlin")
    assert native.hash_options() == {"marlin_dense_split_k": "4"}
    assert native.sources["marlin_dense_split_k"] == "configuration"
    # Native syntax/shape errors remain deferred until the old call checkpoint.
    dormant = Sm70NativeConfig()
    dormant.resolve("marlin")
    assert dormant.hash_options() == {"marlin_dense_split_k": "invalid"}


def test_marlin_input_does_not_change_unrelated_native_family(monkeypatch):
    monkeypatch.setenv("SM70_MARLIN_DENSE_SPLIT_K", "invalid")
    native = Sm70NativeConfig()
    native.resolve("fp8")
    assert not any("marlin" in name for name in native.hash_options())
