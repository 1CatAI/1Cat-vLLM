# SPDX-License-Identifier: Apache-2.0
"""Explicit Volta kernels. No engine dependency or automatic performance admission."""

from importlib import import_module

_EXPORTS = {
    "prepare": ("._jit", "prepare"),
    "build": ("._jit", "build"),
    "provenance": ("._jit", "provenance"),
    "rmsnorm": (".ops", "rmsnorm"),
    "gemma_rmsnorm": (".ops", "gemma_rmsnorm"),
    "fused_add_rmsnorm": (".ops", "fused_add_rmsnorm"),
    "gemma_fused_add_rmsnorm": (".ops", "gemma_fused_add_rmsnorm"),
    "silu_and_mul": (".ops", "silu_and_mul"),
    "gelu_and_mul": (".ops", "gelu_and_mul"),
    "gelu_tanh_and_mul": (".ops", "gelu_tanh_and_mul"),
    "apply_rope_pos_ids": (".ops", "apply_rope_pos_ids"),
    "apply_rope_pos_ids_inplace": (".ops", "apply_rope_pos_ids_inplace"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attr = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
