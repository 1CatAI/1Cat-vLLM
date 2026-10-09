# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native resources bound to an engine and borrowed at host execution boundaries."""

import hashlib
from contextlib import contextmanager
from contextvars import ContextVar

import torch

from vllm.runtime_resources import current_runtime_resources

_active_owner: ContextVar["NativeRuntimeOwner | None"] = ContextVar(
    "sm70_native_owner", default=None
)

NATIVE_OWNERS = (
    "kernel_config.layer_execution.native",
    "kernel_config.sm70_mxfp4",
    *(
        f"kernel_config.sm70_{family}.native"
        for family in ("awq", "fp8", "nvfp4", "gguf")
    ),
    *(
        f"kernel_config.sm70_moe.{family}.native"
        for family in ("awq", "fp8", "nvfp4", "mxfp4")
    ),
)


class NativeRuntimeOwner:
    def __init__(self):
        for namespace in (torch.ops._C, torch.ops._moe_C):
            query = getattr(namespace, "sm70_native_runtime_abi", None)
            if query is None or query() != 1:
                raise RuntimeError(
                    "Engine-owned SM70 native resources require runtime ABI 1; "
                    "rebuild both normal _C and _moe_C extensions."
                )
        self.handles = (
            torch.classes._C.Sm70NativeRuntime(),
            torch.classes._moe_C.Sm70NativeRuntime(),
        )
        self.closed = False

    def bind(self, values, token):
        from vllm.config import get_current_vllm_config_or_none
        from vllm.config.sm70_native import NATIVE_FIELDS

        cfg = get_current_vllm_config_or_none()
        matches: list[str] = []
        for path in NATIVE_OWNERS:
            native = cfg
            for part in path.split("."):
                native = getattr(native, part, None)
            captured = getattr(native, "values", None)
            if captured is values:
                matches.insert(0, path)
            elif captured == values:
                matches.append(path)
        if matches:
            slot = "sm70:slot:" + matches[0]
        else:
            # Explicit standalone preparation under an engine still receives an
            # isolated owner. Derived slots include computation, never diagnostics
            # or addresses. Registering a conflicting policy fails immediately.
            calculation = tuple(
                v for v, field in zip(values, NATIVE_FIELDS) if not field[3]
            )
            slot = (
                "sm70:slot:derived:"
                + hashlib.sha256(repr(calculation).encode()).hexdigest()
            )
        for handle in self.handles:
            handle.bind(slot, token)
        return slot

    @contextmanager
    def activate(self):
        if self.closed:
            raise RuntimeError("SM70 native runtime has been closed")
        if _active_owner.get() is self:
            yield
            return
        token = _active_owner.set(self)
        entered = []
        try:
            for handle in self.handles:
                handle.enter()
                entered.append(handle)
            yield
        finally:
            for handle in reversed(entered):
                handle.exit()
            _active_owner.reset(token)

    def close(self):
        if not self.closed:
            for handle in self.handles:
                handle.close()
            self.closed = True


class BoundNativeCall:
    """AOT records the stable slot; eager initialization borrows the same owner."""

    def __init__(self, operation, owner):
        self.operation = operation
        self.owner = owner

    def __call__(self, *args, **kwargs):
        if torch.compiler.is_compiling():
            return self.operation(*args, **kwargs)
        if _active_owner.get() is self.owner:
            return self.operation(*args, **kwargs)
        with self.owner.activate():
            return self.operation(*args, **kwargs)


def bind_native_runtime():
    resources = current_runtime_resources()
    if resources is None:
        return None
    owner = resources.get("sm70_native_runtime")
    if owner is None:
        owner = resources["sm70_native_runtime"] = NativeRuntimeOwner()
        resources.setdefault("execution_context_owners", []).append(owner)
    return owner
