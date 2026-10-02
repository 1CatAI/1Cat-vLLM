# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reuse matching checkpoint storage without changing initialization or RNG."""

import weakref
from functools import wraps
from threading import RLock

from torch import Tensor, nn, strided

from vllm.logger import init_logger

logger = init_logger(__name__)
_LOCK = RLock()


def _can_assign_checkpoint(module, state_dict):
    targets = module.state_dict(keep_vars=True)
    if targets.keys() != state_dict.keys():
        return False
    storages, source_storages = set(), set()
    for name, target in targets.items():
        value = state_dict[name]
        if (
            not isinstance(target, Tensor)
            or not isinstance(value, Tensor)
            or type(target) not in (Tensor, nn.Parameter)
            or target.__dict__
            or target.layout != strided
            or value.layout != strided
            or target.device.type != "cpu"
            or value.device.type != "cpu"
            or target.dtype != value.dtype
            or target.shape != value.shape
            or target.stride() != value.stride()
            or target.storage_offset() != value.storage_offset()
        ):
            return False
        storage = target.untyped_storage()
        if storage.nbytes():
            identity = storage.data_ptr()
            if identity in storages:
                return False  # Preserve tied parameters and storage aliases.
            storages.add(identity)
            source_identity = value.untyped_storage().data_ptr()
            if source_identity in source_storages:
                return False
            source_storages.add(source_identity)
    return True


def load_with_checkpoint_storage(factory):
    """Reuse complete matching CPU checkpoints in the isolated inference worker.

    Keep ordinary parameter/buffer initialization and RNG consumption unchanged.
    Unsupported layouts, dtype conversions, aliases and partial checkpoints use
    the normal copy loader. Restore the process-local load hook on all exits.
    """
    assigned = {}
    with _LOCK:
        load_state_dict = nn.Module.load_state_dict

        @wraps(load_state_dict)
        def load(module, state_dict, *args, **kwargs):
            if len(args) < 2 and "assign" not in kwargs:
                if _can_assign_checkpoint(module, state_dict):
                    kwargs["assign"] = True
                else:
                    logger.info(
                        "VAE checkpoint storage reuse unavailable: complete matching "
                        "CPU dtype/layout and independent tensor storage are required; "
                        "using normal copy loading"
                    )
            result = load_state_dict(module, state_dict, *args, **kwargs)
            if kwargs.get("assign"):
                missing = set(result.missing_keys)
                for name, parameter in module.named_parameters():
                    if name in state_dict and name not in missing:
                        assigned[id(parameter)] = weakref.ref(parameter)
            return result

        try:
            nn.Module.load_state_dict = load
            model = factory()
        finally:
            nn.Module.load_state_dict = load_state_dict

        parameters = list(model.parameters())
        if parameters and all(
            id(p) in assigned and assigned[id(p)]() is p for p in parameters
        ):
            model._h3_assigned_checkpoint_storage = {
                name: (p.data_ptr(), p.dtype, p.device, tuple(p.shape), p.stride())
                for name, p in model.named_parameters()
            }
        return model


def uses_assigned_checkpoint_storage(model):
    expected = getattr(model, "_h3_assigned_checkpoint_storage", None)
    if not expected:
        return False
    actual = {
        name: (p.data_ptr(), p.dtype, p.device, tuple(p.shape), p.stride())
        for name, p in model.named_parameters()
    }
    return expected == actual
