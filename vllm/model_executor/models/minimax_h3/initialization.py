# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Avoid random parameter fills that a strict checkpoint load replaces."""

import weakref
from functools import wraps
from threading import RLock

from torch import nn

from vllm.logger import init_logger

logger = init_logger(__name__)
_LOCK = RLock()
_RANDOM_INITIALIZERS = (
    "uniform_",
    "normal_",
    "trunc_normal_",
    "kaiming_uniform_",
    "kaiming_normal_",
    "xavier_uniform_",
    "xavier_normal_",
    "orthogonal_",
)


def load_without_random_parameter_init(factory):
    """For the isolated H3 loading worker, skip only replaced Parameters.

    Buffers and ordinary tensors still receive their normal initialization.
    Track actual successful state-dict loads, rather than trusting the factory
    to load every parameter. Unsupported/custom partial loaders retry normally.
    The process-local patch is restored before returning or propagating errors.
    """
    skipped, loaded = {}, {}
    with _LOCK:
        originals = {name: getattr(nn.init, name) for name in _RANDOM_INITIALIZERS}
        load_state_dict = nn.Module.load_state_dict

        def wrap_initializer(original):
            @wraps(original)
            def initialize(tensor, *args, **kwargs):
                if isinstance(tensor, nn.Parameter):
                    skipped[id(tensor)] = weakref.ref(tensor)
                    return tensor
                return original(tensor, *args, **kwargs)

            return initialize

        @wraps(load_state_dict)
        def load(module, state_dict, *args, **kwargs):
            result = load_state_dict(module, state_dict, *args, **kwargs)
            missing = set(result.missing_keys)
            for name, parameter in module.named_parameters():
                if name in state_dict and name not in missing:
                    loaded[id(parameter)] = weakref.ref(parameter)
            return result

        try:
            for name, original in originals.items():
                setattr(nn.init, name, wrap_initializer(original))
            nn.Module.load_state_dict = load
            model = factory()
        finally:
            nn.Module.load_state_dict = load_state_dict
            for name, original in originals.items():
                setattr(nn.init, name, original)

        uncovered = [
            name
            for name, parameter in model.named_parameters()
            if id(parameter) in skipped
            and skipped[id(parameter)]() is parameter
            and (
                id(parameter) not in loaded or loaded[id(parameter)]() is not parameter
            )
        ]
        if not uncovered:
            return model
        logger.warning(
            "VAE loader did not replace all skipped parameters; "
            "retrying with normal initialization (%s)",
            uncovered[:5],
        )
        del model
        return factory()
