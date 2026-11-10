# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Live legacy aliases; the public operation remains the sole mutable owner."""

import types


class OwnerAliases(types.ModuleType):
    def __getattr__(self, name):
        bindings = vars(self).get("LEGACY_ALIASES", {})
        if name in bindings:
            return getattr(self, bindings[name])
        observations = vars(self).get("LEGACY_OBSERVATIONS", {})
        if name in observations:
            owner, target = observations[name]
            return getattr(owner, target)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        bindings = vars(self).get("LEGACY_ALIASES", {})
        if name in bindings:
            name = bindings[name]
        observations = vars(self).get("LEGACY_OBSERVATIONS", {})
        if name in observations:
            owner, target = observations[name]
            setattr(owner, target, value)
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        bindings = vars(self).get("LEGACY_ALIASES", {})
        super().__delattr__(bindings.get(name, name))

    def __dir__(self):
        return sorted(
            set(super().__dir__()) | set(vars(self).get("LEGACY_ALIASES", {}))
        )


def install_owner_aliases(module):
    if vars(module).get("LEGACY_ALIASES") or vars(module).get("LEGACY_OBSERVATIONS"):
        module.__class__ = OwnerAliases
