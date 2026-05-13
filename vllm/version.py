# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

try:
    from ._version import __version__, __version_tuple__
except Exception as e:
    import warnings

    warnings.warn(f"Failed to read commit hash:\n{e}", RuntimeWarning, stacklevel=2)

    __version__ = "dev"
    __version_tuple__ = (0, 0, __version__)


def _prev_minor_version_was(version_str):
    """Check whether a given version matches the previous minor version.

    Return True if version_str matches the previous minor version.

    For example - return True if the current version if 0.7.4 and the
    supplied version_str is '0.6'.

    Used for --show-hidden-metrics-for-version.
    """
    # Match anything if this is a dev tree
    if __version_tuple__[0:2] == (0, 0):
        return True

    major, minor = _prev_minor_version_pair()
    return version_str == f"{major}.{minor}"


def _prev_minor_version():
    """For the purpose of testing, return a previous minor version number."""
    major, minor = _prev_minor_version_pair()
    return f"{major}.{minor}"


def _prev_minor_version_pair():
    assert isinstance(__version_tuple__[0], int)
    assert isinstance(__version_tuple__[1], int)
    major = __version_tuple__[0]
    minor = __version_tuple__[1]
    if minor > 0:
        return major, minor - 1
    if major > 0:
        return major - 1, 0
    # In dev trees, this preserves the historical "0.-1" fallback.
    return 0, -1
