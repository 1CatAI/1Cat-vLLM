# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate registration metadata without importing getters or CUDA packages."""

import ast
import sys
from pathlib import Path

ENVS_FILE = Path("vllm/envs.py")
FIELDS = {
    "description",
    "category",
    "declared_default",
    "effective_default",
    "automatic_conditions",
    "acceleration_paths",
    "user_visible",
}
CATEGORIES = {"configuration", "tuning", "experimental", "debug", "deprecated"}


def static_unset_default(node: ast.expr) -> tuple[bool, object]:
    """Read simple literal defaults without executing any source expression."""
    if isinstance(node, ast.Lambda):
        return static_unset_default(node.body)
    if isinstance(node, ast.Call):
        function = ast.unparse(node.func)
        if function in ("os.getenv", "os.environ.get", "env_with_choices"):
            return (
                static_unset_default(node.args[1])
                if len(node.args) > 1
                else (True, None)
            )
        if function == "deprecated_env" and len(node.args) == 4:
            return static_unset_default(node.args[3])
        if function in ("bool", "int", "float", "str") and len(node.args) == 1:
            known, value = static_unset_default(node.args[0])
            if known:
                try:
                    return True, {"bool": bool, "int": int, "float": float, "str": str}[
                        function
                    ](value)
                except (ValueError, TypeError):
                    pass
        return False, None
    try:
        return True, ast.literal_eval(node)
    except (ValueError, TypeError):
        return False, None


def registrations(source: str) -> dict[str, ast.expr]:
    tree = ast.parse(source)
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "environment_variables"
            and isinstance(node.value, ast.Dict)
        ):
            names = [ast.literal_eval(key) for key in node.value.keys]
            if len(names) != len(set(names)):
                raise ValueError("Duplicate environment variable registration")
            return dict(zip(names, node.value.values, strict=True))
    raise ValueError("environment_variables dictionary not found")


def read_metadata(source: str) -> tuple[dict[str, dict], list[str]]:
    result = {}
    errors = []
    for name, getter in registrations(source).items():
        if not (
            isinstance(getter, ast.Call)
            and isinstance(getter.func, ast.Name)
            and getter.func.id == "env_var"
            and len(getter.args) == 1
        ):
            errors.append(f"{name}: register with env_var and complete metadata")
            continue
        fields = {kw.arg: kw.value for kw in getter.keywords}
        if set(fields) != FIELDS:
            errors.append(f"{name}: metadata fields must be {sorted(FIELDS)}")
            continue
        try:
            metadata = {key: ast.literal_eval(value) for key, value in fields.items()}
        except (ValueError, TypeError):
            errors.append(f"{name}: metadata must use literals; never evaluate getters")
            continue
        for field in ("description", "declared_default", "effective_default"):
            if not isinstance(metadata[field], str) or not metadata[field].strip():
                errors.append(f"{name}: {field} must explain its value")
        if not isinstance(metadata["user_visible"], bool):
            errors.append(f"{name}: user_visible must be a literal boolean")
        if (
            metadata["user_visible"]
            and metadata["acceleration_paths"]
            and "consumer locations" in metadata["description"].lower()
        ):
            errors.append(
                f"{name}: public SM70 descriptions must explain the operation, "
                "default rationale and reason to override; consumer placeholders "
                "belong only in the internal migration inventory"
            )
        if (
            not isinstance(metadata["category"], str)
            or metadata["category"] not in CATEGORIES
        ):
            errors.append(f"{name}: unknown category {metadata['category']!r}")
        for field in ("automatic_conditions", "acceleration_paths"):
            value = metadata[field]
            if not isinstance(value, tuple) or any(
                not isinstance(item, str) or not item.strip() for item in value
            ):
                errors.append(f"{name}: {field} must be a tuple of nonempty strings")
            elif len(value) != len(set(value)):
                errors.append(f"{name}: duplicate {field}")
        known, value = static_unset_default(getter.args[0])
        effective = metadata["effective_default"]
        prefix = repr(value)
        if (
            known
            and isinstance(effective, str)
            and not (
                effective == prefix
                or any(
                    effective.startswith(prefix + separator)
                    for separator in (" ", ";", ".")
                )
            )
        ):
            errors.append(
                f"{name}: effective default must describe getter default {value!r}"
            )
        if known and isinstance(metadata["declared_default"], str):
            try:
                declared = ast.literal_eval(metadata["declared_default"])
            except (ValueError, SyntaxError):
                pass
            else:
                if declared != value and not metadata["automatic_conditions"]:
                    errors.append(
                        f"{name}: declared/getter defaults differ without explanation"
                    )
        result[name] = metadata
    return result, errors


def main() -> int:
    _, errors = read_metadata(ENVS_FILE.read_text())
    for error in errors:
        print(f"{ENVS_FILE}: {error}")
    return bool(errors)


if __name__ == "__main__":
    sys.exit(main())
