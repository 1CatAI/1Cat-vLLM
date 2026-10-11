#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${ROOT_DIR}/flashinfer-sm70"
test -f "${SOURCE_DIR}/UPSTREAM.lock.json"
test -f "${SOURCE_DIR}/pyproject.toml"

printf '%s\n' "${SOURCE_DIR}"
