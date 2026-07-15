#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_REPO="${FLASHINFER_SM70_REPO:-https://github.com/flashinfer-ai/flashinfer.git}"
UPSTREAM_REF="${FLASHINFER_SM70_REF:-v0.6.13}"
UPSTREAM_COMMIT="${FLASHINFER_SM70_COMMIT:-57ba7eeb7ea3003a2d6ad5d9a057c4f952709bac}"
SOURCE_DIR="${FLASHINFER_SM70_SOURCE_DIR:-${ROOT_DIR}/.deps/flashinfer-sm70-v0.6.13}"
PATCH_DIR="${ROOT_DIR}/flashinfer-sm70/patches"
OVERLAY_DIR="${ROOT_DIR}/flashinfer-sm70/include"

if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
    git clone --recursive --branch "${UPSTREAM_REF}" "${UPSTREAM_REPO}" "${SOURCE_DIR}"
fi

actual_commit="$(git -C "${SOURCE_DIR}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${UPSTREAM_COMMIT}" ]]; then
    echo "Unexpected FlashInfer revision: ${actual_commit}" >&2
    echo "Expected: ${UPSTREAM_COMMIT}" >&2
    exit 1
fi

shopt -s nullglob
for patch in "${PATCH_DIR}"/*.patch; do
    if git -C "${SOURCE_DIR}" apply --reverse --check "${patch}" >/dev/null 2>&1; then
        continue
    fi
    git -C "${SOURCE_DIR}" apply --check "${patch}"
    git -C "${SOURCE_DIR}" apply "${patch}"
done

if [[ -d "${OVERLAY_DIR}" ]]; then
    cp -a "${OVERLAY_DIR}/." "${SOURCE_DIR}/include/"
fi

printf '%s\n' "${SOURCE_DIR}"
