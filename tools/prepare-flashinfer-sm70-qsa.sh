#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

# Separate pin from the older WMMA primitive probe. No wheel/package install.
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
sha=6c14bbd5ff34210404d5d4b5f6ff3b4b2527f59f
cccl=16bd510c9b712e82b0ab6cbb630d8e29ba1f7116
source_dir=${FLASHINFER_SM70_QSA_SOURCE:-"$root/.deps/flashinfer-6c14bbd5ff34"}
if [[ ! -e "$source_dir" ]]; then
  git clone --filter=blob:none --no-checkout \
    https://github.com/flashinfer-ai/flashinfer.git "$source_dir"
  git -C "$source_dir" checkout --detach "$sha"
fi
[[ $(git -C "$source_dir" rev-parse HEAD) == "$sha" ]] || {
  echo "FlashInfer source has a different revision; refusing to overwrite" >&2
  exit 1
}
[[ -z $(git -C "$source_dir" status --porcelain --untracked-files=no) ]] || {
  echo "FlashInfer source has local edits; refusing to overwrite" >&2
  exit 1
}
git -C "$source_dir" submodule update --init --depth 1 3rdparty/cccl
[[ $(git -C "$source_dir/3rdparty/cccl" rev-parse HEAD) == "$cccl" ]]
echo "Prepared FlashInfer $sha with CCCL $cccl at $source_dir"
