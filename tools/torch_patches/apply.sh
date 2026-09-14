#!/usr/bin/env bash
# Apply the torch 2.10.0 backports in this directory to the torch of a given interpreter.
# Idempotent: a file already at the patched hash is skipped, the pristine 2.10.0 file is
# patched (original kept as .orig-2.10.0), anything else aborts loudly. Re-run after every
# reinstall of torch (pip install -e . does not touch torch, a venv rebuild does).
#   tools/torch_patches/apply.sh [python]     default: python from PATH
set -euo pipefail
PY=${1:-python}
HERE=$(cd "$(dirname "$0")" && pwd)
TORCH=$("$PY" -c 'import os, torch; print(os.path.dirname(torch.__file__))')
VER=$("$PY" -c 'import torch; print(torch.__version__)')
case "$VER" in 2.10.0*) ;; *) echo "torch $VER: these backports are for 2.10.0 only (2.11+ has them upstream)"; exit 2 ;; esac
sha() { sha256sum "$1" | cut -c1-64; }
# --- 0001: torch #173556, serialize the triton kernel side table into bundled AOT artifacts ---
F="$TORCH/_dynamo/aot_compile_types.py"
ORIG=93f2529e50a3fa31dd0a4dfcd51f1e907dbf74fe782e0d0819d78ef365fbe74e
PATCHED=d3acc67b813f260f8989156ae5fbdcaf1c04c8eee236b567c8681dd377675f35
case "$(sha "$F")" in
  $PATCHED) echo "0001 aot_compile_types.py: already applied" ;;
  $ORIG)    [ -e "$F.orig-2.10.0" ] || cp "$F" "$F.orig-2.10.0"; patch -p1 -d "$(dirname "$TORCH")" --forward --silent < "$HERE/0001-aot-compile-serialize-triton-kernel-side-table.patch"
            [ "$(sha "$F")" = "$PATCHED" ] || { echo "0001: hash after patch unexpected"; exit 1; }; echo "0001 aot_compile_types.py: applied" ;;
  *)        echo "0001: $F is neither pristine 2.10.0 nor patched; refusing"; exit 1 ;;
esac
