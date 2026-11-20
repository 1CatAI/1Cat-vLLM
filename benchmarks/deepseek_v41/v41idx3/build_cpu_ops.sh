#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Usage: build_cpu_ops.sh PYTHON OUT.so [extra nvcc flags, e.g. -DV41IDX3_MUTANT_LIMIT]
# Builds v41idx3_cpu_ops.cu (both scorers' own host bodies and the card's measured adder as CPU torch ops) for the
# host tests, on a machine with no card (nvcc is needed for cuda_fp16.h). Nothing is installed anywhere.
set -euo pipefail
PY=$1; OUT=$(realpath -m "$2"); shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
INC=$("$PY" -c "from torch.utils import cpp_extension as ce; print(' '.join('-I' + p for p in ce.include_paths(device_type='cpu')))")
LIB=$("$PY" -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
mkdir -p "$(dirname "$OUT")"
nvcc -shared -Xcompiler -fPIC -O2 -std=c++17 --expt-relaxed-constexpr -gencode arch=compute_70,code=sm_70 \
  -D_GLIBCXX_USE_CXX11_ABI=1 "$@" $INC -I "$HERE" -I "$HERE/../v41idx" "$HERE/v41idx3_cpu_ops.cu" -o "$OUT" \
  -L "$LIB" -Xlinker -rpath,"$LIB" -ltorch -ltorch_cpu -lc10
echo "built $OUT"
