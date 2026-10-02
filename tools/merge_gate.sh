#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

usage() {
    echo 'Usage: tools/merge_gate.sh [--base REV] [--python PATH] TEST_PATH...' >&2
}

root=$(git rev-parse --show-toplevel)
cd "$root"
base=onecat/main
python="$root/.venv/bin/python"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --base|--python)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            if [[ $1 == --base ]]; then base=$2; else python=$2; fi
            shift 2
            ;;
        --help|-h) usage; exit 0 ;;
        --*) usage; exit 2 ;;
        *) break ;;
    esac
done
[[ $# -gt 0 ]] || { usage; exit 2; }
[[ -x "$python" ]] || { echo "Python is not executable: $python" >&2; exit 2; }
for test_path in "$@"; do
    [[ $test_path == tests/* && -e ${test_path%%::*} ]] || {
        echo "Expected a repository test path: $test_path" >&2
        exit 2
    }
done

# These checks never reserve a GPU or load model weights. Native/GPU validation
# remains required for changes to kernels or their execution paths.
export CUDA_VISIBLE_DEVICES=''
export VLLM_TARGET_DEVICE=cpu
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1

git diff --check "$base" HEAD
"$python" tools/run_cpu_tests.py --noconftest -q \
    tests/config/test_sm70_release_profile.py \
    tests/config/test_sm70_acceleration.py \
    tests/config/test_sm70_dflash2_graph_cache.py \
    tests/config/test_flash_next_release_launcher.py \
    tests/models/qwen4_exp/test_ple_auto_hybrid_budget.py \
    tests/models/qwen4_exp/test_ple_cache_budget.py
"$python" tools/run_cpu_tests.py --noconftest -q "$@"
mapfile -t changed_files < <(git diff --name-only --diff-filter=ACMR "$base" HEAD)
if [[ ${#changed_files[@]} -gt 0 ]]; then
    "$python" -m pre_commit run --files "${changed_files[@]}"
fi
