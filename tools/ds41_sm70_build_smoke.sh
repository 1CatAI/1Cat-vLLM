#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Post-build smoke for tools/bootstrap_ds41_sm70.sh on ONE V100:
#   1. import vllm/torch and report versions and arch list;
#   2. confirm every compiled extension carries sm_70 device code only;
#   3. run the SM70 unit/kernel tests that cover the DeepSeek-V4 path.
# Refuses to start unless the named GPU has no compute processes. Loads no model.
# The exit code covers steps 1-2; per-file test results are in OUT_DIR/summary.txt
# (known stale upstream tests are listed in the v100-research BUILD.md).
#
#   tools/ds41_sm70_build_smoke.sh GPU-<uuid> OUT_DIR
# Needs pytest in the venv: pytest==8.3.5 pytest-asyncio==0.24.0
# pytest-forked==1.6.0 pytest-timeout==2.3.1 tblib==3.1.0 (requirements/test pins).
set -uo pipefail

(($# == 2)) || { echo "usage: $0 GPU-<uuid> OUT_DIR" >&2; exit 2; }
gpu=$1
out=$2
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
py="${repo}/.venv/bin/python"
[[ "${gpu}" == GPU-* ]] || { echo "pin the GPU by UUID (GPU-...)" >&2; exit 2; }
[[ -x "${py}" ]] || { echo "missing venv interpreter ${py}" >&2; exit 1; }
busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "${gpu}")
((busy == 0)) || { echo "refusing: ${gpu} has ${busy} compute process(es)" >&2; exit 1; }
mkdir -p "${out}"
export CUDA_VISIBLE_DEVICES="${gpu}" PYTHONNOUSERSITE=1 VLLM_LOGGING_LEVEL=WARNING
cd "${repo}"

"${py}" -c "import vllm, torch; print(vllm.__version__, torch.__version__, torch.version.cuda, torch.cuda.get_arch_list(), torch.cuda.nccl.version())" \
    | tee "${out}/import.txt" || { echo "import smoke failed" >&2; exit 1; }

cuobjdump=${CUOBJDUMP:-$(command -v cuobjdump || echo /usr/local/cuda/bin/cuobjdump)}
[[ -x "${cuobjdump}" ]] || { echo "cuobjdump not found; set CUOBJDUMP" >&2; exit 1; }
bad=0
while IFS= read -r so; do
    archs=$("${cuobjdump}" --list-elf --list-ptx "${so}" 2>/dev/null | grep -oE 'sm_[0-9]+[a-z]?' | sort -u | tr '\n' ' ')
    printf '%-75s %s\n' "${so}" "${archs:-host-only}"
done < <(find vllm flash-attention-v100/flash_attn_v100 flash_qla -name '*.so' | sort) > "${out}/so-arch.txt"
cat "${out}/so-arch.txt"
[[ -s "${out}/so-arch.txt" ]] || { echo "no compiled extensions found" >&2; bad=1; }
if grep -qvE ' (sm_70 |host-only)$' "${out}/so-arch.txt"; then
    echo "non-SM70 device code found" >&2
    bad=1
fi

tests=(
    tests/kernels/test_deepseek_v4_sm70_indexer.py
    tests/kernels/test_deepseek_v4_sm70_qnorm_rope_kv_insert.py
    tests/kernels/test_deepseek_v4_sm70_fp8_software.py
    tests/kernels/test_sm70_deepseek_v4_fp16_gemv.py
    tests/kernels/test_mhc_sm70_fp16.py
    tests/kernels/test_sm70_hc_local_batch.py
    tests/kernels/attention/test_dsv4_sm70_sparse_bmm.py
    tests/kernels/attention/test_dsv4_sparse_prefill_bmm.py
    tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py
    tests/kernels/quantization/test_sm70_marlin_splitk.py
    tests/kernels/moe/test_skinny_sm70_moe.py
    tests/quantization/test_sm70_mxfp4_moe.py
    tests/quantization/test_sm70_fp8_kernel_selection.py
    tests/quantization/test_sm70_moe_backend_override.py
    tests/models/test_deepseek_v4_sm70_routes.py
    tests/models/test_deepseek_v4_sm70_sparse_policy.py
    tests/models/test_deepseek_v4_fp8_capability.py
    tests/models/test_deepseek_v4_pipeline.py
    tests/models/test_deepseek_v4_mega_moe.py
    tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py
    tests/model_executor/test_sm70_fp8_qpn8_pp2_tp4.py
    tests/v1/core/test_engine_core_structured_drafts.py
)
for t in "${tests[@]}"; do
    name=$(basename "${t}" .py)
    timeout 1200 "${py}" -m pytest -q -rfEs -p no:cacheprovider --timeout=300 \
        --junitxml="${out}/${name}.xml" "${t}" > "${out}/${name}.log" 2>&1
    printf 'rc=%d %-62s %s\n' $? "${t}" "$(tail -n 1 "${out}/${name}.log")"
done | tee "${out}/summary.txt"
exit "${bad}"
