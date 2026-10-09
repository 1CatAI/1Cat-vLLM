#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Reproducible SM70 (Tesla V100) source build of this 1Cat-vLLM tree in an
# isolated virtual environment, for the DeepSeek-V4.1-Flash port.
#
# Modelled on the GLM-5.3 SM70 bootstrap (codex/glm53-sm70-source-audit,
# tools/bootstrap_glm53_sm70.sh). Differences: parallelism is a parameter
# (default 10 jobs), the interpreter is pinned explicitly, 1Cat's torch 2.10.0
# backports (tools/torch_patches) are applied, lint/pre-commit tooling is not
# installed (it is not part of the build), and the CMake FetchContent
# dependencies can optionally be served from SHA-verified local git mirrors.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: tools/bootstrap_ds41_sm70.sh [--execute] [--venv PATH] [--cuda-home PATH]
    [--python PATH] [--jobs N] [--git-mirror-dir DIR]
    [--seed-mirrors-from DOTDEPS]

The default is a command preview; nothing is changed. --execute creates (or
reuses) the virtual environment, installs the pinned dependencies and compiles
every CUDA extension for SM70 only. Re-running is safe: an existing venv is
reused and uv/pip skip satisfied requirements; the editable build is redone.

  --venv PATH        virtual environment (default: <repo>/.venv)
  --cuda-home PATH   CUDA 12.8.x toolkit (default: $CUDA_HOME or
                     /mnt/nvme2/toolchains/cuda-12.8.93); other releases are refused
  --python PATH      interpreter for the venv (default: /usr/bin/python3.12)
  --jobs N           MAX_JOBS / CMAKE_BUILD_PARALLEL_LEVEL (default: 10)
  --git-mirror-dir DIR
                     serve the CMake FetchContent git dependencies (CUTLASS,
                     flash-attention-v100 + its CUTLASS submodule, Triton) from
                     bare mirrors in DIR via git url.insteadOf, scoped to this
                     process. Every pinned revision is verified first.
  --seed-mirrors-from DOTDEPS
                     with --git-mirror-dir: create missing mirrors from the
                     FetchContent checkouts of an earlier 1Cat build (its .deps
                     directory). Read-only on DOTDEPS (git clone --no-local).
EOF
}

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
venv_dir="${repo_root}/.venv"
cuda_home="${CUDA_HOME:-/mnt/nvme2/toolchains/cuda-12.8.93}"
python_req=/usr/bin/python3.12
jobs=10
git_mirror_dir=""
seed_from=""
execute=0

while (($#)); do
    case "$1" in
        --execute) execute=1 ;;
        --venv|--cuda-home|--python|--jobs|--git-mirror-dir|--seed-mirrors-from)
            (($# >= 2)) || { echo "$1 needs a value" >&2; exit 2; }
            case "$1" in
                --venv) venv_dir=$2 ;;
                --cuda-home) cuda_home=$2 ;;
                --python) python_req=$2 ;;
                --jobs) jobs=$2 ;;
                --git-mirror-dir) git_mirror_dir=$2 ;;
                --seed-mirrors-from) seed_from=$2 ;;
            esac
            shift
            ;;
        --help|-h) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

[[ "${jobs}" =~ ^[1-9][0-9]*$ ]] || { echo "--jobs must be a positive integer" >&2; exit 2; }
if [[ -n "${seed_from}" && -z "${git_mirror_dir}" ]]; then
    echo "--seed-mirrors-from requires --git-mirror-dir" >&2
    exit 2
fi

python_bin="${venv_dir}/bin/python"
if [[ -x "${cuda_home}/bin/nvcc" ]]; then
    nvcc_bin="${cuda_home}/bin/nvcc"
    cuda_root="${cuda_home}"
elif [[ -x "${cuda_home}/targets/x86_64-linux/bin/nvcc" ]]; then
    nvcc_bin="${cuda_home}/targets/x86_64-linux/bin/nvcc"
    cuda_root="${cuda_home}/targets/x86_64-linux"
else
    nvcc_bin="${cuda_home}/bin/nvcc"
    cuda_root="${cuda_home}"
fi
if [[ -d "${cuda_home}/targets/x86_64-linux" ]]; then
    cuda_target_root="${cuda_home}/targets/x86_64-linux"
else
    cuda_target_root="${cuda_root}"
fi

# The pinned toolkit is a minimal nvcc + cudart install; cuBLAS, cuSPARSE,
# cuSOLVER, cuRAND, cuFFT, NVRTC and NVTX come from the NVIDIA wheels that
# torch 2.10.0+cu128 installs into the venv, so CMake is pointed at them.
nvidia_package_root="${venv_dir}/lib/python3.12/site-packages/nvidia"
cuda_splayed_lib="${venv_dir}/cuda-splayed/lib"
cuda_cudart_library="${cuda_target_root}/lib/libcudart.so"
[[ -f "${cuda_cudart_library}" ]] || cuda_cudart_library="${cuda_root}/lib64/libcudart.so"

cuda_cmake_args="-DCUDA_TOOLKIT_ROOT_DIR=${cuda_target_root}"
cuda_cmake_args+=" -DCUDA_TOOLKIT_TARGET_DIR=${cuda_target_root}"
cuda_cmake_args+=" -DCUDA_NVCC_EXECUTABLE=${nvcc_bin}"
cuda_cmake_args+=" -DCMAKE_LIBRARY_PATH=${cuda_splayed_lib}"
cuda_cmake_args+=" -DCUDAToolkit_INCLUDE_DIR=${nvidia_package_root}/cublas/include"
cuda_cmake_args+=" -DCUDA_CUDART=${cuda_cudart_library}"
cuda_cmake_args+=" -DCUDA_cublas_LIBRARY=${nvidia_package_root}/cublas/lib/libcublas.so.12"
cuda_cmake_args+=" -DCUDA_cublasLt_LIBRARY=${nvidia_package_root}/cublas/lib/libcublasLt.so.12"
cuda_cmake_args+=" -DCUDA_cusparse_LIBRARY=${nvidia_package_root}/cusparse/lib/libcusparse.so.12"
cuda_cmake_args+=" -DCUDA_cusolver_LIBRARY=${nvidia_package_root}/cusolver/lib/libcusolver.so.11"
cuda_cmake_args+=" -DCUDA_nvrtc_LIBRARY=${nvidia_package_root}/cuda_nvrtc/lib/libnvrtc.so.12"
cuda_cmake_args+=" -DCUDA_curand_LIBRARY=${nvidia_package_root}/curand/lib/libcurand.so.10"
cuda_cmake_args+=" -DCUDA_cufft_LIBRARY=${nvidia_package_root}/cufft/lib/libcufft.so.11"
cuda_cmake_args+=" -DCUDA_nvToolsExt_LIBRARY=${nvidia_package_root}/nvtx/lib/libnvToolsExt.so.1"

cuda_include_paths=(
    "${cuda_root}/include"
    "${cuda_home}/targets/x86_64-linux/include"
    "${nvidia_package_root}/cublas/include"
    "${nvidia_package_root}/cuda_cccl/include"
    "${nvidia_package_root}/cuda_runtime/include"
    "${nvidia_package_root}/curand/include"
    "${nvidia_package_root}/cusolver/include"
    "${nvidia_package_root}/cusparse/include"
)
cuda_library_paths=(
    "${cuda_root}/lib"
    "${cuda_root}/lib64"
    "${cuda_home}/targets/x86_64-linux/lib"
    "${nvidia_package_root}/cublas/lib"
    "${nvidia_package_root}/cuda_runtime/lib"
    "${nvidia_package_root}/cusolver/lib"
    "${nvidia_package_root}/cusparse/lib"
)
cuda_cpath=$(IFS=:; echo "${cuda_include_paths[*]}")
cuda_ldpath=$(IFS=:; echo "${cuda_library_paths[*]}")

# CMake FetchContent dependencies of this tree: mirror name | URL exactly as the
# build requests it | path of an earlier checkout inside a .deps directory |
# revision the build pins | file that must still name that pin.
fetch_pins=(
    "nvidia-cutlass|https://github.com/nvidia/cutlass.git|cutlass-src|v4.4.2|CMakeLists.txt"
    "flash-attention-v100|https://github.com/zhinianqin/flash-attention-v100.git|vllm-flash-attn-src|c2eda5e6115b98c3ba4bfd181570668742eece22|cmake/external_projects/vllm_flash_attn.cmake"
    "NVIDIA-cutlass|https://github.com/NVIDIA/cutlass.git|vllm-flash-attn-src/csrc/cutlass|62750a2b75c802660e4894434dc55e839f322277|-"
    "triton|https://github.com/triton-lang/triton.git|triton_kernels-src|v3.5.1|cmake/external_projects/triton_kernels.cmake"
)

prepare_git_mirrors() {
    [[ -n "${git_mirror_dir}" ]] || return 0
    mkdir -p "${git_mirror_dir}"
    git_mirror_dir=$(cd -- "${git_mirror_dir}" && pwd -P)
    local idx=0 entry name url src rev pinfile mirror
    for entry in "${fetch_pins[@]}"; do
        IFS='|' read -r name url src rev pinfile <<<"${entry}"
        if [[ "${pinfile}" != "-" ]] && ! grep -qF "${rev}" "${repo_root}/${pinfile}"; then
            echo "pin drift: ${pinfile} no longer names ${rev}; update fetch_pins" >&2
            exit 1
        fi
        mirror="${git_mirror_dir}/${name}.git"
        if [[ ! -d "${mirror}" ]]; then
            [[ -n "${seed_from}" ]] || {
                echo "missing mirror ${mirror}; pass --seed-mirrors-from or drop --git-mirror-dir" >&2
                exit 1
            }
            [[ -d "${seed_from}/${src}" ]] || { echo "seed checkout missing: ${seed_from}/${src}" >&2; exit 1; }
            git clone --bare --no-local --quiet "${seed_from}/${src}" "${mirror}"
            # A detached FetchContent checkout is not a ref; pin it explicitly.
            git -C "${mirror}" fetch --quiet --no-tags "${seed_from}/${src}" \
                "+HEAD:refs/pins/seed-head"
        fi
        git -C "${mirror}" rev-parse --verify --quiet "${rev}^{commit}" >/dev/null || {
            echo "mirror ${mirror} lacks pinned revision ${rev}" >&2
            exit 1
        }
        echo "mirror ok: ${url} -> ${mirror} @ ${rev} = $(git -C "${mirror}" rev-parse "${rev}^{commit}")"
        export "GIT_CONFIG_KEY_${idx}=url.file://${mirror}.insteadOf"
        export "GIT_CONFIG_VALUE_${idx}=${url}"
        idx=$((idx + 1))
    done
    # git submodule clones run with GIT_PROTOCOL_FROM_USER=0; allow file://.
    export "GIT_CONFIG_KEY_${idx}=protocol.file.allow"
    export "GIT_CONFIG_VALUE_${idx}=always"
    export GIT_CONFIG_COUNT=$((idx + 1))
}

build_environment() {
    export CUDA_HOME="${cuda_root}"
    export CUDA_PATH="${cuda_root}"
    export NVCC="${nvcc_bin}"
    export CUDACXX="${nvcc_bin}"
    export CMAKE_CUDA_COMPILER="${nvcc_bin}"
    export CMAKE_ARGS="${cuda_cmake_args}"
    export PATH="${venv_dir}/bin:${cuda_root}/bin:${cuda_home}/bin:${PATH}"
    export CPATH="${cuda_cpath}${CPATH:+:${CPATH}}"
    export LIBRARY_PATH="${cuda_ldpath}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
    export LD_LIBRARY_PATH="${cuda_ldpath}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export TORCH_EXTENSIONS_DIR="${venv_dir}/torch_extensions"
    export TORCH_CUDA_ARCH_LIST=7.0
    export CMAKE_CUDA_ARCHITECTURES=70
    export FLASH_ATTN_V100_CUDA_ARCH_LIST=7.0
    export MAX_JOBS="${jobs}"
    export CMAKE_BUILD_PARALLEL_LEVEL="${jobs}"
    export NVCC_THREADS=1
    export PYTHONNOUSERSITE=1
    # vllm/envs.py treats only 1/true as enabled. A wheel-location variable
    # independently enables precompiled mode, so clear it explicitly.
    export VLLM_USE_PRECOMPILED=0
    unset VLLM_PRECOMPILED_WHEEL_LOCATION
}

stage_cuda_library_links() {
    mkdir -p "${cuda_splayed_lib}"
    local name source
    while IFS=' ' read -r name source; do
        [[ -n "${name}" && -f "${source}" ]] || {
            echo "required CUDA runtime library is missing: ${source}" >&2
            exit 1
        }
        ln -sfn "${source}" "${cuda_splayed_lib}/${name}"
    done <<EOF
libcublas.so ${nvidia_package_root}/cublas/lib/libcublas.so.12
libcublasLt.so ${nvidia_package_root}/cublas/lib/libcublasLt.so.12
libcusparse.so ${nvidia_package_root}/cusparse/lib/libcusparse.so.12
libcusolver.so ${nvidia_package_root}/cusolver/lib/libcusolver.so.11
libnvrtc.so ${nvidia_package_root}/cuda_nvrtc/lib/libnvrtc.so.12
libcurand.so ${nvidia_package_root}/curand/lib/libcurand.so.10
libcufft.so ${nvidia_package_root}/cufft/lib/libcufft.so.11
libnvToolsExt.so ${nvidia_package_root}/nvtx/lib/libnvToolsExt.so.1
EOF
}

if (( ! execute )); then
    cat <<EOF
Preview only; no files or packages will be changed.

cd ${repo_root}
uv venv --python ${python_req} ${venv_dir}
uv pip install --python ${python_bin} --torch-backend=cu128 -r requirements/build/cuda.txt
uv pip install --python ${python_bin} --torch-backend=cu128 -r requirements/cuda.txt
tools/torch_patches/apply.sh ${python_bin}
# symlink cuBLAS/cuSPARSE/... from the venv's NVIDIA wheels into ${cuda_splayed_lib}
${git_mirror_dir:+# FetchContent git URLs rewritten to verified mirrors under ${git_mirror_dir}
}CUDA_HOME=${cuda_root} CUDA_PATH=${cuda_root} NVCC=${nvcc_bin} CUDACXX=${nvcc_bin} \\
CMAKE_CUDA_COMPILER=${nvcc_bin} CMAKE_ARGS="${cuda_cmake_args}" \\
TORCH_CUDA_ARCH_LIST=7.0 CMAKE_CUDA_ARCHITECTURES=70 FLASH_ATTN_V100_CUDA_ARCH_LIST=7.0 \\
MAX_JOBS=${jobs} CMAKE_BUILD_PARALLEL_LEVEL=${jobs} NVCC_THREADS=1 PYTHONNOUSERSITE=1 \\
VLLM_USE_PRECOMPILED=0 VLLM_PRECOMPILED_WHEEL_LOCATION= \\
uv pip install --python ${python_bin} --no-build-isolation --torch-backend=cu128 -e ${repo_root}
EOF
    exit 0
fi

command -v uv >/dev/null || { echo "uv is required; install it without using system pip" >&2; exit 1; }
[[ -x "${nvcc_bin}" ]] || { echo "nvcc not found at ${nvcc_bin}" >&2; exit 1; }
cuda_version=$("${nvcc_bin}" --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | tail -n 1)
if [[ "${cuda_version}" != 12.8* ]]; then
    echo "refusing: ${nvcc_bin} reports CUDA ${cuda_version:-unknown}; the SM70 profile requires 12.8.x" >&2
    exit 1
fi
py_version=$("${python_req}" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
[[ "${py_version}" == 3.12 ]] || { echo "refusing: ${python_req} is Python ${py_version}; need 3.12" >&2; exit 1; }

echo "== $(date -u +%FT%TZ) bootstrap start: repo ${repo_root} @ $(git -C "${repo_root}" rev-parse HEAD)"
echo "== nvcc ${cuda_version} (${nvcc_bin}); $(gcc --version | head -n 1); python ${python_req}; jobs ${jobs}"

prepare_git_mirrors

if [[ ! -x "${python_bin}" ]]; then
    uv venv --python "${python_req}" "${venv_dir}"
fi

build_environment
mkdir -p "${TORCH_EXTENSIONS_DIR}"
cd "${repo_root}"
echo "== $(date -u +%FT%TZ) installing build requirements"
uv pip install --python "${python_bin}" --torch-backend=cu128 -r requirements/build/cuda.txt
echo "== $(date -u +%FT%TZ) installing runtime requirements"
uv pip install --python "${python_bin}" --torch-backend=cu128 -r requirements/cuda.txt
tools/torch_patches/apply.sh "${python_bin}"
stage_cuda_library_links

for header in \
    "${nvidia_package_root}/cublas/include/cublas_v2.h" \
    "${nvidia_package_root}/cuda_runtime/include/cuda_runtime.h" \
    "${nvidia_package_root}/curand/include/curand_kernel.h" \
    "${nvidia_package_root}/cusparse/include/cusparse.h"; do
    [[ -f "${header}" ]] || { echo "required CUDA development header is missing: ${header}" >&2; exit 1; }
done

echo "== $(date -u +%FT%TZ) compiling 1Cat for SM70 (editable)"
build_start=$(date +%s)
uv pip install --python "${python_bin}" --no-build-isolation --torch-backend=cu128 -e "${repo_root}"
# The editable install may re-resolve torch from the cu128 wheel URL; the
# backport is idempotent and must be present on the torch that is installed.
tools/torch_patches/apply.sh "${python_bin}"
echo "== $(date -u +%FT%TZ) editable build finished in $(( $(date +%s) - build_start )) s"

uv pip freeze --python "${python_bin}" > "${venv_dir}/ds41-build-freeze.txt"
echo "SM70 source build completed in ${venv_dir} (freeze: ${venv_dir}/ds41-build-freeze.txt)"
