#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
mkdir -p "${RUNNER_TEMP}"

if [[ -n "${HTTP_PROXY_URL:-}" ]]; then
  export HTTP_PROXY="${HTTP_PROXY_URL}"
  export HTTPS_PROXY="${HTTP_PROXY_URL}"
  export http_proxy="${HTTP_PROXY_URL}"
  export https_proxy="${HTTP_PROXY_URL}"
fi

cpu_count="$(nproc)"
build_jobs="${BUILD_JOBS:-8}"
if [[ ! "${cpu_count}" =~ ^[1-9][0-9]*$ || ! "${build_jobs}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid CPU/build parallelism: CPUs=${cpu_count}, jobs=${build_jobs}" >&2
  exit 1
fi
if (( build_jobs > cpu_count )); then
  build_jobs="${cpu_count}"
fi
export MAX_JOBS="${build_jobs}"
export CMAKE_BUILD_PARALLEL_LEVEL="${build_jobs}"
export CARGO_BUILD_JOBS="${build_jobs}"
echo "Build parallelism: ${build_jobs} compile jobs, 1 nvcc thread per job (CPUs: ${cpu_count})"

jlu_ubuntu_source="https://mirrors.jlu.edu.cn/ubuntu"
for source_file in \
  /etc/apt/sources.list \
  /etc/apt/sources.list.d/*.list \
  /etc/apt/sources.list.d/*.sources; do
  if [[ -f "${source_file}" ]]; then
    sed -i \
      -e "s|https\?://archive\.ubuntu\.com/ubuntu|${jlu_ubuntu_source}|g" \
      -e "s|https\?://security\.ubuntu\.com/ubuntu|${jlu_ubuntu_source}|g" \
      "${source_file}"
  fi
done
echo "Using Ubuntu package source: ${jlu_ubuntu_source}"

apt_options=(
  -o Acquire::Retries=5
  -o Acquire::http::Pipeline-Depth=0
)

if ! apt-get "${apt_options[@]}" update; then
  echo "Configured Ubuntu mirror is unavailable; retrying the image's default source." >&2
  for source_file in \
    /etc/apt/sources.list \
    /etc/apt/sources.list.d/*.list \
    /etc/apt/sources.list.d/*.sources; do
    if [[ -f "${source_file}" ]]; then
      sed -i \
        -e 's|https\?://mirrors\.jlu\.edu\.cn/ubuntu|https://archive.ubuntu.com/ubuntu|g' \
        "${source_file}"
    fi
  done
  apt-get "${apt_options[@]}" update
fi

build_packages=(
  build-essential \
  ca-certificates \
  cmake \
  curl \
  git \
  jq \
  libssl-dev \
  ninja-build \
  patchelf \
  pkg-config \
  python3.12 \
  python3.12-dev \
  python3.12-venv \
  unzip
)

for attempt in 1 2 3; do
  if apt-get "${apt_options[@]}" install \
    --fix-missing \
    -y \
    --no-install-recommends \
    "${build_packages[@]}"; then
    break
  fi
  if (( attempt == 3 )); then
    echo "apt package installation failed after ${attempt} attempts." >&2
    exit 1
  fi
  echo "Retrying apt package installation (${attempt}/3)..." >&2
  apt-get "${apt_options[@]}" update
  sleep $((attempt * 5))
done

rm -rf /var/lib/apt/lists/*

curl --retry 5 --retry-delay 3 -LsSf https://astral.sh/uv/install.sh | sh
export PATH="${HOME}/.local/bin:${PATH}"
export UV_INDEX_URL="${PYPI_MIRROR_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

uv venv --python "${PYTHON_VERSION}" --system-site-packages

if [[ "${RUST_CHANGED}" != "true" ]]; then
  baseline_wheel="${RUNNER_TEMP}/base-wheel.whl"
  curl -fL \
    -H "Authorization: Bearer ${GH_TOKEN}" \
    -H "Accept: application/octet-stream" \
    "${BASE_WHEEL_URL}" \
    -o "${baseline_wheel}"
  unzip -p "${baseline_wheel}" vllm/vllm-rs > vllm/vllm-rs
  test -s vllm/vllm-rs
  chmod +x vllm/vllm-rs
fi

if [[ "${RUST_CHANGED}" == "true" ]]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain none
  source "${HOME}/.cargo/env"
  rustup toolchain install 1.95.0 --profile minimal
  export PATH="${HOME}/.cargo/bin:${PATH}"
fi

sed -E -i \
  's/^nvidia-cutlass-dsl\[cu13\]([<>=])/nvidia-cutlass-dsl\1/' \
  requirements/cuda.txt
if ! uv pip install --python .venv/bin/python \
  -r requirements/build/cuda.txt \
  -r requirements/cuda.txt \
  --torch-backend=cu128; then
  echo "Package mirror install failed; retrying through the configured proxy and PyPI." >&2
  export UV_INDEX_URL="https://pypi.org/simple"
  uv pip install --python .venv/bin/python \
    -r requirements/build/cuda.txt \
    -r requirements/cuda.txt \
    --torch-backend=cu128
fi

.venv/bin/python - <<'PY'
import torch

if torch.version.cuda is None:
    raise SystemExit("The resolved PyTorch package is CPU-only.")
print(f"PyTorch CUDA runtime: {torch.version.cuda}")
PY

rm -rf build dist vllm.egg-info
rm -rf .deps/*-build .deps/*-subbuild
if ! .venv/bin/python setup.py bdist_wheel --dist-dir=dist; then
  echo "Native wheel build failed; collecting container diagnostics." >&2
  free -h >&2 || true
  df -h >&2 || true
  exit 1
fi

wheel_path="$(find dist -maxdepth 1 -type f -name '*.whl' -print -quit)"
if [[ -z "${wheel_path}" ]]; then
  echo "The build produced no wheel." >&2
  exit 1
fi
case "$(basename "${wheel_path}")" in
  *-cp312-cp312-linux_x86_64.whl) ;;
  *)
    echo "Unexpected wheel platform: $(basename "${wheel_path}")" >&2
    exit 1
    ;;
esac

wheel_listing="${RUNNER_TEMP}/wheel.listing"
unzip -l "${wheel_path}" > "${wheel_listing}"
grep -q "vllm/vllm-rs" "${wheel_listing}"
grep -q "vllm/_C.abi3.so" "${wheel_listing}"

metadata_name="$(awk '/\.dist-info\/METADATA$/ {print $4; exit}' "${wheel_listing}")"
unzip -p "${wheel_path}" "${metadata_name}" > "${RUNNER_TEMP}/wheel.metadata"
actual_version="$(sed -n 's/^Version: //p' "${RUNNER_TEMP}/wheel.metadata" | sed -n '1p')"
if [[ "${actual_version}" != "${VLLM_VERSION_OVERRIDE}" ]]; then
  echo "Wheel version mismatch: ${actual_version} != ${VLLM_VERSION_OVERRIDE}" >&2
  exit 1
fi
echo "Validated $(basename "${wheel_path}")"
