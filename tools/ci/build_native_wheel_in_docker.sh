#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
mkdir -p "${RUNNER_TEMP}"

cpu_count="$(nproc)"
if (( cpu_count < 20 )); then
  echo "Build container exposes only ${cpu_count} CPUs; 20 are required." >&2
  exit 1
fi
echo "Build parallelism: 20 compile jobs, 1 nvcc thread per job (CPUs: ${cpu_count})"

apt-get update
apt-get install -y --no-install-recommends \
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
rm -rf /var/lib/apt/lists/*

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="${HOME}/.local/bin:${PATH}"

uv venv --python "${PYTHON_VERSION}"

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
uv pip install --python .venv/bin/python \
  -r requirements/build/cuda.txt \
  -r requirements/cuda.txt \
  --torch-backend=cu128

.venv/bin/python - <<'PY'
import torch

if torch.version.cuda is None:
    raise SystemExit("The resolved PyTorch package is CPU-only.")
print(f"PyTorch CUDA runtime: {torch.version.cuda}")
PY

rm -rf build dist vllm.egg-info
rm -rf .deps/*-build .deps/*-subbuild
.venv/bin/python setup.py bdist_wheel --dist-dir=dist

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
