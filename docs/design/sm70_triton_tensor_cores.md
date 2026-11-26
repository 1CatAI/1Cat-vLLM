# Tensor-core Triton on Volta and Turing (sm7x)

Stock Triton lowers every `tl.dot` to FMA before SM80. The SM70-family routes
behind `kernel_config.triton_sm7x_tensor_cores` need a Triton that lowers
`tl.dot` to `mma.m8n8k4` on sm_70 and to `mma.m16n8k8` on sm_75. That Triton
is Triton 3.6.0 plus 26 commits in the NVIDIA backend and the TritonGPU
dialect: branch `sm7x-tensor-cores` of
<https://github.com/Peuqui/triton>, shown upstream in
<https://github.com/triton-lang/triton/discussions/12517>.

## Prebuilt wheel

Release `v3.6.0-sm7x.2adbae8bc2` of that fork,
<https://github.com/Peuqui/triton/releases/tag/v3.6.0-sm7x.2adbae8bc2>:

| File | `triton-3.6.0+sm7x.2adbae8bc2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl` |
| --- | --- |
| Source | commit `2adbae8bc2` of branch `sm7x-tensor-cores` |
| Platform | CPython 3.12, Linux x86_64 (manylinux_2_28 build image) |
| SHA-256 | `7b11b73c51d9cd8c7f494c001785ab0c1d78edb0db994dbfb82224b86d909bf1` |

Install it in place of the stock `triton==3.6.0` of the environment and check
the installed build; `triton.__version__` stays `3.6.0`, the build shows in the
package metadata:

```bash
uv pip install "https://github.com/Peuqui/triton/releases/download/v3.6.0-sm7x.2adbae8bc2/triton-3.6.0+sm7x.2adbae8bc2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
.venv/bin/python -c "import importlib.metadata as m; print(m.version('triton'))"  # 3.6.0+sm7x.2adbae8bc2
```

`uv pip install --reinstall triton==3.6.0` goes back to stock Triton; without
`--reinstall` the installed build already counts as `3.6.0`. The wheel changes the
lowering of every `tl.dot` kernel on sm_70 and sm_75, with or without the
setting; on other architectures it compiles like Triton 3.6.0.

## Using it

Start the engine with `--kernel-config '{"triton_sm7x_tensor_cores":true}'`.
On first use per architecture the engine compiles a 16x16 fp16 `tl.dot` and
requires `mma.sync` in its PTX (`vllm/triton_utils/mma_probe.py`); with a Triton
that lowers it to FMA, the first layer that asks
`KernelConfig.use_triton_sm7x_tensor_cores` stops the engine and names this
page.

To build from source instead, check out the branch and follow the source build
section of Triton's README in that fork.
