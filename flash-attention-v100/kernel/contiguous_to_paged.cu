#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

__global__ void contiguous_to_paged_kernel(
    const __half* __restrict__ key,
    const __half* __restrict__ value,
    const int64_t* __restrict__ slot_mapping,
          __half* __restrict__ key_cache,
          __half* __restrict__ value_cache,
    const int64_t key_stride,
    const int64_t value_stride,
    const int64_t block_stride,
    const int64_t page_stride,
    const int64_t head_stride,
    const int block_size,
    const int num_heads,
    const int head_dim
) {

    const int token_idx = blockIdx.x;
    const int64_t slot_idx = slot_mapping[token_idx];

    if (slot_idx < 0) {
        return;
    }

    const int64_t block_idx = slot_idx / block_size;
    const int64_t block_offset = slot_idx % block_size;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    const __half* __restrict__ key_src = key + token_idx * key_stride;
    const __half* __restrict__ value_src = value + token_idx * value_stride;

    __half* __restrict__ key_dst = key_cache + block_idx * block_stride + block_offset * page_stride;
    __half* __restrict__ value_dst = value_cache + block_idx * block_stride + block_offset * page_stride;

    for (int head_idx = 0; head_idx < num_heads; head_idx++) {
        const __half* key_head_src = key_src + head_idx * head_dim;
        const __half* value_head_src = value_src + head_idx * head_dim;
        __half* key_head_dst = key_dst + head_idx * head_stride;
        __half* value_head_dst = value_dst + head_idx * head_stride;

        for (int elem_idx = tid; elem_idx < head_dim; elem_idx += num_threads) {
            key_head_dst[elem_idx] = key_head_src[elem_idx];
            value_head_dst[elem_idx] = value_head_src[elem_idx];
        }
    }
}

void contiguous_to_paged(
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor slot_mapping,
    torch::Tensor key_cache,
    torch::Tensor value_cache
) {
    const int num_tokens = slot_mapping.size(0);
    const int num_heads = key.size(1);
    const int head_dim = key.size(2);
    const int block_size = key_cache.size(1);

    const int64_t key_stride = key.stride(0);
    const int64_t value_stride = value.stride(0);
    const int64_t block_stride = key_cache.stride(0);
    const int64_t page_stride = key_cache.stride(1);
    const int64_t head_stride = key_cache.stride(2);

    const int threads = 256;
    contiguous_to_paged_kernel<<<num_tokens, threads>>>(
        reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(value.data_ptr<at::Half>()),
        slot_mapping.data_ptr<int64_t>(),
        reinterpret_cast<__half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(value_cache.data_ptr<at::Half>()),
        key_stride,
        value_stride,
        block_stride,
        page_stride,
        head_stride,
        block_size,
        num_heads,
        head_dim
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("contiguous_to_paged", &contiguous_to_paged, "Contiguous K/V to Paged KV Cache");
}
