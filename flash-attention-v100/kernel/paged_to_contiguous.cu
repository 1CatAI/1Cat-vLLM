#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

__global__ void paged_to_contiguous_stride_aware_kernel(
    const __half* __restrict__ paged_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
          __half* __restrict__ contiguous,
    const int64_t block_stride,
    const int64_t page_stride,
    const int64_t head_stride,
    const int block_size,
    const int num_heads,
    const int head_dim,
    const int max_num_blocks
) {

    const int batch_idx = blockIdx.x;
    const int seq_len = seq_lens[batch_idx];
    if (seq_len == 0) return;

    int output_offset = 0;
    for (int i = 0; i < batch_idx; ++i) {
        output_offset += seq_lens[i];
    }

    const int* block_table_seq = block_table + batch_idx * max_num_blocks;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int elements_per_token = num_heads * head_dim;

    for (int token_idx = 0; token_idx < seq_len; ++token_idx) {

        const int virtual_block_idx = token_idx / block_size;
        const int block_offset = token_idx % block_size;
        const int physical_block_idx = block_table_seq[virtual_block_idx];

        const __half* token_base = paged_cache
            + physical_block_idx * block_stride
            + block_offset * page_stride;

        __half* output_token = contiguous + (output_offset + token_idx) * elements_per_token;

        for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
            const __half* head_src = token_base + head_idx * head_stride;
            __half* head_dst = output_token + head_idx * head_dim;

            for (int elem_idx = tid; elem_idx < head_dim; elem_idx += num_threads) {
                head_dst[elem_idx] = head_src[elem_idx];
            }
        }
    }
}

__global__ void paged_kv_to_contiguous_stride_aware_kernel(
    const __half* __restrict__ paged_key_cache,
    const __half* __restrict__ paged_value_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
          __half* __restrict__ contiguous_key,
          __half* __restrict__ contiguous_value,
    const int64_t key_block_stride,
    const int64_t key_page_stride,
    const int64_t key_head_stride,
    const int64_t value_block_stride,
    const int64_t value_page_stride,
    const int64_t value_head_stride,
    const int block_size,
    const int num_heads,
    const int head_dim,
    const int max_num_blocks
) {
    const int batch_idx = blockIdx.x;
    const int seq_len = seq_lens[batch_idx];
    if (seq_len == 0) return;

    int output_offset = 0;
    for (int i = 0; i < batch_idx; ++i) {
        output_offset += seq_lens[i];
    }

    const int* block_table_seq = block_table + batch_idx * max_num_blocks;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int elements_per_token = num_heads * head_dim;

    for (int token_idx = 0; token_idx < seq_len; ++token_idx) {
        const int virtual_block_idx = token_idx / block_size;
        const int block_offset = token_idx % block_size;
        const int physical_block_idx = block_table_seq[virtual_block_idx];

        const __half* key_token_base = paged_key_cache
            + physical_block_idx * key_block_stride
            + block_offset * key_page_stride;
        const __half* value_token_base = paged_value_cache
            + physical_block_idx * value_block_stride
            + block_offset * value_page_stride;

        __half* key_output_token =
            contiguous_key + (output_offset + token_idx) * elements_per_token;
        __half* value_output_token =
            contiguous_value + (output_offset + token_idx) * elements_per_token;

        for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
            const __half* key_head_src = key_token_base + head_idx * key_head_stride;
            const __half* value_head_src =
                value_token_base + head_idx * value_head_stride;
            __half* key_head_dst = key_output_token + head_idx * head_dim;
            __half* value_head_dst = value_output_token + head_idx * head_dim;

            for (int elem_idx = tid; elem_idx < head_dim; elem_idx += num_threads) {
                key_head_dst[elem_idx] = key_head_src[elem_idx];
                value_head_dst[elem_idx] = value_head_src[elem_idx];
            }
        }
    }
}

torch::Tensor paged_to_contiguous(
    torch::Tensor paged_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens
) {
    const int batch_size = block_table.size(0);
    const int max_num_blocks = block_table.size(1);
    const int num_blocks = paged_cache.size(0);
    const int block_size = paged_cache.size(1);
    const int num_heads = paged_cache.size(2);
    const int head_dim = paged_cache.size(3);

    const int64_t block_stride = paged_cache.stride(0);
    const int64_t page_stride = paged_cache.stride(1);
    const int64_t head_stride = paged_cache.stride(2);

    const int total_tokens = batch_size * max_num_blocks * block_size;

    auto contiguous = torch::zeros(
        {total_tokens, num_heads, head_dim},
        torch::TensorOptions().dtype(paged_cache.dtype()).device(paged_cache.device())
    );

    const int threads = 256;
    paged_to_contiguous_stride_aware_kernel<<<batch_size, threads>>>(
        reinterpret_cast<const __half*>(paged_cache.data_ptr<at::Half>()),
        block_table.data_ptr<int>(),
        seq_lens.data_ptr<int>(),
        reinterpret_cast<__half*>(contiguous.data_ptr<at::Half>()),
        block_stride,
        page_stride,
        head_stride,
        block_size,
        num_heads,
        head_dim,
        max_num_blocks
    );

    return contiguous;
}

std::vector<torch::Tensor> paged_kv_to_contiguous(
    torch::Tensor paged_key_cache,
    torch::Tensor paged_value_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens
) {
    const int batch_size = block_table.size(0);
    const int max_num_blocks = block_table.size(1);
    const int block_size = paged_key_cache.size(1);
    const int num_heads = paged_key_cache.size(2);
    const int head_dim = paged_key_cache.size(3);

    const int64_t key_block_stride = paged_key_cache.stride(0);
    const int64_t key_page_stride = paged_key_cache.stride(1);
    const int64_t key_head_stride = paged_key_cache.stride(2);
    const int64_t value_block_stride = paged_value_cache.stride(0);
    const int64_t value_page_stride = paged_value_cache.stride(1);
    const int64_t value_head_stride = paged_value_cache.stride(2);

    const int total_tokens = batch_size * max_num_blocks * block_size;

    auto opts = torch::TensorOptions()
                    .dtype(paged_key_cache.dtype())
                    .device(paged_key_cache.device());
    auto contiguous_key = torch::zeros({total_tokens, num_heads, head_dim}, opts);
    auto contiguous_value = torch::zeros({total_tokens, num_heads, head_dim}, opts);

    const int threads = 256;
    paged_kv_to_contiguous_stride_aware_kernel<<<batch_size, threads>>>(
        reinterpret_cast<const __half*>(paged_key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(paged_value_cache.data_ptr<at::Half>()),
        block_table.data_ptr<int>(),
        seq_lens.data_ptr<int>(),
        reinterpret_cast<__half*>(contiguous_key.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(contiguous_value.data_ptr<at::Half>()),
        key_block_stride,
        key_page_stride,
        key_head_stride,
        value_block_stride,
        value_page_stride,
        value_head_stride,
        block_size,
        num_heads,
        head_dim,
        max_num_blocks
    );

    return {contiguous_key, contiguous_value};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("paged_to_contiguous", &paged_to_contiguous, "Paged KV Cache to Contiguous (Stride-Aware)");
    m.def("paged_kv_to_contiguous",
          &paged_kv_to_contiguous,
          "Paged KV Cache to Contiguous K/V Pair (Stride-Aware)");
}
