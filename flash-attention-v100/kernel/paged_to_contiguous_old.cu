#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

__global__ void paged_to_contiguous_kernel(
    const __half* __restrict__ paged_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
          __half* __restrict__ contiguous,
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

        const size_t paged_offset = (size_t)physical_block_idx * block_size * elements_per_token
                                  + (size_t)block_offset * elements_per_token;

        const size_t cont_offset = (size_t)(output_offset + token_idx) * elements_per_token;

        for (int elem_idx = tid; elem_idx < elements_per_token; elem_idx += num_threads) {
            contiguous[cont_offset + elem_idx] = paged_cache[paged_offset + elem_idx];
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
    const int block_size = paged_cache.size(1);
    const int num_heads = paged_cache.size(2);
    const int head_dim = paged_cache.size(3);

    int total_tokens = 0;
    auto seq_lens_cpu = seq_lens.cpu();
    for (int i = 0; i < batch_size; ++i) {
        total_tokens += seq_lens_cpu[i].item<int>();
    }

    auto contiguous = torch::zeros(
        {total_tokens, num_heads, head_dim},
        torch::TensorOptions().dtype(paged_cache.dtype()).device(paged_cache.device())
    );

    const int threads = 256;
    paged_to_contiguous_kernel<<<batch_size, threads>>>(
        reinterpret_cast<const __half*>(paged_cache.data_ptr<at::Half>()),
        block_table.data_ptr<int>(),
        seq_lens.data_ptr<int>(),
        reinterpret_cast<__half*>(contiguous.data_ptr<at::Half>()),
        block_size,
        num_heads,
        head_dim,
        max_num_blocks
    );

    return contiguous;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("paged_to_contiguous", &paged_to_contiguous, "Paged KV Cache to Contiguous");
}
