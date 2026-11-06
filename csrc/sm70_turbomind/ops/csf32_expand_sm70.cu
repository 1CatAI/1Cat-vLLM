// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Expansion of csf_32x2_v1 expert scales into the skinny QPN scale raster
// ([expert][tile N/32][group K/16][position 32] bytes, see
// vllm/model_executor/layers/quantization/utils/csf32_expand.py).
//
// A CSF block is 32 rows x 2 column groups; a code byte holds one row's two
// scales (low nibble first group, high nibble second), the scale is the row's
// base plus the nibble. The raster stores a block's 32 row scales for one
// group as a 32-byte line, permuted by the card's fragment order. A thread
// owns four neighbouring raster positions, so it reads the four rows' code
// bytes (one 32-byte line of the codes, served from L1 after the first
// thread) and writes two 32-bit words. Several tile pairs are loaded before
// any is stored: the byte-per-lane Triton walk spent about twenty
// instructions per scale and ran at a third of the memory bandwidth. The
// scales the four-bit codes cannot express (exception records, sorted by
// block) are written by the same thread that owns the row, so no second pass
// over the raster is needed.

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

constexpr int kBlocksPerTile = 8;   // 4 block rows x 2 column pairs
constexpr int kLinesPerBlock = 32;  // raster positions = rows of a block
constexpr int kQuadsPerLine = 8;    // 4 positions per thread
constexpr int kThreadsX = kBlocksPerTile * kQuadsPerLine;  // 64
constexpr int kPairLanes = 4;  // tile pairs a block works on at once
constexpr int kUnroll = 2;     // loads in flight per thread and lane

__global__ void csf32_expand_kernel(
    uint8_t* __restrict__ out, int64_t out_expert_stride,
    const int32_t* __restrict__ gids, const int32_t* __restrict__ goff,
    int32_t num_slots, const uint8_t* __restrict__ bases,
    const uint8_t* __restrict__ codes, const int32_t* __restrict__ entries,
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ page_bases,
    const int32_t* __restrict__ exceptions,
    const int64_t* __restrict__ exception_start, int num_pages, int rows,
    int groups, int projections, int pairs_per_program) {
  const int slot = blockIdx.x;
  if (goff[slot] >= num_slots) return;
  const int64_t expert = gids[slot];
  const int row_tiles_per_projection = rows / 128;
  const int projection = blockIdx.y / row_tiles_per_projection;
  const int tile_row = blockIdx.y % row_tiles_per_projection;
  const int tile_pairs = groups / 4;
  const int64_t matrix = expert * projections + projection;

  // Row at each raster position: the inverse of the entries table.
  __shared__ uint8_t row_at_position[kLinesPerBlock];
  if (threadIdx.y == 0 && threadIdx.x < kLinesPerBlock) {
    row_at_position[entries[threadIdx.x]] = static_cast<uint8_t>(threadIdx.x);
  }
  __syncthreads();

  const int at = threadIdx.x / kQuadsPerLine;
  const int quad = threadIdx.x % kQuadsPerLine;
  const int block_row = tile_row * 4 + (at & 3);
  int rows_of[4];
  int base_of[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    rows_of[k] = row_at_position[quad * 4 + k];
    base_of[k] = bases[matrix * rows + block_row * 32 + rows_of[k]];
  }
  const uint8_t* code_block =
      codes + matrix * (static_cast<int64_t>(rows) * groups / 2) +
      static_cast<int64_t>(tile_row) * tile_pairs * 256 + at * 32;
  // Exceptions of this matrix: the offsets row holds 257 entries per page of
  // 256 blocks, a block's records start at the page origin plus its prefix.
  const int num_blocks = (rows / 32) * (groups / 2);
  const int32_t* matrix_offsets = offsets + matrix * (num_blocks + num_pages);
  const int32_t* matrix_pages = page_bases + matrix * num_pages;
  const int32_t* matrix_records = exceptions + exception_start[matrix];
  const int out_tile = projection * (rows / 32) + block_row;
  uint8_t* out_tile_ptr = out + expert * out_expert_stride +
                          static_cast<int64_t>(out_tile) * groups * 32 +
                          quad * 4;

  const int first = blockIdx.z * pairs_per_program;
  const int end = first + pairs_per_program;
  for (int step = first; step < end; step += kPairLanes * kUnroll) {
    uint32_t code[kUnroll][4];
    int record_first[kUnroll], record_last[kUnroll], record_origin[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int tile_pair = step + u * kPairLanes + threadIdx.y;
      if (tile_pair < end) {
        const uint8_t* line =
            code_block + static_cast<int64_t>(tile_pair) * 256;
#pragma unroll
        for (int k = 0; k < 4; ++k) code[u][k] = line[rows_of[k]];
        const int block =
            (tile_row * tile_pairs + tile_pair) * kBlocksPerTile + at;
        const int page = block >> 8;
        record_first[u] = matrix_offsets[block + page];
        record_last[u] = matrix_offsets[block + page + 1];
        record_origin[u] = matrix_pages[page] >> 1;
      }
    }
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int tile_pair = step + u * kPairLanes + threadIdx.y;
      if (tile_pair < end) {
        uint32_t low = 0, high = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          low |= ((base_of[k] + (code[u][k] & 0xF)) & 0xFF) << (8 * k);
          high |= ((base_of[k] + (code[u][k] >> 4)) & 0xFF) << (8 * k);
        }
        // Scales the codes cannot express: a record names a row and a column
        // of the block (low byte 2 * row + column) and carries the value.
        for (int j = record_first[u]; j < record_last[u]; ++j) {
          const int32_t record = matrix_records[record_origin[u] + j];
          const int row = (record & 0xFF) >> 1;
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            if (rows_of[k] == row) {
              uint32_t& word = (record & 1) ? high : low;
              word = (word & ~(0xFFu << (8 * k))) |
                     (static_cast<uint32_t>(record >> 8) << (8 * k));
            }
          }
        }
        // Group pair of this block: two tile pairs' worth per tile pair, the
        // block's half selects which of the pair's two column pairs it is.
        uint8_t* pair_ptr =
            out_tile_ptr + static_cast<int64_t>(tile_pair * 2 + (at >> 2)) * 64;
        *reinterpret_cast<uint32_t*>(pair_ptr) = low;
        *reinterpret_cast<uint32_t*>(pair_ptr + 32) = high;
      }
    }
  }
}

}  // namespace

void csf32_expand_sm70(torch::Tensor out, torch::Tensor bases,
                       torch::Tensor codes, torch::Tensor page_bases,
                       torch::Tensor offsets, torch::Tensor exceptions,
                       torch::Tensor exception_start, torch::Tensor entries,
                       torch::Tensor gids, torch::Tensor goff, int64_t rows,
                       int64_t groups, int64_t projections, int64_t group_rows,
                       int64_t pairs_per_program, int64_t runs) {
  TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kUInt8 && out.dim() == 2 &&
              out.stride(1) == 1);
  const c10::cuda::CUDAGuard device_guard(out.device());
  for (const auto& tensor : {bases, codes, page_bases, offsets, exceptions,
                             exception_start, entries, gids, goff}) {
    TORCH_CHECK(tensor.device() == out.device(), "CSF tensors share a device");
    TORCH_CHECK(tensor.is_contiguous(), "CSF tensors must be contiguous");
  }
  TORCH_CHECK(bases.dtype() == torch::kUInt8 && codes.dtype() == torch::kUInt8);
  TORCH_CHECK(page_bases.dtype() == torch::kInt &&
              offsets.dtype() == torch::kInt &&
              exceptions.dtype() == torch::kInt &&
              exception_start.dtype() == torch::kLong);
  TORCH_CHECK(entries.dtype() == torch::kInt && entries.numel() == 32);
  TORCH_CHECK(gids.dtype() == torch::kInt && goff.dtype() == torch::kInt);
  TORCH_CHECK(
      rows > 0 && rows % 128 == 0 && groups > 0 && groups % 4 == 0,
      "CSF expansion supports full tiles only (rows % 128, groups % 4)");
  const int64_t tile_pairs = groups / 4;
  TORCH_CHECK(runs > 0 && pairs_per_program > 0 &&
                  pairs_per_program * runs == tile_pairs,
              "runs must split the row's tile pairs evenly");
  TORCH_CHECK(group_rows > 0 && group_rows <= gids.size(0) &&
              goff.size(0) == gids.size(0) + 1);
  TORCH_CHECK(out.stride(0) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(out.data_ptr()) % 4 == 0,
              "raster rows must be 4-byte aligned");
  const int64_t matrices = bases.numel() / rows;
  TORCH_CHECK(bases.numel() == matrices * rows &&
                  codes.numel() == matrices * rows * groups / 2,
              "CSF plane sizes");
  TORCH_CHECK(out.size(0) * projections >= matrices &&
                  out.size(1) >= projections * rows * groups,
              "the raster holds every expert's projections");
  const int64_t num_blocks = (rows / 32) * (groups / 2);
  TORCH_CHECK(
      page_bases.dim() == 2 && page_bases.size(0) == matrices &&
          offsets.numel() == matrices * (num_blocks + page_bases.size(1)) &&
          exception_start.numel() == matrices,
      "CSF exception tables");
  const int64_t row_tiles = projections * (rows / 128);
  TORCH_CHECK(row_tiles <= 65535 && runs <= 65535 && group_rows <= 2147483647);

  const dim3 grid(static_cast<unsigned>(group_rows),
                  static_cast<unsigned>(row_tiles),
                  static_cast<unsigned>(runs));
  const dim3 block(kThreadsX, kPairLanes);
  csf32_expand_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      out.data_ptr<uint8_t>(), out.stride(0), gids.data_ptr<int32_t>(),
      goff.data_ptr<int32_t>(), static_cast<int32_t>(gids.size(0)),
      bases.data_ptr<uint8_t>(), codes.data_ptr<uint8_t>(),
      entries.data_ptr<int32_t>(), offsets.data_ptr<int32_t>(),
      page_bases.data_ptr<int32_t>(), exceptions.data_ptr<int32_t>(),
      exception_start.data_ptr<int64_t>(), static_cast<int>(page_bases.size(1)),
      static_cast<int>(rows), static_cast<int>(groups),
      static_cast<int>(projections), static_cast<int>(pairs_per_program));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
