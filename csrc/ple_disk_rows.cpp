// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/all.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#ifdef __linux__
  #include <sys/mman.h>
  #include <unistd.h>
#endif

void ple_disk_gather_u8(torch::Tensor ids, torch::Tensor pointers,
                        int64_t shard_size, int64_t num_rows, int64_t row_bytes,
                        torch::Tensor out) {
  TORCH_CHECK(ids.device().is_cpu() && pointers.device().is_cpu() &&
                  out.device().is_cpu(),
              "PLE disk gather requires CPU tensors");
  TORCH_CHECK(ids.scalar_type() == at::kLong &&
                  pointers.scalar_type() == at::kLong &&
                  out.scalar_type() == at::kByte,
              "PLE disk gather requires int64 IDs/pointers and uint8 output");
  TORCH_CHECK(
      ids.is_contiguous() && pointers.is_contiguous() && out.is_contiguous(),
      "PLE disk gather requires contiguous tensors");
  TORCH_CHECK(shard_size > 0 && num_rows > 0 && row_bytes > 0,
              "PLE disk gather geometry must be positive");
  TORCH_CHECK(
      out.numel() / row_bytes == ids.numel() && out.numel() % row_bytes == 0,
      "PLE disk gather output size mismatch");
  const auto shards = num_rows / shard_size + (num_rows % shard_size != 0);
  TORCH_CHECK(pointers.numel() == shards,
              "PLE disk gather shard count mismatch");

  const auto* indices = ids.data_ptr<int64_t>();
  const auto* bases = pointers.data_ptr<int64_t>();
  auto* destination = out.data_ptr<uint8_t>();
  std::vector<const uint8_t*> sources;
  sources.reserve(ids.numel());
  std::vector<uintptr_t> pages;
#ifdef __linux__
  const auto system_page_size = sysconf(_SC_PAGE_SIZE);
  TORCH_CHECK(system_page_size > 0,
              "PLE disk gather cannot determine page size");
  const auto page_size = static_cast<uintptr_t>(system_page_size);
#endif
  for (int64_t row = 0; row < ids.numel(); ++row) {
    const auto index = indices[row];
    TORCH_CHECK_INDEX(index >= 0 && index < num_rows,
                      "PLE disk row id out of range: ", index, " not in [0, ",
                      num_rows, ")");
    const auto shard = index / shard_size;
    const auto local = index % shard_size;
    TORCH_CHECK(bases[shard] > 0, "PLE disk shard pointer is invalid");
    TORCH_CHECK(static_cast<uint64_t>(local) <=
                    std::numeric_limits<uintptr_t>::max() / row_bytes,
                "PLE disk row offset overflow");
    const auto offset = static_cast<uintptr_t>(local) * row_bytes;
    const auto base = static_cast<uintptr_t>(bases[shard]);
    TORCH_CHECK(
        base <= std::numeric_limits<uintptr_t>::max() - offset &&
            base + offset <= std::numeric_limits<uintptr_t>::max() - row_bytes,
        "PLE disk row address overflow");
    const auto address = base + offset;
    sources.push_back(reinterpret_cast<const uint8_t*>(address));
#ifdef __linux__
    const auto last_page = (address + row_bytes - 1) / page_size * page_size;
    for (auto page = address / page_size * page_size;; page += page_size) {
      pages.push_back(page);
      if (page == last_page) break;
    }
#endif
  }

#ifdef __linux__
  // Only pages covering requested rows are considered. Never populate the
  // whole table. Residency can change after this snapshot; memcpy remains
  // the ordinary file-backed read if a page is subsequently reclaimed.
  std::sort(pages.begin(), pages.end());
  pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
  for (const auto page : pages) {
    unsigned char resident = 0;
    const auto status =
        mincore(reinterpret_cast<void*>(page), page_size, &resident);
    TORCH_CHECK(status == 0 || errno != ENOMEM,
                "PLE disk row page is not mapped");
    if (status == 0 && !(resident & 1)) {
      // Advice failure affects performance only, not byte selection/order.
      madvise(reinterpret_cast<void*>(page), page_size, MADV_WILLNEED);
    }
  }
#endif
  for (size_t row = 0; row < sources.size(); ++row) {
    std::memcpy(destination + row * row_bytes, sources[row], row_bytes);
  }
}

void ple_disk_cached_gather_u8(torch::Tensor ids, torch::Tensor pointers,
                               int64_t shard_size, int64_t num_rows,
                               int64_t row_bytes, torch::Tensor out,
                               torch::Tensor cache_ids,
                               torch::Tensor cache_rows) {
  TORCH_CHECK(ids.device().is_cpu() && out.device().is_cpu() &&
                  cache_ids.device().is_cpu() && cache_rows.device().is_cpu(),
              "PLE cached gather requires CPU storage");
  TORCH_CHECK(ids.scalar_type() == at::kLong &&
                  out.scalar_type() == at::kByte &&
                  cache_ids.scalar_type() == at::kLong &&
                  cache_rows.scalar_type() == at::kByte,
              "PLE cached gather requires int64 keys and uint8 rows");
  TORCH_CHECK(ids.is_contiguous() && out.is_contiguous() &&
                  cache_ids.is_contiguous() && cache_rows.is_contiguous(),
              "PLE cached gather requires contiguous storage");
  const auto capacity = cache_ids.numel();
  TORCH_CHECK(row_bytes > 0 && capacity > 0 &&
                  cache_rows.numel() / row_bytes == capacity &&
                  cache_rows.numel() % row_bytes == 0 &&
                  out.numel() / row_bytes == ids.numel() &&
                  out.numel() % row_bytes == 0,
              "PLE cached gather geometry mismatch");
  const auto* indices = ids.data_ptr<int64_t>();
  auto* keys = cache_ids.data_ptr<int64_t>();
  auto* cache = cache_rows.data_ptr<uint8_t>();
  auto* destination = out.data_ptr<uint8_t>();
  std::vector<int64_t> misses;
  std::vector<int64_t> positions;
  for (int64_t i = 0; i < ids.numel(); ++i) {
    TORCH_CHECK_INDEX(indices[i] >= 0 && indices[i] < num_rows,
                      "PLE disk row id out of range");
  }
  for (int64_t i = 0; i < ids.numel(); ++i) {
    const auto id = indices[i];
    const auto slot = id % capacity;
    if (keys[slot] == id) {
      std::memcpy(destination + i * row_bytes, cache + slot * row_bytes,
                  row_bytes);
    } else {
      misses.push_back(id);
      positions.push_back(i);
    }
  }
  if (misses.empty()) return;
  auto miss_ids = torch::tensor(misses, ids.options());
  auto rows = torch::empty({static_cast<int64_t>(misses.size()), row_bytes},
                           out.options());
  ple_disk_gather_u8(miss_ids, pointers, shard_size, num_rows, row_bytes, rows);
  const auto* data = rows.data_ptr<uint8_t>();
  for (size_t i = 0; i < misses.size(); ++i) {
    const auto slot = misses[i] % capacity;
    std::memcpy(destination + positions[i] * row_bytes, data + i * row_bytes,
                row_bytes);
    std::memcpy(cache + slot * row_bytes, data + i * row_bytes, row_bytes);
    keys[slot] = misses[i];
  }
}
