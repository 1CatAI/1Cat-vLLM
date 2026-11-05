// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/all.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#ifdef __linux__
  #include <sys/mman.h>
  #include <unistd.h>

namespace {

// Copies the rows in address order and advises only a bounded window of pages
// ahead of the copy. Advising every page of a long request up front let
// reclaim take the read-ahead pages, which enter the oldest LRU generation
// untouched, before the copy reached them; the copy then read them back one
// synchronous page at a time (2026-10-18: ~3,500 pages/s from the checkpoint
// SSD). Only pages covering requested rows are considered, never the whole
// table. Residency can change after mincore; memcpy remains the ordinary
// file-backed read if a page is reclaimed anyway.
void copy_rows_with_prefetch_window(const std::vector<const uint8_t*>& sources,
                                    int64_t row_bytes,
                                    int64_t prefetch_window_bytes,
                                    uint8_t* destination) {
  const auto system_page_size = sysconf(_SC_PAGE_SIZE);
  TORCH_CHECK(system_page_size > 0,
              "PLE disk gather cannot determine page size");
  const auto page_size = static_cast<uintptr_t>(system_page_size);
  const auto bytes = static_cast<uintptr_t>(row_bytes);
  std::vector<size_t> order(sources.size());
  std::iota(order.begin(), order.end(), size_t{0});
  std::sort(order.begin(), order.end(),
            [&](size_t a, size_t b) { return sources[a] < sources[b]; });
  // Unique pages in copy order and, per copied row, one past its last page.
  std::vector<uintptr_t> pages;
  std::vector<size_t> pages_through_row(order.size());
  for (size_t i = 0; i < order.size(); ++i) {
    const auto address = reinterpret_cast<uintptr_t>(sources[order[i]]);
    const auto last_page = (address + bytes - 1) / page_size * page_size;
    for (auto page = address / page_size * page_size;; page += page_size) {
      if (pages.empty() || pages.back() < page) pages.push_back(page);
      if (page == last_page) break;
    }
    pages_through_row[i] = pages.size();
  }
  const auto window_pages =
      (static_cast<uintptr_t>(prefetch_window_bytes) + page_size - 1) /
      page_size;
  size_t advised = 0;
  for (size_t i = 0; i < order.size(); ++i) {
    const auto target =
        std::min(pages.size(), pages_through_row[i] + window_pages);
    for (; advised < target; ++advised) {
      auto* page = reinterpret_cast<void*>(pages[advised]);
      unsigned char resident = 0;
      const auto status = mincore(page, page_size, &resident);
      TORCH_CHECK(status == 0 || errno != ENOMEM,
                  "PLE disk row page is not mapped");
      if (status == 0 && !(resident & 1)) {
        // Advice failure affects performance only, not byte selection/order.
        madvise(page, page_size, MADV_WILLNEED);
      }
    }
    std::memcpy(destination + order[i] * bytes, sources[order[i]], bytes);
  }
}

}  // namespace
#endif

void ple_disk_gather_u8(torch::Tensor ids, torch::Tensor pointers,
                        int64_t shard_size, int64_t num_rows, int64_t row_bytes,
                        torch::Tensor out, int64_t prefetch_window_bytes) {
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
  TORCH_CHECK(prefetch_window_bytes >= 0,
              "PLE disk prefetch window must not be negative");
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
    sources.push_back(reinterpret_cast<const uint8_t*>(base + offset));
  }

#ifdef __linux__
  if (prefetch_window_bytes > 0) {
    copy_rows_with_prefetch_window(sources, row_bytes, prefetch_window_bytes,
                                   destination);
    return;
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
                               torch::Tensor cache_rows,
                               int64_t prefetch_window_bytes) {
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
  ple_disk_gather_u8(miss_ids, pointers, shard_size, num_rows, row_bytes, rows,
                     prefetch_window_bytes);
  const auto* data = rows.data_ptr<uint8_t>();
  for (size_t i = 0; i < misses.size(); ++i) {
    const auto slot = misses[i] % capacity;
    std::memcpy(destination + positions[i] * row_bytes, data + i * row_bytes,
                row_bytes);
    std::memcpy(cache + slot * row_bytes, data + i * row_bytes, row_bytes);
    keys[slot] = misses[i];
  }
}
