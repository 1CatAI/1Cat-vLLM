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
    TORCH_CHECK(index >= 0 && index < num_rows, "PLE disk row ID out of range");
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
