// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <cstdlib>
#include <algorithm>
#include <unordered_map>
#include <vector>

namespace vllm::gguf {

struct SharedEngramTable {
  int fd;
  uint64_t offset, rows;
};

// Two independently prefetched table channels, one raw row buffer per table.
// Each GPU writes its own request record and consumption sequence. The CPU
// lane never calls CUDA; it publishes readiness only after all preads and
// duplicate-row scatters complete. A buffer is reused only after all ranks
// consume its previous sequence.
class SharedEngramReader {
 public:
  static constexpr uint64_t kChannels = 2, kRanks = 4, kHeader = 8;

  static uint64_t rank_words(uint64_t maximum) { return maximum + 8; }
  static uint64_t channel_words(uint64_t maximum) {
    return kHeader + kRanks * rank_words(maximum);
  }
  static uint64_t control_words(uint64_t maximum) {
    return kChannels * channel_words(maximum);
  }

  SharedEngramReader(uint8_t* data, uint64_t bytes, uint64_t* control,
                     uint64_t maximum, uint64_t row_bytes,
                     std::vector<SharedEngramTable> tables, uint64_t timeout_ns)
      : data_(data),
        control_(control),
        maximum_(maximum),
        row_bytes_(row_bytes),
        timeout_(timeout_ns),
        tables_(std::move(tables)) {
    if (!data || !control || reinterpret_cast<uintptr_t>(control) % 64 ||
        !maximum || maximum > 16384 * 24 || row_bytes != 144 ||
        tables_.size() != kChannels || !timeout_ ||
        bytes < kChannels * maximum * row_bytes)
      throw std::invalid_argument("invalid shared Engram geometry");
    try {
      for (const auto& table : tables_) {
        struct stat info{};
        if (table.fd < 0 || !table.rows || fstat(table.fd, &info) ||
            info.st_size < 0 || table.offset > uint64_t(info.st_size) ||
            table.rows > (uint64_t(info.st_size) - table.offset) / row_bytes)
          throw std::invalid_argument("shared Engram table exceeds its file");
        const int fd = fcntl(table.fd, F_DUPFD_CLOEXEC, 0);
        if (fd < 0)
          throw std::runtime_error("cannot duplicate Engram source fd");
        fds_.push_back(fd);
      }
      std::fill_n(control_, control_words(maximum), 0);
      // A short decode reads at most 144 rows per channel. Creating and
      // joining eight threads for each batch can outlast the cached reads.
      // Keep the same bounded I/O parallelism for both hot and cold rows.
      // Cold rows are random 144-byte NVMe reads: queue depth (threads)
      // bounds IOPS. VLLM_V41_ENGRAM_READERS overrides the default.
      size_t readers = 64;
      if (const char* v = std::getenv("VLLM_V41_ENGRAM_READERS"))
        readers = std::max(1, std::atoi(v));
      for (size_t i = 0; i < readers; ++i)
        readers_.emplace_back([this] { read_worker(); });
      thread_ = std::thread([this] { run(); });
    } catch (...) {
      stop_readers();
      for (int fd : fds_) close(fd);
      throw;
    }
  }

  ~SharedEngramReader() {
    stop();
    for (int fd : fds_) close(fd);
  }

  void stop() {
    stopped_.store(true, std::memory_order_release);
    if (thread_.joinable()) thread_.join();
    stop_readers();
    for (size_t c = 0; c < kChannels; ++c) store(header(c) + 1, 1);
  }

  std::string error() {
    std::lock_guard<std::mutex> guard(mutex_);
    return error_;
  }

 private:
  static uint64_t now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }
  static uint64_t load(const uint64_t* p) {
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
  }
  static void store(uint64_t* p, uint64_t v) {
    __atomic_store_n(p, v, __ATOMIC_RELEASE);
  }
  uint64_t* header(size_t c) { return control_ + c * channel_words(maximum_); }
  uint64_t* rank(size_t c, size_t r) {
    return header(c) + kHeader + r * rank_words(maximum_);
  }
  void check(uint64_t begin) {
    if (stopped_.load(std::memory_order_acquire))
      throw std::runtime_error("shared Engram reader stopped during request");
    if (now() - begin >= timeout_)
      throw std::runtime_error("shared Engram request timed out");
  }

  void stop_readers() {
    {
      std::lock_guard<std::mutex> guard(read_mutex_);
      readers_stopped_ = true;
    }
    read_ready_.notify_all();
    for (auto& reader : readers_)
      if (reader.joinable()) reader.join();
  }

  void read_worker() {
    uint64_t completed = 0;
    std::unique_lock<std::mutex> guard(read_mutex_);
    for (;;) {
      read_ready_.wait(guard, [&] {
        return readers_stopped_ || read_generation_ != completed;
      });
      if (readers_stopped_) return;
      completed = read_generation_;
      // The coordinator cannot replace this task until all workers have
      // acknowledged it. Its captures remain alive in read_rows below.
      guard.unlock();
      read_task_();
      guard.lock();
      if (!--read_remaining_) read_done_.notify_one();
    }
  }

  void read_rows(size_t channel, uint64_t count, uint64_t begin) {
    auto* raw = data_ + channel * maximum_ * row_bytes_;
    const auto* ids = rank(channel, 0) + 8;
    std::unordered_map<uint64_t, uint64_t> first;
    std::vector<std::pair<uint64_t, uint64_t>> unique, duplicate;
    first.reserve(count);
    unique.reserve(count);
    for (uint64_t i = 0; i < count; ++i) {
      if (ids[i] >= tables_[channel].rows)
        throw std::runtime_error("shared Engram row id outside table");
      const auto inserted = first.emplace(ids[i], i);
      if (inserted.second)
        unique.emplace_back(ids[i], i);
      else
        duplicate.emplace_back(i, inserted.first->second);
    }
    std::sort(unique.begin(), unique.end());
    std::atomic<size_t> next{0};
    std::atomic<bool> failed{false};
    {
      std::lock_guard<std::mutex> guard(read_mutex_);
      read_task_ = [&] {
        for (;;) {
          const size_t i = next.fetch_add(1, std::memory_order_relaxed);
          if (i >= unique.size() || failed.load(std::memory_order_acquire))
            break;
          const auto [row, output] = unique[i];
          auto* target = raw + output * row_bytes_;
          uint64_t offset = tables_[channel].offset + row * row_bytes_;
          uint64_t left = row_bytes_;
          while (left) {
            const auto got = pread(fds_[channel], target, left, off_t(offset));
            if (got <= 0) {
              failed.store(true, std::memory_order_release);
              break;
            }
            left -= uint64_t(got);
            offset += uint64_t(got);
            target += got;
          }
        }
      };
      read_remaining_ = readers_.size();
      ++read_generation_;
    }
    read_ready_.notify_all();
    {
      std::unique_lock<std::mutex> guard(read_mutex_);
      read_done_.wait(guard, [&] { return !read_remaining_; });
      read_task_ = {};
    }
    if (failed.load(std::memory_order_acquire))
      throw std::runtime_error("shared Engram source returned a short read");
    check(begin);
    for (const auto& [to, from] : duplicate)
      std::memcpy(raw + to * row_bytes_, raw + from * row_bytes_, row_bytes_);
    auto* h = header(channel);
    h[2] = unique.size();
    h[3] = unique.size() * row_bytes_;
    h[4] = now() - begin;
    h[5] = count;
  }

  void run() {
    std::array<uint64_t, kChannels> sequence{1, 1}, started{0, 0};
    try {
      while (!stopped_.load(std::memory_order_acquire)) {
        bool work = false;
        for (size_t c = 0; c < kChannels; ++c) {
          bool all = true, consumed = true;
          for (size_t r = 0; r < kRanks; ++r) {
            auto* request = rank(c, r);
            const auto tag = load(request);
            if (tag > sequence[c])
              throw std::runtime_error("shared Engram sequence skipped");
            if (tag == sequence[c] && !started[c]) started[c] = now();
            all &= tag == sequence[c];
            consumed &= load(request + 1) == sequence[c] - 1;
          }
          if (!started[c]) continue;
          check(started[c]);
          if (!all || !consumed) continue;
          const auto count = rank(c, 0)[2];
          if (!count || count > maximum_)
            throw std::runtime_error("invalid shared Engram row count");
          for (size_t r = 1; r < kRanks; ++r) {
            if (rank(c, r)[2] != count ||
                !std::equal(rank(c, 0) + 8, rank(c, 0) + 8 + count,
                            rank(c, r) + 8))
              throw std::runtime_error(
                  "shared Engram ranks requested different rows");
          }
          read_rows(c, count, started[c]);
          store(header(c), sequence[c]);
          ++sequence[c];
          started[c] = 0;
          work = true;
        }
        if (!work) std::this_thread::sleep_for(std::chrono::microseconds(10));
      }
    } catch (const std::exception& e) {
      {
        std::lock_guard<std::mutex> guard(mutex_);
        error_ = e.what();
      }
      for (size_t c = 0; c < kChannels; ++c) store(header(c) + 1, 1);
    }
  }

  uint8_t* data_;
  uint64_t* control_;
  uint64_t maximum_, row_bytes_, timeout_;
  std::vector<SharedEngramTable> tables_;
  std::vector<int> fds_;
  std::atomic<bool> stopped_{false};
  std::thread thread_;
  std::vector<std::thread> readers_;
  std::mutex read_mutex_;
  std::condition_variable read_ready_, read_done_;
  std::function<void()> read_task_;
  uint64_t read_generation_ = 0;
  size_t read_remaining_ = 0;
  bool readers_stopped_ = false;
  std::mutex mutex_;
  std::string error_;
};
}  // namespace vllm::gguf
