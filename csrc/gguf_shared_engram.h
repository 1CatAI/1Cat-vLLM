// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <sched.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#if defined(__x86_64__) || defined(_M_X64)
  #include <immintrin.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace vllm::gguf {

struct SharedEngramTable {
  int fd;
  uint64_t offset, rows;
};

class SharedEngramReader {
 public:
  static constexpr uint64_t kChannels = 2, kRanks = 4, kHeader = 8;
  static constexpr uint32_t kPeerMagicGet = 0x56343147u;  // "V41G"

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
        !maximum || maximum > 16384 * 24 ||
        (row_bytes != 144 && row_bytes != 264) || tables_.size() != kChannels ||
        !timeout_ || bytes < kChannels * maximum * row_bytes)
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

      if (const char* sr = std::getenv("VLLM_V41_ENGRAM_SPLIT_ROW")) {
        split_row_ = std::strtoull(sr, nullptr, 10);
      }
      if (const char* peer = std::getenv("VLLM_V41_ENGRAM_PEER")) {
        std::string ps(peer);
        const auto colon = ps.find(':');
        if (colon != std::string::npos && split_row_ > 0) {
          peer_host_ = ps.substr(0, colon);
          peer_port_ = std::atoi(ps.substr(colon + 1).c_str());
          if (!peer_host_.empty() && peer_port_ > 0) {
            peer_enabled_ = true;
          }
        }
      }

      pairs_.resize(maximum);
      local_unique_.reserve(maximum);
      remote_unique_.reserve(maximum);
      duplicate_.reserve(maximum);
      if (peer_enabled_) {
        remote_req_buf_.reserve(2 + maximum);
        remote_resp_buf_.reserve(8 + maximum * row_bytes);
      }

      // When 100% of rows live in local RAM + IB peer RAM, 24 NUMA-interleaved
      // threads on dedicated cores hide local pread (<6 ms) under Layer 0 GPU!
      size_t readers = peer_enabled_ ? 24 : 32;
      if (const char* v = std::getenv("VLLM_V41_ENGRAM_READERS"))
        readers = std::max(1, std::atoi(v));
      for (size_t i = 0; i < readers; ++i)
        readers_.emplace_back([this, i] { read_worker(i); });
      thread_ = std::thread([this] { run(); });
    } catch (...) {
      stop_readers();
      close_peer();
      for (int fd : fds_) close(fd);
      throw;
    }
  }

  ~SharedEngramReader() {
    stop();
    close_peer();
    for (int fd : fds_) close(fd);
  }

  void stop() {
    stopped_.store(true, std::memory_order_release);
    if (thread_.joinable()) thread_.join();
    stop_readers();
    close_peer();
    for (size_t c = 0; c < kChannels; ++c) store(header(c) + 1, 1);
  }

  std::string error() {
    std::lock_guard<std::mutex> guard(mutex_);
    return error_;
  }

 private:
  struct RowOut {
    uint64_t row;
    uint32_t out;
  };

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

  void close_peer() {
    if (peer_fd_ >= 0) {
      ::close(peer_fd_);
      peer_fd_ = -1;
    }
  }

  bool ensure_peer_connected() {
    if (peer_fd_ >= 0) return true;
    int fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) return false;
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    int buf = 32 * 1024 * 1024;
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buf, sizeof(buf));
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf, sizeof(buf));
    setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof(one));
    timeval tv{5, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(uint16_t(peer_port_));
    if (::inet_pton(AF_INET, peer_host_.c_str(), &addr.sin_addr) != 1 ||
        ::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
      ::close(fd);
      return false;
    }
    peer_fd_ = fd;
    return true;
  }

  static bool sock_write_all(int fd, const void* buf, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(buf);
    while (n > 0) {
      ssize_t w = ::send(fd, p, n, MSG_NOSIGNAL);
      if (w <= 0) {
        if (w < 0 && errno == EINTR) continue;
        return false;
      }
      p += w;
      n -= size_t(w);
    }
    return true;
  }

  static bool sock_read_all(int fd, void* buf, size_t n) {
    uint8_t* p = static_cast<uint8_t*>(buf);
    while (n > 0) {
      ssize_t r = ::recv(fd, p, n, 0);
      if (r <= 0) {
        if (r < 0 && errno == EINTR) continue;
        return false;
      }
      p += r;
      n -= size_t(r);
    }
    return true;
  }

  bool peer_send_request(const uint32_t* req_words, size_t nwords) {
    for (int attempt = 0; attempt < 2; ++attempt) {
      if (!ensure_peer_connected()) return false;
      if (sock_write_all(peer_fd_, req_words, nwords * sizeof(uint32_t)))
        return true;
      close_peer();
    }
    return false;
  }

  bool peer_recv_rows(uint32_t expected_count, uint8_t* raw) {
    const size_t total = 8 + size_t(expected_count) * row_bytes_;
    if (remote_resp_buf_.size() < total) remote_resp_buf_.resize(total);
    if (!sock_read_all(peer_fd_, remote_resp_buf_.data(), total)) {
      close_peer();
      return false;
    }
    const uint32_t* rhdr =
        reinterpret_cast<const uint32_t*>(remote_resp_buf_.data());
    if (rhdr[0] != kPeerMagicGet || rhdr[1] != expected_count) {
      close_peer();
      return false;
    }
    const uint8_t* src = remote_resp_buf_.data() + 8;
    for (uint32_t i = 0; i < expected_count; ++i) {
      std::memcpy(raw + size_t(remote_unique_[i].out) * row_bytes_,
                  src + size_t(i) * row_bytes_, row_bytes_);
    }
    return true;
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

  void read_worker(size_t idx) {
    if (peer_enabled_) {
      // Dedicated CPUs avoiding GPU workers (2,4,3,5) and EngineCore
      // (20,22,60,62)
      static constexpr int kSafeCpus[48] = {
          8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 24, 25, 26, 27,
          28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51,
          52, 53, 54, 55, 56, 57, 58, 59, 64, 65, 66, 67, 68, 69, 70, 71};
      const int ncpus = int(sysconf(_SC_NPROCESSORS_ONLN));
      const int target = kSafeCpus[idx % 48];
      if (target < ncpus) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(target, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
      }
    }
    uint64_t completed = 0;
    std::unique_lock<std::mutex> guard(read_mutex_);
    for (;;) {
      read_ready_.wait(guard, [&] {
        return readers_stopped_ || read_generation_ != completed;
      });
      if (readers_stopped_) return;
      completed = read_generation_;
      guard.unlock();
      read_task_();
      guard.lock();
      if (!--read_remaining_) read_done_.notify_one();
    }
  }

  bool pread_row(int fd, uint64_t offset, uint8_t* target) const {
    uint64_t left = row_bytes_;
    while (left) {
      const auto got = pread(fd, target, left, off_t(offset));
      if (got <= 0) {
        if (got < 0 && errno == EINTR) continue;
        return false;
      }
      left -= uint64_t(got);
      offset += uint64_t(got);
      target += got;
    }
    return true;
  }

  void pread_batch(size_t channel, const std::vector<RowOut>& items,
                   uint8_t* raw, size_t inline_limit) {
    if (items.empty()) return;
    if (items.size() <= inline_limit) {
      for (const auto& it : items) {
        if (!pread_row(fds_[channel],
                       tables_[channel].offset + it.row * row_bytes_,
                       raw + size_t(it.out) * row_bytes_))
          throw std::runtime_error(
              "shared Engram source returned a short read");
      }
      return;
    }
    std::atomic<size_t> next{0};
    std::atomic<bool> failed{false};
    constexpr size_t kStep = 16;
    {
      std::lock_guard<std::mutex> guard(read_mutex_);
      read_task_ = [&] {
        for (;;) {
          const size_t base = next.fetch_add(kStep, std::memory_order_relaxed);
          if (base >= items.size() || failed.load(std::memory_order_acquire))
            break;
          const size_t end = std::min(items.size(), base + kStep);
          for (size_t i = base; i < end; ++i) {
            const auto& it = items[i];
            if (!pread_row(fds_[channel],
                           tables_[channel].offset + it.row * row_bytes_,
                           raw + size_t(it.out) * row_bytes_)) {
              failed.store(true, std::memory_order_release);
              return;
            }
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
  }

  void read_rows(size_t channel, uint64_t count, uint64_t begin) {
    auto* raw = data_ + channel * maximum_ * row_bytes_;
    const auto* ids = rank(channel, 0) + 8;
    const uint64_t table_rows = tables_[channel].rows;

    for (uint64_t i = 0; i < count; ++i) {
      const uint64_t r = ids[i];
      if (__builtin_expect(r >= table_rows, 0))
        throw std::runtime_error("shared Engram row id outside table");
      pairs_[i] = RowOut{r, uint32_t(i)};
    }
    std::sort(pairs_.begin(), pairs_.begin() + count,
              [](const RowOut& a, const RowOut& b) {
                return a.row < b.row ? true
                                     : (a.row > b.row ? false : a.out < b.out);
              });

    local_unique_.clear();
    remote_unique_.clear();
    duplicate_.clear();

    const bool use_peer = peer_enabled_ && split_row_ < table_rows;
    if (use_peer) {
      remote_req_buf_.resize(2);
      remote_req_buf_[0] = kPeerMagicGet;
    }

    uint64_t prev_row = UINT64_MAX;
    uint32_t first_out = 0;
    for (uint64_t i = 0; i < count; ++i) {
      const auto p = pairs_[i];
      if (p.row == prev_row) {
        duplicate_.emplace_back(p.out, first_out);
      } else {
        prev_row = p.row;
        first_out = p.out;
        if (use_peer && p.row >= split_row_) {
          remote_unique_.push_back(p);
          remote_req_buf_.push_back(uint32_t(p.row - split_row_));
        } else {
          local_unique_.push_back(p);
        }
      }
    }

    // 1. Kick off IB peer request FIRST so remote RAM gather + 40G IB transfer
    //    runs simultaneously with local reader pool!
    bool peer_inflight = false;
    if (!remote_unique_.empty()) {
      remote_req_buf_[1] = uint32_t(remote_unique_.size());
      peer_inflight =
          peer_send_request(remote_req_buf_.data(), remote_req_buf_.size());
    }

    // 2. Launch local pread across worker pool WITHOUT blocking yet when peer
    //    recv is in flight, so thread_ drains the 40G IB socket IN PARALLEL!
    bool peer_ok = false;
    if (peer_inflight && !local_unique_.empty() && local_unique_.size() > 8) {
      std::atomic<size_t> next{0};
      std::atomic<bool> failed{false};
      constexpr size_t kStep = 16;
      {
        std::lock_guard<std::mutex> guard(read_mutex_);
        read_task_ = [&] {
          for (;;) {
            const size_t base =
                next.fetch_add(kStep, std::memory_order_relaxed);
            if (base >= local_unique_.size() ||
                failed.load(std::memory_order_acquire))
              break;
            const size_t end = std::min(local_unique_.size(), base + kStep);
            for (size_t i = base; i < end; ++i) {
              const auto& it = local_unique_[i];
              if (!pread_row(fds_[channel],
                             tables_[channel].offset + it.row * row_bytes_,
                             raw + size_t(it.out) * row_bytes_)) {
                failed.store(true, std::memory_order_release);
                return;
              }
            }
          }
        };
        read_remaining_ = readers_.size();
        ++read_generation_;
      }
      read_ready_.notify_all();

      // Drain 40G IB socket concurrently while workers pread local page cache!
      peer_ok = peer_recv_rows(uint32_t(remote_unique_.size()), raw);

      {
        std::unique_lock<std::mutex> guard(read_mutex_);
        read_done_.wait(guard, [&] { return !read_remaining_; });
        read_task_ = {};
      }
      if (failed.load(std::memory_order_acquire))
        throw std::runtime_error("shared Engram source returned a short read");
    } else {
      pread_batch(channel, local_unique_, raw, use_peer ? 32 : 4);
      if (peer_inflight) {
        peer_ok = peer_recv_rows(uint32_t(remote_unique_.size()), raw);
      }
    }

    // 3. Seamless local pread fallback if peer socket had any error.
    if (!remote_unique_.empty() && !peer_ok) {
      pread_batch(channel, remote_unique_, raw, 4);
    }

    check(begin);
    for (const auto& [to, from] : duplicate_) {
      std::memcpy(raw + size_t(to) * row_bytes_,
                  raw + size_t(from) * row_bytes_, row_bytes_);
    }
    const uint64_t total_unique = local_unique_.size() + remote_unique_.size();
    auto* h = header(channel);
    h[2] = total_unique;
    h[3] = total_unique * row_bytes_;
    h[4] = now() - begin;
    h[5] = count;
  }

  void run() {
    if (peer_enabled_) {
      // Pin main SharedEngramReader polling thread to dedicated CPU 6 (NUMA 0)
      const int ncpus = int(sysconf(_SC_NPROCESSORS_ONLN));
      if (6 < ncpus) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(6, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
      }
    }
    std::array<uint64_t, kChannels> sequence{1, 1}, started{0, 0};
    int idle_spins = 0;
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
        if (work) {
          idle_spins = 0;
        } else if (idle_spins < 4000) {
          ++idle_spins;
#if defined(__x86_64__) || defined(_M_X64)
          _mm_pause();
#endif
        } else {
          std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
      }
    } catch (const std::exception& e) {
      std::fprintf(stderr, "[SharedEngramReader] fatal error: %s\n", e.what());
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
  uint64_t split_row_ = 0;
  std::string peer_host_;
  int peer_port_ = 0;
  bool peer_enabled_ = false;
  int peer_fd_ = -1;
  std::vector<RowOut> pairs_, local_unique_, remote_unique_;
  std::vector<std::pair<uint32_t, uint32_t>> duplicate_;
  std::vector<uint32_t> remote_req_buf_;
  std::vector<uint8_t> remote_resp_buf_;
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
