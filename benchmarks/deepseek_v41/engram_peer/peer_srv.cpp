// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace {

constexpr uint32_t MAGIC_PING = 0x56343150u;  // "V41P"
constexpr uint32_t MAGIC_LOAD = 0x5634314Cu;  // "V41L"
constexpr uint32_t MAGIC_GET = 0x56343147u;   // "V41G"

bool read_full(int fd, void* buf, size_t n) {
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

bool write_full(int fd, const void* buf, size_t n) {
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

void pin_cpu(int cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

uint64_t fnv1a_words(uint64_t h, const uint8_t* data, size_t len) {
  const size_t words = len / 8;
  const uint64_t* wp = reinterpret_cast<const uint64_t*>(data);
  for (size_t i = 0; i < words; ++i) {
    h ^= wp[i];
    h *= 1099511628211ULL;
  }
  for (size_t i = words * 8; i < len; ++i) {
    h ^= data[i];
    h *= 1099511628211ULL;
  }
  return h;
}

class GatherPool {
 public:
  explicit GatherPool(int nthreads = 8) {
    for (int i = 0; i < nthreads; ++i) {
      // CPUs 24..31: even on NUMA 0, odd on NUMA 1, disjoint from vLLM
      // GPU/Engine cores
      threads_.emplace_back([this, i] { worker(i, 24 + i); });
    }
  }
  ~GatherPool() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_start_.notify_all();
    for (auto& t : threads_)
      if (t.joinable()) t.join();
  }

  void run(std::function<void(int, int)> fn) {
    std::unique_lock<std::mutex> lk(call_mu_);
    {
      std::lock_guard<std::mutex> g(mu_);
      fn_ = std::move(fn);
      rem_ = int(threads_.size());
      ++gen_;
    }
    cv_start_.notify_all();
    std::unique_lock<std::mutex> g(mu_);
    cv_done_.wait(g, [&] { return rem_ == 0; });
    fn_ = nullptr;
  }

 private:
  void worker(int tid, int cpu) {
    pin_cpu(cpu);
    uint64_t seen = 0;
    const int nt = 8;
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
      cv_start_.wait(lk, [&] { return stop_ || gen_ != seen; });
      if (stop_) return;
      seen = gen_;
      auto f = fn_;
      lk.unlock();
      f(tid, nt);
      lk.lock();
      if (--rem_ == 0) cv_done_.notify_one();
    }
  }

  std::vector<std::thread> threads_;
  std::mutex call_mu_;
  std::mutex mu_;
  std::condition_variable cv_start_, cv_done_;
  std::function<void(int, int)> fn_;
  uint64_t gen_ = 0;
  int rem_ = 0;
  bool stop_ = false;
};

struct PeerState {
  std::mutex load_mu;
  uint8_t* mem = nullptr;
  size_t mem_bytes = 0;
  std::atomic<bool> ready{false};
  uint64_t split_row = 0;
  uint64_t tail_rows = 0;
  uint64_t row_bytes = 0;
  uint64_t hash = 0;
  GatherPool pool{8};
};

void tune_socket(int fd) {
  int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  int buf = 32 * 1024 * 1024;
  setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buf, sizeof(buf));
  setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf, sizeof(buf));
  setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof(one));
}

void handle_client(int fd, PeerState* st) {
  tune_socket(fd);
  std::vector<uint32_t> req_rows;
  std::vector<uint8_t> out_buf;
  req_rows.reserve(65536);
  out_buf.reserve(65536 * 264);

  for (;;) {
    uint32_t hdr[2];
    if (!read_full(fd, hdr, sizeof(hdr))) break;
    const uint32_t magic = hdr[0];
    const uint32_t arg = hdr[1];

    if (magic == MAGIC_PING) {
      struct __attribute__((packed)) PingReply {
        uint32_t magic;
        uint32_t ready;
        uint64_t split_row;
        uint64_t tail_rows;
        uint64_t row_bytes;
        uint64_t hash;
      } rep{
          MAGIC_PING,    st->ready.load(std::memory_order_acquire) ? 1u : 0u,
          st->split_row, st->tail_rows,
          st->row_bytes, st->hash,
      };
      if (!write_full(fd, &rep, sizeof(rep))) break;
    } else if (magic == MAGIC_LOAD) {
      uint64_t meta[3];
      if (!read_full(fd, meta, sizeof(meta))) break;
      const uint64_t split_row = meta[0];
      const uint64_t tail_rows = meta[1];
      const uint64_t row_bytes = meta[2];
      const size_t total_bytes = size_t(tail_rows) * size_t(row_bytes);
      if (row_bytes == 0 || tail_rows == 0 ||
          total_bytes / row_bytes != tail_rows)
        break;

      std::lock_guard<std::mutex> lk(st->load_mu);
      st->ready.store(false, std::memory_order_release);
      if (st->mem && st->mem_bytes != total_bytes) {
        munmap(st->mem, st->mem_bytes);
        st->mem = nullptr;
        st->mem_bytes = 0;
      }
      if (!st->mem) {
        void* p = mmap(nullptr, total_bytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) {
          std::perror("mmap failed");
          break;
        }
#ifdef MADV_HUGEPAGE
        madvise(p, total_bytes, MADV_HUGEPAGE);
#endif
        st->mem = static_cast<uint8_t*>(p);
        st->mem_bytes = total_bytes;
        // First-touch 16 MiB blocks interleaved across NUMA 0 (even) and NUMA 1
        // (odd)
        constexpr size_t kBlock = 16ULL * 1024 * 1024;
        const size_t nblocks = (total_bytes + kBlock - 1) / kBlock;
        st->pool.run([&](int tid, int nt) {
          // tid % 2 == 0 is on NUMA 0, tid % 2 == 1 is on NUMA 1!
          // Assign block b to threads with (b % 2) == (tid % 2).
          for (size_t b = tid; b < nblocks; b += nt) {
            const size_t off = b * kBlock;
            const size_t len = std::min(kBlock, total_bytes - off);
            std::memset(st->mem + off, 0, len);
          }
        });
      }

      constexpr size_t kChunk = 16ULL * 1024 * 1024;
      size_t off = 0;
      uint64_t h = 14695981039346656037ULL;
      bool ok = true;
      while (off < total_bytes) {
        const size_t n = std::min(kChunk, total_bytes - off);
        if (!read_full(fd, st->mem + off, n)) {
          ok = false;
          break;
        }
        h = fnv1a_words(h, st->mem + off, n);
        off += n;
      }
      if (!ok) break;
      st->split_row = split_row;
      st->tail_rows = tail_rows;
      st->row_bytes = row_bytes;
      st->hash = h;
      st->ready.store(true, std::memory_order_release);
      std::fprintf(stderr,
                   "[peer_srv] Loaded split_row=%llu tail_rows=%llu bytes=%.2f "
                   "GiB hash=%016llx\n",
                   (unsigned long long)split_row, (unsigned long long)tail_rows,
                   double(total_bytes) / (1024.0 * 1024.0 * 1024.0),
                   (unsigned long long)h);

      struct __attribute__((packed)) LoadReply {
        uint32_t magic;
        uint32_t status;
        uint64_t hash;
      } rep{MAGIC_LOAD, 1u, h};
      if (!write_full(fd, &rep, sizeof(rep))) break;
    } else if (magic == MAGIC_GET) {
      const uint32_t count = arg;
      if (!st->ready.load(std::memory_order_acquire) || count == 0 ||
          count > 524288)
        break;
      req_rows.resize(count);
      if (!read_full(fd, req_rows.data(), size_t(count) * sizeof(uint32_t)))
        break;
      const size_t rb = size_t(st->row_bytes);
      const uint64_t max_rel = st->tail_rows;
      const size_t out_bytes = 8 + size_t(count) * rb;
      if (out_buf.size() < out_bytes) out_buf.resize(out_bytes);

      uint32_t* rhdr = reinterpret_cast<uint32_t*>(out_buf.data());
      rhdr[0] = MAGIC_GET;
      rhdr[1] = count;
      uint8_t* dst_base = out_buf.data() + 8;
      const uint8_t* src_base = st->mem;

      std::atomic<bool> bad{false};
      if (count < 128) {
        const uint32_t* rr = req_rows.data();
        for (uint32_t i = 0; i < count; ++i) {
          if (i + 8 < count) {
            const uint32_t pr = rr[i + 8];
            if (pr < max_rel) {
              const uint8_t* pf = src_base + size_t(pr) * rb;
              __builtin_prefetch(pf, 0, 1);
              __builtin_prefetch(pf + 64, 0, 1);
              __builtin_prefetch(pf + 128, 0, 1);
              __builtin_prefetch(pf + 192, 0, 1);
            }
          }
          const uint32_t r = rr[i];
          if (__builtin_expect(r >= max_rel, 0)) {
            bad.store(true, std::memory_order_relaxed);
            break;
          }
          std::memcpy(dst_base + size_t(i) * rb, src_base + size_t(r) * rb, rb);
        }
      } else {
        const uint32_t* rr = req_rows.data();
        st->pool.run([&](int tid, int nt) {
          const uint32_t begin =
              (uint64_t(count) * uint64_t(tid)) / uint64_t(nt);
          const uint32_t end =
              (uint64_t(count) * uint64_t(tid + 1)) / uint64_t(nt);
          for (uint32_t i = begin; i < end; ++i) {
            if (i + 8 < end) {
              const uint32_t pr = rr[i + 8];
              if (pr < max_rel) {
                const uint8_t* pf = src_base + size_t(pr) * rb;
                __builtin_prefetch(pf, 0, 1);
                __builtin_prefetch(pf + 64, 0, 1);
                __builtin_prefetch(pf + 128, 0, 1);
                __builtin_prefetch(pf + 192, 0, 1);
              }
            }
            const uint32_t r = rr[i];
            if (__builtin_expect(r >= max_rel, 0)) {
              bad.store(true, std::memory_order_relaxed);
              break;
            }
            std::memcpy(dst_base + size_t(i) * rb, src_base + size_t(r) * rb,
                        rb);
          }
        });
      }
      if (bad.load(std::memory_order_relaxed)) break;
      if (!write_full(fd, out_buf.data(), out_bytes)) break;
    } else {
      break;
    }
  }
  ::close(fd);
}

}  // namespace

int main(int argc, char** argv) {
  signal(SIGPIPE, SIG_IGN);
  int port = (argc > 1) ? std::atoi(argv[1]) : 29541;
  int lfd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (lfd < 0) {
    std::perror("socket");
    return 1;
  }
  int one = 1;
  setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(uint16_t(port));
  if (::bind(lfd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    std::perror("bind");
    return 1;
  }
  if (::listen(lfd, 64) < 0) {
    std::perror("listen");
    return 1;
  }
  std::fprintf(stderr, "[peer_srv] Listening on 0.0.0.0:%d\n", port);
  PeerState state;
  for (;;) {
    int cfd = ::accept(lfd, nullptr, nullptr);
    if (cfd < 0) {
      if (errno == EINTR) continue;
      std::perror("accept");
      continue;
    }
    std::thread([cfd, &state] { handle_client(cfd, &state); }).detach();
  }
}
