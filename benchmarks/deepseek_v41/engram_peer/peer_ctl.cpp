// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <arpa/inet.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr uint32_t MAGIC_PING = 0x56343150u;
constexpr uint32_t MAGIC_LOAD = 0x5634314Cu;

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

bool pread_full(int fd, void* buf, size_t n, uint64_t off) {
  uint8_t* p = static_cast<uint8_t*>(buf);
  while (n > 0) {
    ssize_t r = ::pread(fd, p, n, off_t(off));
    if (r <= 0) {
      if (r < 0 && errno == EINTR) continue;
      return false;
    }
    p += r;
    n -= size_t(r);
    off += uint64_t(r);
  }
  return true;
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

int connect_peer(const char* host, int port) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;
  int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  int buf = 32 * 1024 * 1024;
  setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buf, sizeof(buf));
  setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf, sizeof(buf));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(uint16_t(port));
  if (::inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
    ::close(fd);
    return -1;
  }
  if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    ::close(fd);
    return -1;
  }
  return fd;
}

struct __attribute__((packed)) PingReply {
  uint32_t magic;
  uint32_t ready;
  uint64_t split_row;
  uint64_t tail_rows;
  uint64_t row_bytes;
  uint64_t hash;
};

bool do_ping(const char* host, int port, PingReply* out) {
  int fd = connect_peer(host, port);
  if (fd < 0) return false;
  uint32_t hdr[2] = {MAGIC_PING, 0};
  bool ok = write_full(fd, hdr, sizeof(hdr)) &&
            read_full(fd, out, sizeof(*out)) && out->magic == MAGIC_PING;
  ::close(fd);
  return ok;
}

void pin_cpu(int cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

void drop_safetensors_in_dir(const std::string& dir) {
  DIR* d = ::opendir(dir.c_str());
  if (!d) return;
  while (dirent* e = ::readdir(d)) {
    std::string name = e->d_name;
    if (name.size() > 12 && name.substr(name.size() - 12) == ".safetensors") {
      std::string full = dir + "/" + name;
      int fd = ::open(full.c_str(), O_RDONLY);
      if (fd >= 0) {
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        ::close(fd);
      }
    }
  }
  ::closedir(d);
}

}  // namespace

int main(int argc, char** argv) {
  signal(SIGPIPE, SIG_IGN);
  if (argc < 2) {
    std::fprintf(stderr, "Usage: peer_ctl <ping|load|warm_local> ...\n");
    return 1;
  }
  std::string cmd = argv[1];

  if (cmd == "ping") {
    if (argc < 4) return 1;
    PingReply rep{};
    if (!do_ping(argv[2], std::atoi(argv[3]), &rep)) {
      std::printf("OFFLINE\n");
      return 1;
    }
    std::printf(
        "ready=%u split_row=%llu tail_rows=%llu row_bytes=%llu hash=%016llx\n",
        rep.ready, (unsigned long long)rep.split_row,
        (unsigned long long)rep.tail_rows, (unsigned long long)rep.row_bytes,
        (unsigned long long)rep.hash);
    return rep.ready == 1 ? 0 : 2;
  }

  if (cmd == "load") {
    // peer_ctl load <host> <port> <rows_path> <split_row> <row_bytes>
    if (argc < 7) return 1;
    const char* host = argv[2];
    const int port = std::atoi(argv[3]);
    const char* path = argv[4];
    const uint64_t split_row = std::strtoull(argv[5], nullptr, 10);
    const uint64_t row_bytes = std::strtoull(argv[6], nullptr, 10);

    int ffd = ::open(path, O_RDONLY);
    if (ffd < 0) {
      std::perror("open rows_path");
      return 1;
    }
    struct stat st{};
    if (::fstat(ffd, &st) < 0) {
      std::perror("fstat");
      ::close(ffd);
      return 1;
    }
    const uint64_t total_rows = uint64_t(st.st_size) / row_bytes;
    if (split_row >= total_rows) {
      std::fprintf(stderr, "split_row >= total_rows\n");
      ::close(ffd);
      return 1;
    }
    const uint64_t tail_rows = total_rows - split_row;

    PingReply ping{};
    if (do_ping(host, port, &ping) && ping.ready == 1 &&
        ping.split_row == split_row && ping.tail_rows == tail_rows &&
        ping.row_bytes == row_bytes) {
      std::printf(
          "[peer_ctl] %s:%d ALREADY LOADED split_row=%llu tail_rows=%llu "
          "hash=%016llx\n",
          host, port, (unsigned long long)split_row,
          (unsigned long long)tail_rows, (unsigned long long)ping.hash);
      // Drop remote tail from local page cache too
      posix_fadvise(ffd, off_t(split_row * row_bytes), 0, POSIX_FADV_DONTNEED);
      ::close(ffd);
      return 0;
    }

    int sfd = connect_peer(host, port);
    if (sfd < 0) {
      std::perror("connect_peer");
      ::close(ffd);
      return 1;
    }

    auto t0 = std::chrono::steady_clock::now();
    struct __attribute__((packed)) LoadReq {
      uint32_t magic;
      uint32_t arg;
      uint64_t split_row;
      uint64_t tail_rows;
      uint64_t row_bytes;
    } req{MAGIC_LOAD, 0u, split_row, tail_rows, row_bytes};
    if (!write_full(sfd, &req, sizeof(req))) {
      std::fprintf(stderr, "write LoadReq failed\n");
      return 1;
    }

    const uint64_t start_off = split_row * row_bytes;
    const uint64_t total_bytes = tail_rows * row_bytes;
    posix_fadvise(ffd, off_t(start_off), off_t(total_bytes),
                  POSIX_FADV_SEQUENTIAL);
    constexpr size_t kChunk = 16ULL * 1024 * 1024;
    std::vector<uint8_t> buf(kChunk);
    uint64_t sent = 0;
    uint64_t h = 14695981039346656037ULL;
    while (sent < total_bytes) {
      const size_t n = std::min<uint64_t>(kChunk, total_bytes - sent);
      if (!pread_full(ffd, buf.data(), n, start_off + sent)) {
        std::fprintf(stderr, "pread failed\n");
        return 1;
      }
      // Drop each streamed chunk from local page cache immediately!
      posix_fadvise(ffd, off_t(start_off + sent), off_t(n),
                    POSIX_FADV_DONTNEED);
      h = fnv1a_words(h, buf.data(), n);
      if (!write_full(sfd, buf.data(), n)) {
        std::fprintf(stderr, "socket write failed\n");
        return 1;
      }
      sent += n;
    }
    posix_fadvise(ffd, off_t(start_off), 0, POSIX_FADV_DONTNEED);
    ::close(ffd);

    struct __attribute__((packed)) LoadReply {
      uint32_t magic;
      uint32_t status;
      uint64_t hash;
    } rep{};
    if (!read_full(sfd, &rep, sizeof(rep)) || rep.magic != MAGIC_LOAD ||
        rep.status != 1 || rep.hash != h) {
      std::fprintf(
          stderr,
          "LoadReply mismatch: status=%u srv_hash=%016llx cli_hash=%016llx\n",
          rep.status, (unsigned long long)rep.hash, (unsigned long long)h);
      ::close(sfd);
      return 1;
    }
    ::close(sfd);
    double sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    double gib = double(total_bytes) / (1024.0 * 1024.0 * 1024.0);
    std::printf(
        "[peer_ctl] Uploaded %.2f GiB to %s:%d in %.2f s (%.0f MB/s) "
        "hash=%016llx OK!\n",
        gib, host, port, sec, (gib * 1024.0) / sec, (unsigned long long)h);
    return 0;
  }

  if (cmd == "warm_local") {
    // peer_ctl warm_local <rows_path> <split_row> <row_bytes> <model_dir>
    // [dspark_dir]
    if (argc < 6) return 1;
    const char* path = argv[2];
    const uint64_t split_row = std::strtoull(argv[3], nullptr, 10);
    const uint64_t row_bytes = std::strtoull(argv[4], nullptr, 10);
    drop_safetensors_in_dir(argv[5]);
    if (argc >= 7) drop_safetensors_in_dir(argv[6]);

    int ffd = ::open(path, O_RDONLY);
    if (ffd < 0) {
      std::perror("open rows_path");
      return 1;
    }
    struct stat st{};
    if (::fstat(ffd, &st) < 0) {
      ::close(ffd);
      return 1;
    }
    const uint64_t keep_bytes =
        std::min<uint64_t>(uint64_t(st.st_size), split_row * row_bytes);
    if (uint64_t(st.st_size) > keep_bytes) {
      posix_fadvise(ffd, off_t(keep_bytes), 0, POSIX_FADV_DONTNEED);
    }
    posix_fadvise(ffd, 0, off_t(keep_bytes), POSIX_FADV_SEQUENTIAL);

    auto t0 = std::chrono::steady_clock::now();
    constexpr size_t kChunk = 16ULL * 1024 * 1024;
    const size_t nchunks = (keep_bytes + kChunk - 1) / kChunk;

    auto worker = [&](int parity, int cpu) {
      pin_cpu(cpu);
      std::vector<uint8_t> buf(kChunk);
      volatile uint8_t sink = 0;
      for (size_t c = size_t(parity); c < nchunks; c += 2) {
        const uint64_t off = uint64_t(c) * kChunk;
        const size_t n = std::min<uint64_t>(kChunk, keep_bytes - off);
        if (pread_full(ffd, buf.data(), n, off)) {
          sink ^= buf[0];
        }
      }
      (void)sink;
    };
    // CPU 24 is on NUMA 0 (even), CPU 25 is on NUMA 1 (odd)!
    std::thread t_numa0(worker, 0, 24);
    std::thread t_numa1(worker, 1, 25);
    t_numa0.join();
    t_numa1.join();

    posix_fadvise(ffd, 0, off_t(keep_bytes), POSIX_FADV_NORMAL);
    if (uint64_t(st.st_size) > keep_bytes) {
      posix_fadvise(ffd, off_t(keep_bytes), 0, POSIX_FADV_DONTNEED);
    }
    ::close(ffd);
    double sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    double gib = double(keep_bytes) / (1024.0 * 1024.0 * 1024.0);
    std::printf(
        "[peer_ctl] Warmed %.2f GiB local RAM (NUMA 0+1 interleaved) in %.2f "
        "s\n",
        gib, sec);
    return 0;
  }

  return 1;
}
