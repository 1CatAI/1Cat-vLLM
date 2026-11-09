// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "csrc/gguf_shared_engram.h"

#include <cassert>
#include <iostream>

using vllm::gguf::SharedEngramReader;
using vllm::gguf::SharedEngramTable;

constexpr size_t maximum = 16, row_bytes = 144;
constexpr size_t words = 2 * (8 + 4 * (maximum + 8));

struct Fixture {
  alignas(64) std::array<uint64_t, words> control{};
  std::vector<uint8_t> data = std::vector<uint8_t>(2 * maximum * row_bytes, 99);
  std::vector<int> fds;
  std::vector<SharedEngramTable> tables;
  std::unique_ptr<SharedEngramReader> reader;

  Fixture(uint64_t timeout = 2000000000ULL) {
    for (unsigned table = 0; table < 2; ++table) {
      char path[] = "/tmp/vllm-engram-XXXXXX";
      int fd = mkstemp(path);
      assert(fd >= 0);
      unlink(path);
      // Sparse files exercise real 64-bit pread offsets without allocating
      // a multi-gigabyte test table.
      const uint64_t start = (uint64_t{1} << 32) + 416;
      assert(ftruncate(fd, off_t(start + 64 * row_bytes)) == 0);
      for (unsigned row = 0; row < 64; ++row) {
        std::array<uint8_t, row_bytes> bytes;
        for (size_t col = 0; col < row_bytes; ++col)
          bytes[col] = uint8_t(row * 7 + col + table * 113);
        assert(pwrite(fd, bytes.data(), row_bytes,
                      off_t(start + row * row_bytes)) == row_bytes);
      }
      fds.push_back(fd);
      tables.push_back({fd, start, 64});
    }
    reader = std::make_unique<SharedEngramReader>(data.data(), data.size(),
                                                  control.data(), maximum,
                                                  row_bytes, tables, timeout);
  }
  ~Fixture() {
    reader->stop();
    reader.reset();
    for (int fd : fds) close(fd);
  }
  uint64_t* header(size_t channel) {
    return control.data() +
           channel * SharedEngramReader::channel_words(maximum);
  }
  uint64_t* rank(size_t channel, size_t r) {
    return header(channel) + 8 + r * SharedEngramReader::rank_words(maximum);
  }
  static void store(uint64_t* p, uint64_t v) {
    __atomic_store_n(p, v, __ATOMIC_RELEASE);
  }
  static uint64_t load(uint64_t* p) {
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
  }
  template <class Predicate>
  void wait(Predicate p) {
    const auto begin = std::chrono::steady_clock::now();
    while (!p()) {
      assert(std::chrono::steady_clock::now() - begin <
             std::chrono::seconds(5));
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
  }
  void publish(size_t channel, unsigned r, uint64_t sequence,
               const std::vector<uint64_t>& ids) {
    auto* request = rank(channel, r);
    request[2] = ids.size();
    std::copy(ids.begin(), ids.end(), request + 8);
    store(request, sequence);
  }
  void check(size_t channel, const std::vector<uint64_t>& ids) {
    for (size_t i = 0; i < ids.size(); ++i)
      for (size_t col = 0; col < row_bytes; ++col)
        assert(data[(channel * maximum + i) * row_bytes + col] ==
               uint8_t(ids[i] * 7 + col + channel * 113));
  }
};

int main(int argc, char** argv) {
  assert(argc == 2);
  const std::string test = argv[1];
  if (test == "rows_and_consumption") {
    Fixture f;
    const std::vector<uint64_t> ids{7, 31, 7, 0, 31, 63, 2, 63};
    for (size_t c = 0; c < 2; ++c) {
      for (unsigned r : {2, 0, 3, 1}) f.publish(c, r, 1, ids);
      f.wait([&] { return Fixture::load(f.header(c)) == 1; });
      f.check(c, ids);
      assert(f.header(c)[2] == 5 && f.header(c)[3] == 5 * row_bytes);
      // Three ranks have consumed the buffer. The slow fourth GPU still
      // reads old rows while the next request arrives on the other ranks.
      for (unsigned r = 0; r < 3; ++r) {
        Fixture::store(f.rank(c, r) + 1, 1);
        f.publish(c, r, 2, {4, 5});
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      assert(Fixture::load(f.header(c)) == 1);
      f.check(c, ids);
      Fixture::store(f.rank(c, 3) + 1, 1);
      f.publish(c, 3, 2, {4, 5});
      f.wait([&] { return Fixture::load(f.header(c)) == 2; });
      f.check(c, {4, 5});
      for (unsigned r = 0; r < 4; ++r) Fixture::store(f.rank(c, r) + 1, 2);
    }
    assert(f.reader->error().empty());
  } else if (test == "repeated_requests") {
    Fixture f;
    for (uint64_t sequence = 1; sequence <= 128; ++sequence) {
      for (size_t c = 0; c < 2; ++c) {
        const auto row = sequence % 64;
        const std::vector<uint64_t> ids{row, 63 - row, row, 7, 7};
        for (unsigned r : {2, 0, 3, 1}) f.publish(c, r, sequence, ids);
        f.wait([&] { return Fixture::load(f.header(c)) == sequence; });
        f.check(c, ids);
        for (unsigned r = 0; r < 4; ++r)
          Fixture::store(f.rank(c, r) + 1, sequence);
      }
    }
    assert(f.reader->error().empty());
  } else if (test == "short_read") {
    Fixture f;
    assert(ftruncate(f.fds[0], off_t(f.tables[0].offset + row_bytes)) == 0);
    for (unsigned r = 0; r < 4; ++r) f.publish(0, r, 1, {63});
    f.wait([&] { return Fixture::load(f.header(0) + 1) != 0; });
    assert(f.reader->error().find("short read") != std::string::npos);
    assert(Fixture::load(f.header(0)) == 0);
  } else if (test == "timeout") {
    Fixture f(100000000);
    f.publish(0, 0, 1, {1, 2});
    f.wait([&] { return Fixture::load(f.header(0) + 1) != 0; });
    assert(f.reader->error().find("timed out") != std::string::npos);
  } else if (test == "diverged_rows" || test == "out_of_bounds") {
    Fixture f;
    for (unsigned r = 0; r < 4; ++r) {
      const auto id = test == "out_of_bounds" ? 64 : (r == 3 ? 2 : 1);
      f.publish(0, r, 1, {uint64_t(id)});
    }
    f.wait([&] { return Fixture::load(f.header(0) + 1) != 0; });
    assert(!f.reader->error().empty());
    assert(Fixture::load(f.header(0)) == 0);
  } else if (test == "stop") {
    Fixture f;
    f.publish(1, 0, 1, {1, 2});
    f.reader->stop();
    assert(Fixture::load(f.header(1) + 1) != 0);
  } else {
    return 2;
  }
  std::cout << test << " passed\n";
}
