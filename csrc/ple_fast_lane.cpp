// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Native PLE request lane for CPU-owned packed IQ4_NL n-gram tables.
//
// TP0's model stream publishes each request (token IDs, query offsets and
// n-gram context) into a host-mapped buffer and releases a sequence number.
// One native thread observes it, hashes the n-gram rows, serves packed rows
// from a set-associative cache or the mapped table, writes every TP rank's
// pinned result buffer and releases their completion flags. No Python, GIL,
// socket or CUDA call is on this path. Results are byte-identical to the
// Python owner's packed result (PackedGGUFRowReader.lookup_packed_iq4nl).

#include <torch/all.h>

#include <c10/util/Half.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef __linux__
  #include <sys/mman.h>
  #include <sys/prctl.h>
  #include <time.h>
  #include <unistd.h>
#endif
#if defined(__x86_64__) || defined(__i386__)
  #include <immintrin.h>
#endif

namespace {

constexpr int64_t kHeader = 8;
constexpr int64_t kHeads = 16;
constexpr int64_t kBlockBytes = 18;
constexpr int64_t kBlockValues = 32;
constexpr int64_t kWays = 8;
constexpr float kFp16Max = 65504.0f;
constexpr float kSafeScale = 512.0f;
constexpr float kIq4nlValues[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                    1,    13,   25,  38,  53,  69,  89,  113};

enum Stat : int64_t {
  kRequests = 0,
  kRows,
  kHits,
  kMisses,
  kTotalNs,
  kMaxNs,
  kError,
  kLastSeq,
  kHashNs,
  kLookupNs,
  kResetWaitNs,
  kPrefetchPositions,
  kPrefetchPages,
  kColdPages,
  kWindowChecked,
  kWindowMatched,
  kStats
};

enum Error : int64_t {
  kOk = 0,
  kBadRequest = 1,
  kNonfinite = 2,
  kOverflow = 3,
};

inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
  _mm_pause();
#endif
}

inline int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Floor modulo of the signed reinterpretation, as Python computes it.
inline int64_t python_mod(uint64_t mixed, int64_t size) {
  int64_t value = static_cast<int64_t>(mixed);
  int64_t remainder = value % size;
  return remainder < 0 ? remainder + size : remainder;
}

// 0: valid for FP16 output, otherwise the Error explaining the rejection.
int64_t validate_row(const uint8_t* row, int64_t row_bytes) {
  for (int64_t block = 0; block < row_bytes / kBlockBytes; ++block) {
    const uint8_t* bytes = row + block * kBlockBytes;
    uint16_t bits;
    std::memcpy(&bits, bytes, sizeof(bits));
    if ((bits & 0x7C00u) == 0x7C00u) return kNonfinite;
    float scale = static_cast<float>(c10::Half(bits, c10::Half::from_bits()));
    if (std::fabs(scale) <= kSafeScale) continue;
    // Unusual scale: apply the decoder's value-dependent FP32 check.
    for (int64_t i = 0; i < 16; ++i) {
      uint8_t packed = bytes[2 + i];
      for (uint8_t code : {static_cast<uint8_t>(packed & 15),
                           static_cast<uint8_t>(packed >> 4)}) {
        if (std::fabs(scale * kIq4nlValues[code]) > kFp16Max) return kOverflow;
      }
    }
  }
  return kOk;
}

class RowCache {
 public:
  RowCache(int64_t capacity_rows, int64_t row_bytes) : row_bytes_(row_bytes) {
    sets_ = capacity_rows / kWays;
    if (sets_ <= 0) return;
    tags_.assign(sets_ * kWays, -1);
    stamps_.assign(sets_ * kWays, 0);
    data_.reset(new uint8_t[sets_ * kWays * row_bytes]);
  }

  bool enabled() const { return sets_ > 0; }

  const uint8_t* find(int64_t row) {
    int64_t base = set_of(row) * kWays;
    for (int64_t way = 0; way < kWays; ++way) {
      if (tags_[base + way] == row) {
        stamps_[base + way] = ++clock_;
        return data_.get() + (base + way) * row_bytes_;
      }
    }
    return nullptr;
  }

  bool contains(int64_t row) const {
    int64_t base = set_of(row) * kWays;
    for (int64_t way = 0; way < kWays; ++way) {
      if (tags_[base + way] == row) return true;
    }
    return false;
  }

  void insert(int64_t row, const uint8_t* bytes) {
    int64_t base = set_of(row) * kWays;
    int64_t victim = base;
    for (int64_t way = 0; way < kWays; ++way) {
      if (tags_[base + way] < 0) {
        victim = base + way;
        break;
      }
      if (stamps_[base + way] < stamps_[victim]) victim = base + way;
    }
    tags_[victim] = row;
    stamps_[victim] = ++clock_;
    std::memcpy(data_.get() + victim * row_bytes_, bytes, row_bytes_);
  }

 private:
  int64_t set_of(int64_t row) const {
    uint64_t mixed = static_cast<uint64_t>(row) * 0x9E3779B97F4A7C15ull;
    return static_cast<int64_t>((mixed >> 17) % static_cast<uint64_t>(sets_));
  }

  int64_t row_bytes_;
  int64_t sets_ = 0;
  uint64_t clock_ = 0;
  std::vector<int64_t> tags_;
  std::vector<uint64_t> stamps_;
  std::unique_ptr<uint8_t[]> data_;
};

struct LaneConfig {
  // The lane owns references to the shared buffers it reads, so their
  // storage outlives every pointer below regardless of the caller.
  torch::Tensor request_owner;
  torch::Tensor ring_owner;
  int64_t* request;  // host-mapped request buffer (int64)
  int64_t max_reqs;
  int64_t max_tokens;
  const uint8_t* table;
  int64_t row_bytes;
  int64_t num_rows;
  uint64_t multipliers[3];
  int64_t sizes[kHeads];
  int64_t offsets[kHeads];
  int64_t eos;
  std::vector<uint8_t*> results;
  std::vector<int32_t*> flags;
  int64_t idle_spin_us;
  bool clamp_tokens;
  // Optional draft ring: header[8] | context+bonus [reqs, 3] | drafts [reqs, K]
  const int64_t* ring = nullptr;
  int64_t ring_reqs = 0;
  int64_t ring_spec = 0;
};

class Lane {
 public:
  Lane(LaneConfig config, int64_t cache_rows)
      : config_(std::move(config)), cache_(cache_rows, config_.row_bytes) {
    rows_.resize(config_.max_tokens * kHeads);
    misses_.reserve(config_.max_tokens * kHeads);
    for (auto& value : stats_) value.store(0);
    last_seq_ = load_acquire(&config_.request[0]);
    thread_ = std::thread([this] { run(); });
  }

  ~Lane() {
    stop_.store(true, std::memory_order_release);
    if (thread_.joinable()) thread_.join();
  }

  void stats(int64_t* out) const {
    for (int64_t i = 0; i < kStats; ++i) out[i] = stats_[i].load();
  }

 private:
  static int64_t load_acquire(const int64_t* address) {
    return __atomic_load_n(address, __ATOMIC_ACQUIRE);
  }

  void run() {
#ifdef __linux__
    prctl(PR_SET_TIMERSLACK, 1UL, 0, 0, 0);
#endif
    int64_t idle_since = now_ns();
    uint64_t spins = 0;
    while (!stop_.load(std::memory_order_acquire)) {
      int64_t seq = load_acquire(&config_.request[0]);
      if (seq != last_seq_) {
        last_seq_ = seq;
        int64_t error = serve();
        if (error != kOk) {
          stats_[kError].store(error);
          return;  // The owner surfaces the error; never publish bad rows.
        }
        idle_since = now_ns();
        continue;
      }
      if (config_.ring != nullptr && prefetch_drafts()) {
        idle_since = now_ns();
        continue;
      }
      if ((++spins & 63) != 0) {
        cpu_relax();
        continue;
      }
      if (now_ns() - idle_since > config_.idle_spin_us * 1000) {
#ifdef __linux__
        timespec pause{0, 20000};
        nanosleep(&pause, nullptr);
#else
        std::this_thread::yield();
#endif
      }
    }
  }

  int64_t serve() {
    const int64_t start = now_ns();
    const int64_t* request = config_.request;
    const int64_t num_tokens = request[1];
    const int64_t num_reqs = request[2];
    if (num_tokens <= 0 || num_tokens > config_.max_tokens || num_reqs <= 0 ||
        num_reqs > config_.max_reqs) {
      return kBadRequest;
    }
    const int64_t* starts = request + kHeader;
    const int64_t* context = starts + config_.max_reqs + 1;
    const int64_t* tokens = context + config_.max_reqs * 2;
    hash(tokens, starts, context, num_tokens, num_reqs);
    if (config_.ring != nullptr)
      check_window(tokens, starts, context, num_reqs);
    const int64_t hashed = now_ns();

    // The consumers must have released the previous result.
    for (int32_t* flag : config_.flags) {
      while (__atomic_load_n(flag, __ATOMIC_ACQUIRE) != 0) {
        if (stop_.load(std::memory_order_acquire)) return kOk;
        cpu_relax();
      }
    }
    const int64_t reset = now_ns();

    int64_t error = gather(num_tokens);
    if (error != kOk) return error;
    const int64_t width = kHeads * config_.row_bytes;
    for (size_t i = 1; i < config_.results.size(); ++i) {
      std::memcpy(config_.results[i], config_.results[0], num_tokens * width);
    }
    for (int32_t* flag : config_.flags) {
      __atomic_store_n(flag, 1, __ATOMIC_RELEASE);
    }
    const int64_t done = now_ns();
    stats_[kRequests].fetch_add(1);
    stats_[kRows].fetch_add(num_tokens * kHeads);
    stats_[kHashNs].fetch_add(hashed - start);
    stats_[kResetWaitNs].fetch_add(reset - hashed);
    stats_[kLookupNs].fetch_add(done - reset);
    stats_[kTotalNs].fetch_add(done - start);
    if (done - start > stats_[kMaxNs].load())
      stats_[kMaxNs].store(done - start);
    stats_[kLastSeq].store(last_seq_);
    return kOk;
  }

  // 16 rows of one window position: token after (previous2, previous).
  void hash_position(uint64_t current, int64_t previous_value,
                     int64_t previous2_value, int64_t* out) const {
    const int64_t eos = config_.eos;
    const uint64_t* m = config_.multipliers;
    uint64_t previous = static_cast<uint64_t>(previous_value);
    uint64_t previous2 =
        static_cast<uint64_t>(previous_value == eos ? eos : previous2_value);
    uint64_t mixed2 = (current * m[0]) ^ (previous * m[1]);
    uint64_t mixed3 = mixed2 ^ (previous2 * m[2]);
    for (int64_t h = 0; h < 8; ++h) {
      out[h] = python_mod(mixed2, config_.sizes[h]) + config_.offsets[h];
    }
    for (int64_t h = 8; h < kHeads; ++h) {
      out[h] = python_mod(mixed3, config_.sizes[h]) + config_.offsets[h];
    }
  }

  // Ask the kernel to read the table pages of newly drafted tokens. Only
  // readahead is issued (never a blocking read), and only for rows that are
  // neither cached here nor resident, so a request is never delayed by it.
  bool prefetch_drafts() {
    const int64_t* ring = config_.ring;
    // The writer releases the round after its count, and each later count.
    const int64_t round = __atomic_load_n(&ring[0], __ATOMIC_ACQUIRE);
    const int64_t valid = __atomic_load_n(&ring[1], __ATOMIC_ACQUIRE);
    if (round == 0) return false;  // nothing published yet
    if (round != ring_round_) {
      ring_round_ = round;
      ring_done_ = 0;
    }
    const int64_t limit = std::min<int64_t>(valid, config_.ring_spec + 1);
    const int64_t num_reqs = ring[2];
    if (limit <= ring_done_ || num_reqs <= 0 || num_reqs > config_.ring_reqs) {
      return false;
    }
    const int64_t* context = ring + kHeader;
    const int64_t* drafts = context + config_.ring_reqs * 3;
    int64_t rows[kHeads];
    int64_t pages = 0;
#ifdef __linux__
    static const uintptr_t page =
        static_cast<uintptr_t>(sysconf(_SC_PAGE_SIZE));
#endif
    for (int64_t r = 0; r < num_reqs; ++r) {
      // window: context0, context1, bonus, d1 .. dK
      auto token = [&](int64_t index) -> int64_t {
        return index < 3 ? context[r * 3 + index]
                         : drafts[r * config_.ring_spec + index - 3];
      };
      for (int64_t p = ring_done_; p < limit; ++p) {
        hash_position(static_cast<uint64_t>(token(p + 2)), token(p + 1),
                      token(p), rows);
        for (int64_t h = 0; h < kHeads; ++h) {
          int64_t row = rows[h];
          if (row < 0 || row >= config_.num_rows) continue;
          if (cache_.enabled() && cache_.contains(row)) continue;
#ifdef __linux__
          uintptr_t address = reinterpret_cast<uintptr_t>(
              config_.table + row * config_.row_bytes);
          uintptr_t first = address / page * page;
          uintptr_t last = (address + config_.row_bytes - 1) / page * page;
          for (uintptr_t q = first; q <= last; q += page) {
            unsigned char resident = 0;
            if (mincore(reinterpret_cast<void*>(q), page, &resident) == 0 &&
                !(resident & 1)) {
              madvise(reinterpret_cast<void*>(q), page, MADV_WILLNEED);
              ++pages;
            }
          }
#endif
        }
      }
    }
    // A newer round may have replaced the window while it was read; the
    // advice is then merely wasted, never wrong.
    stats_[kPrefetchPositions].fetch_add((limit - ring_done_) * num_reqs);
    stats_[kPrefetchPages].fetch_add(pages);
    ring_done_ = limit;
    return true;
  }

  // Port of Qwen4ExpNGramEmbedding._compute_ngram_ids_cpu_small.
  void hash(const int64_t* tokens, const int64_t* starts,
            const int64_t* context, int64_t num_tokens, int64_t num_reqs) {
    const int64_t eos = config_.eos;
    int64_t max_seq = 1;
    for (int64_t r = 0; r < num_reqs; ++r) {
      max_seq = std::max(max_seq, starts[r + 1] - starts[r]);
    }
    const int64_t num_valid = std::min(starts[num_reqs], num_tokens);
    auto token_at = [&](int64_t req, int64_t index) -> int64_t {
      // index into [context0, context1, request tokens..., eos padding]
      if (index < 2) return context[req * 2 + index];
      int64_t begin = std::min(starts[req], num_valid);
      int64_t end = std::min(starts[req + 1], num_valid);
      int64_t position = begin + index - 2;
      if (position < end) {
        // Speculative placeholders are negative; the owner clamps them.
        return config_.clamp_tokens ? std::max<int64_t>(tokens[position], 0)
                                    : tokens[position];
      }
      return eos;
    };
    int64_t req = 0;
    for (int64_t pos = 0; pos < num_tokens; ++pos) {
      while (req + 1 < num_reqs && pos >= starts[req + 1]) ++req;
      int64_t col =
          std::min(std::max<int64_t>(pos - starts[req], 0), max_seq - 1);
      hash_position(static_cast<uint64_t>(token_at(req, col + 2)),
                    token_at(req, col + 1), token_at(req, col),
                    rows_.data() + pos * kHeads);
    }
  }

  // Diagnostics: whether a verify request equals the last published window
  // (context, bonus, drafts), i.e. whether prefetch named the right rows.
  void check_window(const int64_t* tokens, const int64_t* starts,
                    const int64_t* context, int64_t num_reqs) {
    const int64_t* ring = config_.ring;
    if (ring[2] != num_reqs) return;
    const int64_t window = config_.ring_spec + 1;
    const int64_t* ring_context = ring + kHeader;
    const int64_t* drafts = ring_context + config_.ring_reqs * 3;
    for (int64_t r = 0; r < num_reqs; ++r) {
      if (starts[r + 1] - starts[r] != window) continue;
      stats_[kWindowChecked].fetch_add(1);
      bool same = ring_context[r * 3] == context[r * 2] &&
                  ring_context[r * 3 + 1] == context[r * 2 + 1] &&
                  ring_context[r * 3 + 2] == tokens[starts[r]];
      for (int64_t k = 1; same && k < window; ++k) {
        same = drafts[r * config_.ring_spec + k - 1] == tokens[starts[r] + k];
      }
      if (same) stats_[kWindowMatched].fetch_add(1);
    }
  }

  int64_t gather(int64_t num_tokens) {
    const int64_t count = num_tokens * kHeads;
    const int64_t row_bytes = config_.row_bytes;
    uint8_t* out = config_.results[0];
    misses_.clear();
    for (int64_t i = 0; i < count; ++i) {
      int64_t row = rows_[i];
      if (row < 0 || row >= config_.num_rows) return kBadRequest;
      const uint8_t* cached = cache_.enabled() ? cache_.find(row) : nullptr;
      if (cached != nullptr) {
        std::memcpy(out + i * row_bytes, cached, row_bytes);
      } else {
        misses_.push_back(i);
      }
    }
    stats_[kHits].fetch_add(count - static_cast<int64_t>(misses_.size()));
    stats_[kMisses].fetch_add(static_cast<int64_t>(misses_.size()));
    if (misses_.empty()) return kOk;
#ifdef __linux__
    // Start every cold read before touching any, so the I/O overlaps.
    static const uintptr_t page =
        static_cast<uintptr_t>(sysconf(_SC_PAGE_SIZE));
    for (int64_t i : misses_) {
      uintptr_t address =
          reinterpret_cast<uintptr_t>(config_.table + rows_[i] * row_bytes);
      uintptr_t first = address / page * page;
      uintptr_t last = (address + row_bytes - 1) / page * page;
      for (uintptr_t p = first; p <= last; p += page) {
        unsigned char resident = 0;
        if (mincore(reinterpret_cast<void*>(p), page, &resident) == 0 &&
            !(resident & 1)) {
          madvise(reinterpret_cast<void*>(p), page, MADV_WILLNEED);
          stats_[kColdPages].fetch_add(1);
        }
      }
    }
#endif
    for (int64_t i : misses_) {
      const uint8_t* source = config_.table + rows_[i] * row_bytes;
      uint8_t* target = out + i * row_bytes;
      std::memcpy(target, source, row_bytes);
      int64_t error = validate_row(target, row_bytes);
      if (error != kOk) return error;
      if (cache_.enabled()) cache_.insert(rows_[i], target);
    }
    return kOk;
  }

  LaneConfig config_;
  RowCache cache_;
  std::vector<int64_t> rows_;
  std::vector<int64_t> misses_;
  int64_t last_seq_ = 0;
  int64_t ring_round_ = -1;
  int64_t ring_done_ = 0;
  std::atomic<bool> stop_{false};
  std::atomic<int64_t> stats_[kStats];
  std::thread thread_;
};

std::mutex lanes_mutex;
std::unordered_map<int64_t, std::unique_ptr<Lane>> lanes;
int64_t next_lane = 1;

}  // namespace

int64_t ple_fast_lane_start(
    torch::Tensor request, int64_t max_reqs, int64_t max_tokens,
    int64_t table_ptr, int64_t row_bytes, int64_t num_rows,
    std::vector<int64_t> multipliers, std::vector<int64_t> sizes,
    std::vector<int64_t> offsets, int64_t eos, std::vector<int64_t> result_ptrs,
    std::vector<int64_t> flag_ptrs, int64_t cache_bytes, int64_t idle_spin_us,
    bool clamp_tokens, std::optional<torch::Tensor> draft_ring,
    int64_t ring_reqs, int64_t ring_spec) {
  TORCH_CHECK(request.device().is_cpu() && request.scalar_type() == at::kLong &&
                  request.is_contiguous(),
              "PLE fast lane request must be a contiguous CPU int64 tensor");
  TORCH_CHECK(max_reqs > 0 && max_tokens > 0, "PLE fast lane needs capacity");
  TORCH_CHECK(request.numel() >= kHeader + max_reqs * 3 + 1 + max_tokens,
              "PLE fast lane request buffer is too small");
  TORCH_CHECK(table_ptr != 0 && num_rows > 0 && row_bytes > 0 &&
                  row_bytes % kBlockBytes == 0,
              "PLE fast lane requires a packed IQ4_NL table");
  TORCH_CHECK(multipliers.size() == 3 && sizes.size() == kHeads &&
                  offsets.size() == kHeads,
              "PLE fast lane supports trigram tables with 16 heads");
  TORCH_CHECK(!result_ptrs.empty() && result_ptrs.size() == flag_ptrs.size(),
              "PLE fast lane needs one result buffer per completion flag");
  LaneConfig config;
  config.request_owner = request;
  config.request = request.data_ptr<int64_t>();
  config.max_reqs = max_reqs;
  config.max_tokens = max_tokens;
  config.table = reinterpret_cast<const uint8_t*>(table_ptr);
  config.row_bytes = row_bytes;
  config.num_rows = num_rows;
  for (int i = 0; i < 3; ++i) {
    config.multipliers[i] = static_cast<uint64_t>(multipliers[i]);
  }
  for (int64_t h = 0; h < kHeads; ++h) {
    TORCH_CHECK(sizes[h] > 0, "PLE n-gram head sizes must be positive");
    config.sizes[h] = sizes[h];
    config.offsets[h] = offsets[h];
  }
  config.eos = eos;
  for (size_t i = 0; i < result_ptrs.size(); ++i) {
    TORCH_CHECK(result_ptrs[i] != 0 && flag_ptrs[i] != 0,
                "PLE fast lane result pointers must be valid");
    config.results.push_back(reinterpret_cast<uint8_t*>(result_ptrs[i]));
    config.flags.push_back(reinterpret_cast<int32_t*>(flag_ptrs[i]));
  }
  config.idle_spin_us = std::max<int64_t>(idle_spin_us, 0);
  config.clamp_tokens = clamp_tokens;
  if (draft_ring.has_value() && ring_reqs > 0 && ring_spec > 0) {
    const torch::Tensor& ring = *draft_ring;
    TORCH_CHECK(ring.device().is_cpu() && ring.scalar_type() == at::kLong &&
                    ring.is_contiguous() &&
                    ring.numel() >= kHeader + ring_reqs * (3 + ring_spec),
                "PLE draft ring must be a large enough CPU int64 tensor");
    config.ring_owner = ring;
    config.ring = ring.data_ptr<int64_t>();
    config.ring_reqs = ring_reqs;
    config.ring_spec = ring_spec;
  }
  std::lock_guard<std::mutex> guard(lanes_mutex);
  int64_t handle = next_lane++;
  lanes.emplace(handle, std::make_unique<Lane>(
                            std::move(config),
                            std::max<int64_t>(cache_bytes, 0) / row_bytes));
  return handle;
}

torch::Tensor ple_fast_lane_stats(int64_t handle) {
  auto out = torch::zeros({kStats}, torch::kLong);
  std::lock_guard<std::mutex> guard(lanes_mutex);
  auto it = lanes.find(handle);
  TORCH_CHECK(it != lanes.end(), "Unknown PLE fast lane handle");
  it->second->stats(out.data_ptr<int64_t>());
  return out;
}

void ple_fast_lane_stop(int64_t handle) {
  std::unique_ptr<Lane> lane;
  {
    std::lock_guard<std::mutex> guard(lanes_mutex);
    auto it = lanes.find(handle);
    if (it == lanes.end()) return;
    lane = std::move(it->second);
    lanes.erase(it);
  }
}
