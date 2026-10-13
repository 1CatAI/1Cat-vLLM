// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Host lane of the routed-expert GPU cache.
//
// Every MoE layer of a decode step publishes its routed expert ids into a
// host-mapped ring (one entry per layer call). This thread consumes the ring
// in order and keeps per-layer LRU recency. When an entry reports experts
// without a GPU slot, it plans the copies: a slot per missing expert (a spare
// one, or the least recently used expert's, written back first when the host
// has no copy of it), plus evictions that restore the spare slots. The plan is
// written into a host-mapped reply that the model stream polls; the model
// stream then performs the copies itself and updates its slot table.
//
// The lane never issues GPU work on the decode path. A release that had to
// pass through a CUDA stream could sit in the same hardware queue behind the
// very wait it releases (streams share a few hardware queues), which
// deadlocks. Results never depend on residency: a slot holds the same bytes
// as the expert's original bank row.

#include <torch/extension.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
  #include <immintrin.h>
#endif

namespace {

constexpr int kShards = 3;
constexpr int kEntryHeader = 8;  // seq, layer, count, miss, reserved...
constexpr int kReplyHeader = 8;  // seq, ops, failed, reserved...
constexpr int kOpInts = 5;       // slot, load host slot, writeback host slot,
                                 // expert now in slot, expert evicted

enum Stat : int {
  kEntries = 0,
  kRequests,
  kLoaded,
  kEvicted,
  kWritebacks,
  kServeNs,
  kMaxServeNs,
  kSkipped,
  kErrors,
  kPrefills,
  kPublished,
  kConsumed,
  kStats
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

void check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("expert cache: ") + what + ": " +
                             cudaGetErrorString(status));
  }
}

class ExpertCacheLane {
 public:
  ExpertCacheLane(int64_t device, int64_t layers, int64_t experts,
                  int64_t slots, int64_t host_slots, int64_t spare,
                  std::vector<int64_t> slot_bytes,
                  std::vector<int64_t> device_banks,
                  std::vector<int64_t> host_banks, int64_t ring,
                  int64_t ring_entries, int64_t entry_ints, int64_t published,
                  int64_t reply, int64_t max_ops, torch::Tensor slot_of,
                  torch::Tensor host_slot_of)
      : device_(device),
        layers_(layers),
        experts_(experts),
        slots_(slots),
        host_slots_(host_slots),
        spare_(spare),
        max_ops_(max_ops),
        slot_bytes_(std::move(slot_bytes)),
        device_banks_(std::move(device_banks)),
        host_banks_(std::move(host_banks)),
        ring_(reinterpret_cast<int32_t*>(ring)),
        ring_entries_(ring_entries),
        entry_ints_(entry_ints),
        published_(reinterpret_cast<int64_t*>(published)),
        reply_(reinterpret_cast<int32_t*>(reply)),
        slot_of_(slot_of.data_ptr<int32_t>()),
        host_slot_of_(host_slot_of.data_ptr<int32_t>()) {
    TORCH_CHECK(slot_bytes_.size() == size_t(layers * kShards) &&
                    device_banks_.size() == size_t(layers * kShards) &&
                    host_banks_.size() == size_t(layers * kShards),
                "expert cache: bad bank tables");
    TORCH_CHECK(slot_of.numel() == layers * experts &&
                    host_slot_of.numel() == layers * experts,
                "expert cache: bad residency tables");
    TORCH_CHECK(entry_ints_ > kEntryHeader && ring_entries_ > 0,
                "expert cache: bad ring");
    TORCH_CHECK(max_ops_ >= entry_ints_ - kEntryHeader + spare_,
                "expert cache: reply too small");
    expert_of_slot_.assign(layers * slots, -1);
    host_owner_.assign(layers * host_slots, -1);
    last_use_.assign(layers * experts, 0);
    clock_.assign(layers, 0);
    free_slots_.resize(layers);
    free_host_.resize(layers);
    for (int64_t l = 0; l < layers; ++l) {
      for (int64_t e = 0; e < experts; ++e) {
        int32_t s = slot_of_[l * experts + e];
        int32_t h = host_slot_of_[l * experts + e];
        if (s >= 0) {
          TORCH_CHECK(s < slots && expert_of_slot_[l * slots + s] < 0,
                      "expert cache: duplicate slot");
          expert_of_slot_[l * slots + s] = e;
        } else {
          TORCH_CHECK(h >= 0, "expert cache: expert without storage");
        }
        if (h >= 0) {
          TORCH_CHECK(h < host_slots && host_owner_[l * host_slots + h] < 0,
                      "expert cache: duplicate host slot");
          host_owner_[l * host_slots + h] = e;
        }
      }
      for (int64_t s = slots - 1; s >= 0; --s)
        if (expert_of_slot_[l * slots + s] < 0)
          free_slots_[l].push_back(static_cast<int32_t>(s));
      for (int64_t h = host_slots - 1; h >= 0; --h)
        if (host_owner_[l * host_slots + h] < 0)
          free_host_[l].push_back(static_cast<int32_t>(h));
    }
    for (auto& value : stats_) value.store(0);
    check(cudaHostAlloc(reinterpret_cast<void**>(&prefill_maps_),
                        layers * experts * sizeof(int32_t),
                        cudaHostAllocPortable),
          "alloc prefill maps");
    check(cudaHostAlloc(reinterpret_cast<void**>(&residency_maps_),
                        2 * layers * experts * sizeof(int32_t),
                        cudaHostAllocPortable),
          "alloc residency maps");
    consumed_ = __atomic_load_n(published_, __ATOMIC_ACQUIRE);
    consumed_seen_.store(consumed_);
    thread_ = std::thread([this] { run(); });
  }

  ~ExpertCacheLane() { stop(); }

  void stop() {
    stop_.store(true, std::memory_order_release);
    if (thread_.joinable()) thread_.join();
    if (prefill_maps_ != nullptr) {
      cudaFreeHost(prefill_maps_);
      prefill_maps_ = nullptr;
    }
    if (residency_maps_ != nullptr) {
      cudaFreeHost(residency_maps_);
      residency_maps_ = nullptr;
    }
  }

  std::vector<int64_t> stats() const {
    std::vector<int64_t> out(kStats);
    for (int i = 0; i < kStats; ++i) out[i] = stats_[i].load();
    out[kPublished] = __atomic_load_n(published_, __ATOMIC_ACQUIRE);
    out[kConsumed] = consumed_seen_.load();
    return out;
  }

  // Copy layer `layer`'s complete bank into `dst` (one per shard, `capacity`
  // rows each): the slot bank first, then every non-resident expert, on
  // `stream`. Writes the expert -> row map to `map_device`. The caller orders
  // `stream` after the decode steps that last changed this layer.
  int64_t prefill(int64_t layer, std::vector<int64_t> dst, int64_t capacity,
                  int64_t map_device, int64_t stream_ptr) {
    TORCH_CHECK(layer >= 0 && layer < layers_ && dst.size() == kShards,
                "expert cache: bad prefill request");
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    std::lock_guard<std::mutex> guard(mutex_);
    check(cudaSetDevice(static_cast<int>(device_)), "set device");
    for (int shard = 0; shard < kShards; ++shard) {
      check(cudaMemcpyAsync(reinterpret_cast<void*>(dst[shard]),
                            bank(layer, shard), slots_ * bytes(layer, shard),
                            cudaMemcpyDeviceToDevice, stream),
            "prefill slot bank");
    }
    int32_t* map = prefill_maps_ + layer * experts_;
    int64_t cold = 0;
    for (int64_t e = 0; e < experts_; ++e) {
      int32_t s = slot_of_[layer * experts_ + e];
      if (s >= 0) {
        map[e] = s;
        continue;
      }
      int64_t row = slots_ + cold++;
      TORCH_CHECK(row < capacity, "expert cache: prefill bank too small");
      int32_t h = host_slot_of_[layer * experts_ + e];
      for (int shard = 0; shard < kShards; ++shard) {
        check(cudaMemcpyAsync(reinterpret_cast<uint8_t*>(dst[shard]) +
                                  row * bytes(layer, shard),
                              host(layer, shard, h), bytes(layer, shard),
                              cudaMemcpyHostToDevice, stream),
              "prefill cold expert");
      }
      map[e] = static_cast<int32_t>(row);
    }
    check(cudaMemcpyAsync(reinterpret_cast<void*>(map_device), map,
                          experts_ * sizeof(int32_t), cudaMemcpyHostToDevice,
                          stream),
          "prefill map");
    stats_[kPrefills]++;
    return cold;
  }

  // Copy layer `layer`'s residency to `map_device` on `stream`: int32
  // [2, experts], the GPU slot of each expert (-1 when not resident), then
  // its host slot (-1 when the host has no copy). The prefill transcode
  // reads every expert from where this snapshot says it lives. The caller
  // orders `stream` after the decode steps that last changed this layer.
  // Returns the host slots that hold non-resident experts: [0, extent).
  int64_t residency(int64_t layer, int64_t map_device, int64_t stream_ptr) {
    TORCH_CHECK(layer >= 0 && layer < layers_, "expert cache: bad layer");
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    std::lock_guard<std::mutex> guard(mutex_);
    int32_t* map = residency_maps_ + 2 * layer * experts_;
    std::memcpy(map, slot_of_ + layer * experts_, experts_ * sizeof(int32_t));
    std::memcpy(map + experts_, host_slot_of_ + layer * experts_,
                experts_ * sizeof(int32_t));
    check(cudaSetDevice(static_cast<int>(device_)), "set device");
    check(cudaMemcpyAsync(reinterpret_cast<void*>(map_device), map,
                          2 * experts_ * sizeof(int32_t),
                          cudaMemcpyHostToDevice, stream),
          "residency map");
    stats_[kPrefills]++;
    int64_t extent = 0;
    for (int64_t e = 0; e < experts_; ++e)
      if (map[e] < 0) extent = std::max<int64_t>(extent, map[experts_ + e] + 1);
    return extent;
  }

 private:
  int64_t bytes(int64_t layer, int shard) const {
    return slot_bytes_[layer * kShards + shard];
  }
  void* bank(int64_t layer, int shard) const {
    return reinterpret_cast<void*>(device_banks_[layer * kShards + shard]);
  }
  uint8_t* host(int64_t layer, int shard, int64_t h) const {
    return reinterpret_cast<uint8_t*>(host_banks_[layer * kShards + shard]) +
           h * bytes(layer, shard);
  }

  void run() {
    try {
      int64_t idle_since = now_ns();
      while (!stop_.load(std::memory_order_acquire)) {
        int64_t published = __atomic_load_n(published_, __ATOMIC_ACQUIRE);
        if (published == consumed_) {
          if (now_ns() - idle_since > 5'000'000) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
          } else {
            cpu_relax();
          }
          continue;
        }
        if (published - consumed_ > ring_entries_ - 2) {
          // The model stream never runs ahead of an unserved miss, so only
          // recency information is lost by skipping old entries.
          stats_[kSkipped] += published - 1 - consumed_;
          consumed_ = published - 1;
        }
        while (consumed_ < published) {
          try {
            serve(consumed_);
          } catch (const std::exception& error) {
            fail(consumed_, error.what());
          }
          ++consumed_;
        }
        consumed_seen_.store(consumed_, std::memory_order_release);
        idle_since = now_ns();
      }
    } catch (const std::exception& error) {
      stats_[kErrors]++;
      fprintf(stderr, "expert cache lane stopped: %s\n", error.what());
    }
  }

  // A request that cannot be planned still has to release the model stream:
  // answer with no copies and a sticky failure flag, which every later wait
  // also honours. Residency state may be inconsistent after this.
  void fail(int64_t seq, const char* what) {
    stats_[kErrors]++;
    fprintf(stderr, "expert cache lane: entry %lld failed: %s\n",
            static_cast<long long>(seq), what);
    __atomic_store_n(reply_ + 2, 1, __ATOMIC_RELEASE);
    __atomic_store_n(reply_ + 1, 0, __ATOMIC_RELEASE);
    __atomic_store_n(reply_, static_cast<int32_t>(seq), __ATOMIC_RELEASE);
  }

  void serve(int64_t seq) {
    const int32_t* entry = ring_ + (seq % ring_entries_) * entry_ints_;
    // The entry's tag is released after its body; wait for it rather than
    // dropping a request the model stream is blocked on.
    int32_t tag = __atomic_load_n(entry, __ATOMIC_ACQUIRE);
    for (int64_t deadline = now_ns() + 1'000'000'000;
         tag != static_cast<int32_t>(seq) && now_ns() < deadline;) {
      cpu_relax();
      tag = __atomic_load_n(entry, __ATOMIC_ACQUIRE);
    }
    int32_t layer = __atomic_load_n(entry + 1, __ATOMIC_ACQUIRE);
    int32_t count = __atomic_load_n(entry + 2, __ATOMIC_ACQUIRE);
    int32_t miss = __atomic_load_n(entry + 3, __ATOMIC_ACQUIRE);
    if (tag != static_cast<int32_t>(seq) || layer < 0 || layer >= layers_ ||
        count < 0 || count > entry_ints_ - kEntryHeader) {
      char what[96];
      snprintf(what, sizeof(what), "bad entry (tag %d layer %d count %d)", tag,
               layer, count);
      throw std::runtime_error(what);
    }
    const int32_t* ids = entry + kEntryHeader;
    stats_[kEntries]++;
    uint64_t now = ++clock_[layer];
    uint64_t* last = last_use_.data() + layer * experts_;
    for (int32_t i = 0; i < count; ++i) {
      int32_t e = ids[i];
      if (e >= 0 && e < experts_) last[e] = now;
    }
    if (!miss) return;
    if (__atomic_load_n(reply_ + 2, __ATOMIC_ACQUIRE))
      throw std::runtime_error("lane already failed");
    int64_t start = now_ns();
    std::lock_guard<std::mutex> guard(mutex_);
    int32_t* slot_of = slot_of_ + layer * experts_;
    ops_.clear();
    for (int32_t i = 0; i < count; ++i) {
      int32_t e = ids[i];
      if (e < 0 || e >= experts_ || slot_of[e] >= 0) continue;
      int32_t s, writeback = -1, victim = -1;
      int32_t load = host_slot_of_[layer * experts_ + e];
      if (free_slots_[layer].empty()) {
        s = evict(layer, now, &writeback, &victim, e);
        TORCH_CHECK(s >= 0, "expert cache: no evictable slot");
      } else {
        s = free_slots_[layer].back();
        free_slots_[layer].pop_back();
      }
      ops_.insert(ops_.end(), {s, load, writeback, e, victim});
      slot_of[e] = s;
      expert_of_slot_[layer * slots_ + s] = e;
      stats_[kLoaded]++;
    }
    // Restore the spare slots in the same plan; these slots are not loaded
    // before a later request, so their write-backs need no ordering here.
    // Best effort: a later request retries when the host arena is short.
    while (static_cast<int64_t>(free_slots_[layer].size()) < spare_) {
      int32_t writeback = -1, victim = -1;
      int32_t s = evict(layer, now, &writeback, &victim, -1);
      if (s < 0) break;
      ops_.insert(ops_.end(), {s, -1, writeback, -1, victim});
      free_slots_[layer].push_back(s);
    }
    int64_t n = static_cast<int64_t>(ops_.size()) / kOpInts;
    TORCH_CHECK(n <= max_ops_, "expert cache: plan exceeds reply capacity");
    std::memcpy(reply_ + kReplyHeader, ops_.data(),
                ops_.size() * sizeof(int32_t));
    __atomic_store_n(reply_ + 1, static_cast<int32_t>(n), __ATOMIC_RELEASE);
    __atomic_store_n(reply_, static_cast<int32_t>(seq), __ATOMIC_RELEASE);
    stats_[kRequests]++;
    int64_t served = now_ns() - start;
    stats_[kServeNs] += served;
    if (served > stats_[kMaxServeNs].load()) stats_[kMaxServeNs].store(served);
  }

  // Evict the least recently used expert not routed by the current entry and
  // return its slot (-1 when nothing can be evicted). Reports the host slot it
  // must be written back to (-1 when the host keeps a copy) and the evicted
  // expert. With the host arena full, the victim swaps places with `incoming`
  // (the expert loaded into its slot by the same op): the copy kernel reads
  // both sides of each word before writing either.
  int32_t evict(int32_t layer, uint64_t now, int32_t* writeback,
                int32_t* victim_out, int32_t incoming) {
    const uint64_t* last = last_use_.data() + layer * experts_;
    int32_t* owners = expert_of_slot_.data() + layer * slots_;
    int32_t victim_slot = -1;
    uint64_t best = UINT64_MAX;
    for (int64_t s = 0; s < slots_; ++s) {
      int32_t e = owners[s];
      if (e < 0 || last[e] >= now) continue;
      if (last[e] < best) {
        best = last[e];
        victim_slot = static_cast<int32_t>(s);
      }
    }
    if (victim_slot < 0) return -1;
    int32_t victim = owners[victim_slot];
    int32_t* host_slot = host_slot_of_ + layer * experts_;
    *writeback = -1;
    if (host_slot[victim] < 0) {
      int32_t h = take_host_slot(layer, now);
      if (h < 0 && incoming >= 0 && host_slot[incoming] >= 0) {
        h = host_slot[incoming];
        host_slot[incoming] = -1;
      }
      if (h < 0) return -1;
      host_slot[victim] = h;
      host_owner_[layer * host_slots_ + h] = victim;
      *writeback = h;
      stats_[kWritebacks]++;
    }
    slot_of_[layer * experts_ + victim] = -1;
    owners[victim_slot] = -1;
    *victim_out = victim;
    stats_[kEvicted]++;
    return victim_slot;
  }

  // A free host slot, or the host copy of the most recently used resident
  // expert (the least likely to be evicted next), or -1. Copies of experts
  // routed by the current entry are never taken: this plan may still load
  // them.
  int32_t take_host_slot(int32_t layer, uint64_t now) {
    if (!free_host_[layer].empty()) {
      int32_t h = free_host_[layer].back();
      free_host_[layer].pop_back();
      return h;
    }
    const uint64_t* last = last_use_.data() + layer * experts_;
    int32_t* owners = host_owner_.data() + layer * host_slots_;
    int32_t chosen = -1;
    uint64_t best = 0;
    for (int64_t h = 0; h < host_slots_; ++h) {
      int32_t e = owners[h];
      if (e < 0 || slot_of_[layer * experts_ + e] < 0 || last[e] >= now)
        continue;
      if (chosen < 0 || last[e] >= best) {
        best = last[e];
        chosen = static_cast<int32_t>(h);
      }
    }
    if (chosen < 0) return -1;
    host_slot_of_[layer * experts_ + owners[chosen]] = -1;
    owners[chosen] = -1;
    return chosen;
  }

  int64_t device_, layers_, experts_, slots_, host_slots_, spare_, max_ops_;
  std::vector<int64_t> slot_bytes_, device_banks_, host_banks_;
  int32_t* ring_;
  int64_t ring_entries_, entry_ints_;
  int64_t* published_;
  int32_t* reply_;
  int32_t* slot_of_;
  int32_t* host_slot_of_;
  std::vector<int32_t> expert_of_slot_, host_owner_, ops_;
  std::vector<uint64_t> last_use_, clock_;
  std::vector<std::vector<int32_t>> free_slots_, free_host_;
  int32_t* prefill_maps_ = nullptr;
  int32_t* residency_maps_ = nullptr;
  int64_t consumed_ = 0;
  std::atomic<int64_t> consumed_seen_{0};
  std::atomic<bool> stop_{false};
  std::atomic<int64_t> stats_[kStats];
  std::mutex mutex_;
  std::thread thread_;
};

int64_t host_register(int64_t pointer, int64_t bytes) {
  check(cudaHostRegister(reinterpret_cast<void*>(pointer), bytes,
                         cudaHostRegisterPortable | cudaHostRegisterMapped),
        "register host arena");
  return pointer;
}

void host_unregister(int64_t pointer) {
  cudaHostUnregister(reinterpret_cast<void*>(pointer));
}

// Device view of registered host memory (no copy). vLLM's
// get_cuda_view_from_cpu_tensor copies tensors it does not see as pinned,
// which includes cudaHostRegister'ed memory.
torch::Tensor mapped_view(torch::Tensor host, int64_t device) {
  void* pointer = nullptr;
  check(cudaHostGetDevicePointer(&pointer, host.data_ptr(), 0),
        "map host arena");
  return torch::from_blob(
      pointer, host.sizes(), host.strides(), [host](void*) {},
      host.options().device(torch::kCUDA, static_cast<int>(device)));
}

// Asynchronous host -> device copy by the copy engine. `src` may be
// cudaHostRegister'ed memory, which torch does not see as pinned (its copy
// would stage synchronously).
void copy_h2d(int64_t dst, int64_t src, int64_t bytes, int64_t stream_ptr) {
  check(cudaMemcpyAsync(reinterpret_cast<void*>(dst),
                        reinterpret_cast<const void*>(src), bytes,
                        cudaMemcpyHostToDevice,
                        reinterpret_cast<cudaStream_t>(stream_ptr)),
        "copy host to device");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<ExpertCacheLane>(m, "ExpertCacheLane")
      .def(pybind11::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                          std::vector<int64_t>, std::vector<int64_t>,
                          std::vector<int64_t>, int64_t, int64_t, int64_t,
                          int64_t, int64_t, int64_t, torch::Tensor,
                          torch::Tensor>())
      .def("stop", &ExpertCacheLane::stop,
           pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("stats", &ExpertCacheLane::stats)
      .def("prefill", &ExpertCacheLane::prefill,
           pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("residency", &ExpertCacheLane::residency,
           pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("host_register", &host_register);
  m.def("host_unregister", &host_unregister);
  m.def("mapped_view", &mapped_view);
  m.def("copy_h2d", &copy_h2d);
}
