// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "csrc/gguf_shared_expert_state.h"
#include "csrc/gguf_shared_expert_planner.h"
#include "csrc/gguf_shared_expert_lane.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <thread>

using vllm::gguf::SharedExpertSwapState;
constexpr uint64_t kHigh = uint64_t{5} << 30;

template <class Exception = std::runtime_error, class Function>
void rejects(Function fn) {
  bool rejected = false;
  try {
    fn();
  } catch (const Exception&) {
    rejected = true;
  }
  assert(rejected);
}

void slow_rank_and_prefetch() {
  SharedExpertSwapState state(kHigh + 8192, 4);
  auto lease = state.acquire({kHigh, 4096});
  state.begin(1, {{kHigh, 4096}}, {{0, 4096}}, 100, 1000);
  state.acknowledge(3, 1, 101);
  state.acknowledge(0, 1, 102);
  state.acknowledge(2, 1, 103);
  assert(!state.ready(1, 104));
  rejects([&] { state.finish(1, 104); });
  state.acknowledge(1, 1, 105);
  assert(!state.ready(1, 106));  // All ranks done, prefetch still reading.
  rejects([&] { state.acquire({kHigh + 1024, 128}); });
  rejects([&] { state.acquire({0, 128}); });
  state.release(lease);
  assert(state.ready(1, 107));
  auto freed = state.finish(1, 108);
  assert(freed.size() == 1 && freed[0].offset == kHigh);
  assert(freed[0].bytes == 4096);
  auto next = state.acquire({0, 4096});
  state.release(next);
}

void repeated_eviction_and_completion_orders() {
  std::array<unsigned, 4> ranks{0, 1, 2, 3};
  SharedExpertSwapState state(kHigh + 8192, 4);
  uint64_t sequence = 0;
  do {
    const uint64_t source = sequence % 2 ? 0 : kHigh;
    const uint64_t destination = source == 0 ? kHigh : 0;
    state.begin(++sequence, {{source, 4096}}, {{destination, 4096}}, 1, 100);
    for (unsigned i = 0; i < ranks.size(); ++i) {
      state.acknowledge(ranks[i], sequence, 2);
      assert(state.ready(sequence, 3) == (i == 3));
    }
    assert(state.finish(sequence, 4)[0].offset == source);
  } while (std::next_permutation(ranks.begin(), ranks.end()));
  assert(sequence == 24);
}

void invalid_sequences_fail_closed() {
  for (int mode = 0; mode < 4; ++mode) {
    SharedExpertSwapState state(8192, 4);
    state.begin(1, {{0, 4096}}, {{4096, 4096}}, 1, 100);
    rejects([&] {
      if (mode == 0) state.acknowledge(0, 2, 2);
      if (mode == 1) {
        state.acknowledge(0, 1, 2);
        state.acknowledge(0, 1, 3);
      }
      if (mode == 2) state.acknowledge(4, 1, 2);
      if (mode == 3) state.release(1234);
    });
    rejects([&] { state.ready(1, 4); });
    rejects([&] { state.begin(2, {{0, 4096}}, {{4096, 4096}}, 1, 100); });
  }
}

void timeout_and_coordinator_exit() {
  SharedExpertSwapState state(8192, 4);
  state.begin(1, {{0, 4096}}, {{4096, 4096}}, 100, 10);
  state.acknowledge(0, 1, 109);
  rejects([&] { state.acknowledge(1, 1, 110); });
  rejects([&] { state.ready(1, 111); });
  rejects([&] { state.finish(1, 111); });
  SharedExpertSwapState stopped(8192, 4);
  stopped.begin(1, {{0, 4096}}, {{4096, 4096}}, 1, 100);
  stopped.stop();
  rejects([&] { stopped.acknowledge(0, 1, 2); });
  rejects([&] { stopped.acquire({0, 4096}); });
  rejects([&] { stopped.finish(1, 2); });
}

void reject_aliases_and_invalid_ranges() {
  SharedExpertSwapState state(32768, 4);
  rejects<std::invalid_argument>(
      [&] { state.begin(1, {{0, 4096}}, {{1024, 4096}}, 1, 100); });
  rejects<std::invalid_argument>([&] {
    state.begin(1, {{0, 4096}, {8192, 4096}}, {{16384, 4096}, {17408, 4096}}, 1,
                100);
  });
  rejects<std::invalid_argument>(
      [&] { state.acquire({std::numeric_limits<uint64_t>::max() - 1, 4096}); });
  rejects<std::invalid_argument>([&] { state.acquire({0, 0}); });
  auto lease = state.acquire({8192, 4096});
  rejects<std::invalid_argument>(
      [&] { state.begin(1, {{0, 4096}}, {{8192, 4096}}, 1, 100); });
  state.release(lease);
  state.begin(1, {{0, 4096}}, {{8192, 4096}}, 1, 100);
  auto unrelated = state.acquire({16384, 4096});
  state.release(unrelated);
}

void concurrent_completion_publication() {
  SharedExpertSwapState state(8192, 4);
  state.begin(1, {{0, 4096}}, {{4096, 4096}}, 1, 100);
  std::vector<std::thread> workers;
  for (unsigned rank = 0; rank < 4; ++rank)
    workers.emplace_back([&, rank] { state.acknowledge(rank, 1, 2); });
  for (auto& worker : workers) worker.join();
  assert(state.ready(1, 3));
  assert(state.finish(1, 4).size() == 1);
}

void shared_planner_preserves_cold_bytes_across_evictions() {
  using vllm::gguf::SharedExpertPlanner;
  // Two differently sized layers and one reusable exchange reserve.
  std::vector<uint8_t> arena(4 * 16 + 4 * 24 + 4 * 24, 0);
  SharedExpertPlanner planner(arena.data(), arena.size(),
                              {{16, 8, 4}, {24, 8, 4}}, 4, 4, 4 * 24);
  std::vector<std::vector<uint8_t>> gpu(4, std::vector<uint8_t>(24));
  uint64_t sequence = 0;
  for (size_t layer = 0; layer < 2; ++layer) {
    const size_t bytes = layer == 0 ? 16 : 24;
    for (int expert = 0; expert < 8; ++expert) {
      if (expert < 4)
        std::fill(gpu[expert].begin(), gpu[expert].end(), expert);
      else {
        auto offset = planner.cold_offset(layer, expert);
        std::fill_n(arena.begin() + offset, bytes, expert);
      }
    }
    for (int iteration = 0; iteration < 40; ++iteration) {
      const int base = iteration % 2 ? 0 : 4;
      const std::vector<int32_t> routes{base, base + 1, base + 2, base + 3};
      auto copies = planner.plan(layer, ++sequence, routes, 1, 100);
      assert(copies.size() == 4);
      for (auto copy : copies) {
        assert(copy.load_offset != copy.save_offset);
        std::memcpy(arena.data() + copy.save_offset, gpu[copy.slot].data(),
                    bytes);
        std::memcpy(gpu[copy.slot].data(), arena.data() + copy.load_offset,
                    bytes);
      }
      for (unsigned rank : {3, 1, 0, 2}) planner.acknowledge(rank, sequence, 2);
      assert(planner.ready(sequence, 3));
      planner.finish(sequence, 4);
      for (int expert = 0; expert < 8; ++expert) {
        const int slot = planner.slot(layer, expert);
        const uint8_t* data =
            slot >= 0 ? gpu[slot].data()
                      : arena.data() + planner.cold_offset(layer, expert);
        assert(std::all_of(data, data + bytes,
                           [=](auto b) { return b == expert; }));
      }
    }
  }
}

void shared_planner_leases_hits_duplicates_and_capacity() {
  using vllm::gguf::SharedExpertPlanner;
  std::vector<uint8_t> arena(128);
  SharedExpertPlanner planner(arena.data(), arena.size(), {{16, 8, 4}}, 4, 4,
                              64);
  const auto lease = planner.acquire(0, 4);
  auto copies = planner.plan(0, 1, {4, 4, 1, 1}, 1, 100);
  assert(copies.size() == 1 && copies[0].victim == 0);
  for (unsigned rank = 0; rank < 4; ++rank) planner.acknowledge(rank, 1, 2);
  assert(!planner.ready(1, 3));
  rejects([&] { planner.finish(1, 3); });
  planner.release(lease.first);
  planner.finish(1, 4);
  assert(planner.slot(0, 4) == 0 && planner.slot(0, 0) == -1);
  assert(planner.plan(0, 2, {4, 1, 1}, 5, 100).empty());
  for (unsigned rank = 0; rank < 4; ++rank) planner.acknowledge(rank, 2, 6);
  planner.finish(2, 7);
  rejects<std::invalid_argument>(
      [&] { planner.plan(0, 3, {0, 1, 2, 3, 4}, 8, 100); });
  rejects<std::invalid_argument>([&] { planner.plan(0, 3, {-1}, 8, 100); });
  rejects<std::invalid_argument>([&] {
    SharedExpertPlanner invalid(arena.data(), arena.size(), {{16, 8, 4}}, 4, 4,
                                63);
  });
}

void wait_for(const std::function<bool()>& predicate) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!predicate()) {
    assert(std::chrono::steady_clock::now() < deadline);
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
}

void publish(uint64_t* control, unsigned rank, uint64_t sequence,
             uint64_t expert) {
  using namespace vllm::gguf;
  auto* request = control + kSharedRankStart + rank * kSharedRankWords;
  request[2] = 0;
  request[3] = 1;
  request[4] = expert;
  __atomic_store_n(request, sequence, __ATOMIC_RELEASE);
}

void coordinator_thread_waits_for_all_ranks_and_readers() {
  using namespace vllm::gguf;
  alignas(64) std::array<uint64_t, kSharedHitTotalWords> control{};
  std::vector<uint8_t> arena(128, 4);
  SharedExpertCoordinator lane(arena.data(), arena.size(), control.data(),
                               {{16, 8, 4}}, 4, 4, 64, 1000000000);
  const auto lease = lane.acquire(0, 4);
  for (unsigned rank = 0; rank < 4; ++rank) publish(control.data(), rank, 1, 4);
  wait_for([&] { return __atomic_load_n(&control[0], __ATOMIC_ACQUIRE) == 1; });
  assert(control[3] == 1 && control[kSharedOpStart + 1] == 4);
  const auto input = control[kSharedOpStart + 3],
             output = control[kSharedOpStart + 4];
  std::fill_n(arena.begin() + output, 16, 0);
  for (unsigned rank = 0; rank < 4; ++rank) {
    auto* request = control.data() + kSharedRankStart + rank * kSharedRankWords;
    __atomic_store_n(request + 1, 1, __ATOMIC_RELEASE);
  }
  assert(__atomic_load_n(&control[1], __ATOMIC_ACQUIRE) == 0);
  assert(arena[input] == 4);
  lane.release(lease.first);
  wait_for([&] { return __atomic_load_n(&control[1], __ATOMIC_ACQUIRE) == 1; });
  assert(arena[input] == 0 && lane.slot(0, 4) == 0);
  for (unsigned rank = 0; rank < 4; ++rank) publish(control.data(), rank, 2, 4);
  wait_for([&] { return __atomic_load_n(&control[0], __ATOMIC_ACQUIRE) == 2; });
  assert(control[3] == 0);
  for (unsigned rank = 0; rank < 4; ++rank) {
    auto* request = control.data() + kSharedRankStart + rank * kSharedRankWords;
    __atomic_store_n(request + 1, 2, __ATOMIC_RELEASE);
  }
  wait_for([&] { return __atomic_load_n(&control[1], __ATOMIC_ACQUIRE) == 2; });
  lane.stop();
  assert(lane.error().empty());
}

void coordinator_rank_divergence_and_timeout() {
  using namespace vllm::gguf;
  for (bool timeout : {false, true}) {
    alignas(64) std::array<uint64_t, kSharedHitTotalWords> control{};
    std::vector<uint8_t> arena(128, 4);
    SharedExpertCoordinator lane(arena.data(), arena.size(), control.data(),
                                 {{16, 8, 4}}, 4, 4, 64, 40000000);
    for (unsigned rank = 0; rank < (timeout ? 3 : 4); ++rank)
      publish(control.data(), rank, 1, rank == 3 ? 5 : 4);
    wait_for(
        [&] { return __atomic_load_n(&control[2], __ATOMIC_ACQUIRE) == 1; });
    assert(__atomic_load_n(&control[0], __ATOMIC_ACQUIRE) == 0);
    assert(lane.error().find(timeout ? "timed out" : "routes differ") !=
           std::string::npos);
    lane.stop();
  }
}

int main(int argc, char** argv) {
  assert(argc == 2);
  const std::unordered_map<std::string, std::function<void()>> cases{
      {"slow_rank_and_prefetch", slow_rank_and_prefetch},
      {"repeated_eviction", repeated_eviction_and_completion_orders},
      {"invalid_sequences", invalid_sequences_fail_closed},
      {"timeout_and_exit", timeout_and_coordinator_exit},
      {"aliases_and_ranges", reject_aliases_and_invalid_ranges},
      {"concurrent_completion", concurrent_completion_publication},
      {"planner_eviction",
       shared_planner_preserves_cold_bytes_across_evictions},
      {"planner_leases", shared_planner_leases_hits_duplicates_and_capacity},
      {"coordinator_thread",
       coordinator_thread_waits_for_all_ranks_and_readers},
      {"coordinator_failure", coordinator_rank_divergence_and_timeout}};
  cases.at(argv[1])();
}
