// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include "gguf_shared_expert_state.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <unordered_map>
#include <unordered_set>

namespace vllm::gguf {

struct SharedExpertLayer {
  uint64_t expert_bytes;
  int32_t experts;
  int32_t resident;
};

struct SharedExpertCopy {
  int32_t slot;
  int32_t incoming;
  int32_t victim;
  uint64_t load_offset;
  uint64_t save_offset;
};

// Background copy-engine job for every rank's job lane (VLLM_V41_ASYNC_EVICT):
// kind 0 saves slot -> host offset (device to host, the rank's exclusive
// bytes), kind 1 loads host offset -> slot.
struct SharedExpertJob {
  uint64_t id;  // ring index; ordinal counts jobs of this kind
  int32_t kind, layer, slot, expert;
  uint64_t offset, ordinal;
};

constexpr uint64_t kSharedNoOffset = ~uint64_t{0};

// One owner per TP group, called by its native coordination lane. Persistent
// cold slots contain only original, unsliced experts. The exchange reserve
// contains independent victim destinations during a plan. Once every rank
// and prefetch reader completes, compact victims into the vacated cold slots
// before reusing exchange space. This bounded host byte copy avoids allocator
// fragmentation across layers with different expert encodings/sizes; it does
// not create a second persistent cold bank or perform CPU matrix arithmetic.
class SharedExpertPlanner {
 public:
  SharedExpertPlanner(uint8_t* arena, uint64_t arena_bytes,
                      std::vector<SharedExpertLayer> layers, unsigned ranks,
                      unsigned max_routes, uint64_t exchange_bytes)
      : arena_(arena), max_routes_(max_routes), fence_(arena_bytes, ranks) {
    if (!arena || layers.empty() || !max_routes || exchange_bytes > arena_bytes)
      throw std::invalid_argument("invalid shared expert planner dimensions");
    uint64_t offset = 0;
    for (auto config : layers) {
      if (!config.expert_bytes || config.experts <= 1 || config.resident <= 0 ||
          config.resident >= config.experts ||
          max_routes > unsigned(config.resident))
        throw std::invalid_argument("invalid shared expert layer capacity");
      const auto cold = uint64_t(config.experts - config.resident);
      if (config.expert_bytes > (arena_bytes - offset) / cold)
        throw std::invalid_argument("cold experts exceed the shared arena");
      Layer layer{config, offset, {}, {}, {}};
      layer.slot.assign(config.experts, -1);
      layer.cold.assign(config.experts, -1);  // absolute arena offsets
      layer.use.assign(config.experts, 0);
      for (int32_t expert = 0; expert < config.experts; ++expert) {
        if (expert < config.resident)
          layer.slot[expert] = expert;
        else
          layer.cold[expert] =
              int64_t(offset +
                      uint64_t(expert - config.resident) * config.expert_bytes);
      }
      layers_.push_back(std::move(layer));
      offset += cold * config.expert_bytes;
      if (max_routes > exchange_bytes / config.expert_bytes)
        throw std::invalid_argument(
            "shared exchange cannot hold maximum routes");
    }
    if (offset > arena_bytes - exchange_bytes)
      throw std::invalid_argument("cold experts overlap the exchange reserve");
    exchange_offset_ = offset;
    if (free_mode_) {
      // Free-slot exchange (VLLM_V41_SWAP_FREE): the exchange reserve becomes
      // per-size-class free cold allocations. A victim is saved into a free
      // slot while the incoming is read (no aliasing, no host compaction); the
      // incoming's vacated allocation joins its class list after every reader.
      const char* env = std::getenv("VLLM_V41_FREE_SLOTS");
      const uint64_t per_class = env ? std::strtoull(env, nullptr, 10) : 12;
      // Async eviction keeps async_free_ empty GPU slots per layer, so as
      // many more experts per layer live on the host.
      std::unordered_map<uint64_t, uint64_t> layers_of;
      for (const auto& layer : layers_) ++layers_of[layer.config.expert_bytes];
      uint64_t at = exchange_offset_;
      for (const auto& layer : layers_) {
        const uint64_t bytes = layer.config.expert_bytes;
        if (free_.count(bytes)) continue;
        auto& list = free_[bytes];
        const uint64_t count =
            per_class + uint64_t(async_free_) * layers_of[bytes];
        for (uint64_t i = 0; i < count; ++i) {
          if (at + bytes > exchange_offset_ + exchange_bytes)
            throw std::invalid_argument(
                "free-slot exchange exceeds the reserve");
          list.push_back(at);
          at += bytes;
        }
      }
    }
  }

  // Exact truncated verification (cutoff > 0): a cold expert whose earliest
  // route position (route / top_k = verify token) is >= cutoff is not fetched
  // and truncates the verify block there; experts used only at positions
  // >= trunc are never fetched. trunc resets when the layer index wraps.
  int32_t trunc() const { return trunc_; }

  std::vector<SharedExpertCopy> plan(size_t layer_index, uint64_t sequence,
                                     const std::vector<int32_t>& raw_routes,
                                     uint64_t now_ns, uint64_t timeout_ns,
                                     int32_t cutoff = 0, int32_t top_k = 6,
                                     uint64_t round = 0) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (pending_)
      throw std::runtime_error("shared expert plan already pending");
    auto& layer = layers_.at(layer_index);
    // Deferred drop candidates (bit 30, VLLM_V41_SKIP_DEFER): an expert a
    // background load already placed (ghost) or whose eviction is still in
    // its slot is adopted without a copy; the others stay dropped this time
    // and are queued for a background load, so later uses compute exactly.
    std::vector<int32_t> routes(raw_routes);
    background_.clear();
    for (auto& r : routes) {
      if (r == kSkipRoute || !(r & kDeferFlag)) continue;
      const int32_t e = r & ~kDeferFlag;
      if (e < 0 || e >= layer.config.experts)
        throw std::invalid_argument("shared expert deferred id outside layer");
      if (layer.ghost.count(e) || layer.evicting.count(e)) {
        r = e;
        ++stat_adopt_;
      } else {
        r = kSkipRoute;
        if (!layer.prefetching.count(e) &&
            std::find(background_.begin(), background_.end(), e) ==
                background_.end())
          background_.push_back(e);
      }
    }
    if (routes.empty() || routes.size() > max_routes_)
      throw std::invalid_argument(
          "shared expert routes exceed request capacity");
    std::unordered_set<int32_t> wanted;
    for (auto expert : routes) {
      if (expert == kSkipRoute) continue;  // dropped on the GPU (lossy skip)
      if (expert < 0 || expert >= layer.config.experts)
        throw std::invalid_argument("shared expert id outside layer");
      wanted.insert(expert);
    }
    if (clock_ == std::numeric_limits<uint64_t>::max()) {
      fence_.stop();
      throw std::runtime_error("shared expert LRU sequence exhausted");
    }
    // The GPU numbers its forwards; truncation never carries across them.
    if (round != round_) {
      trunc_ = kNoTrunc;
      round_ = round;
    }
    std::unordered_set<int32_t> skipped;
    std::vector<uint64_t> saves_by_order;
    auto* pool = free_mode_ ? &free_.at(layer.config.expert_bytes) : nullptr;
    size_t available = pool ? pool->size() : 0;
    size_t free_slots = layer.free_slots.size();
    for (const auto& [e, s] : layer.ghost)
      if (wanted.count(e)) --free_slots;  // its rescue takes that free slot
    if (cutoff > 0 || trunc_ != kNoTrunc || free_mode_) {
      // earliest position per non-resident expert, visited in position order
      std::vector<std::pair<int32_t, int32_t>> need;
      for (size_t i = 0; i < routes.size(); ++i) {
        const int32_t e = routes[i], pos = int32_t(i) / top_k;
        if (e == kSkipRoute || layer.slot[e] >= 0) continue;
        auto it = std::find_if(need.begin(), need.end(),
                               [&](const auto& v) { return v.second == e; });
        if (it == need.end()) need.push_back({pos, e});
      }
      std::stable_sort(need.begin(), need.end());
      for (const auto& [pos, e] : need) {
        if (pos >= trunc_)
          skipped.insert(e);
        else if (cutoff > 0 && pos >= cutoff) {
          trunc_ = pos;
          skipped.insert(e);
        } else if (layer.evicting.count(e) || layer.ghost.count(e)) {
          // still intact in its old slot: reinstated without a copy
        } else if (free_slots > 0) {
          --free_slots;
        } else if (pool && available == 0) {
          // No free allocation of this class: exact truncation at the
          // earliest position that still needs this expert.
          if (pos == 0)
            throw std::runtime_error(
                "free-slot exchange cannot hold verify position 0");
          trunc_ = pos;
          skipped.insert(e);
        } else if (pool) {
          --available;
        }
      }
    }
    std::vector<SharedExpertCopy> copies;
    std::vector<SharedExpertRange> inputs, outputs;
    std::unordered_set<int32_t> evicting;
    // Route order is stable; unordered_set is used only for membership.
    for (auto incoming : routes) {
      if (incoming == kSkipRoute || layer.slot[incoming] >= 0 ||
          skipped.count(incoming) ||
          std::any_of(copies.begin(), copies.end(),
                      [&](const auto& op) { return op.incoming == incoming; }))
        continue;
      if (auto it = layer.evicting.find(incoming); it != layer.evicting.end()) {
        // Rescue: the GPU table dropped it but the save only reads the slot.
        copies.push_back(
            {it->second.slot, incoming, -1, kSharedNoOffset, kSharedNoOffset});
        ++stat_rescue_;
        continue;
      }
      if (auto it = layer.ghost.find(incoming); it != layer.ghost.end()) {
        // Victim cache: a free slot not yet reused still holds this expert.
        const int32_t slot = it->second;
        layer.free_slots.erase(
            std::find(layer.free_slots.begin(), layer.free_slots.end(), slot));
        copies.push_back(
            {slot, incoming, -1, kSharedNoOffset, kSharedNoOffset});
        ghost_rescues_.push_back({incoming, layer.cold[incoming]});
        layer.ghost.erase(it);
        ++stat_ghost_;
        if (prefetched_.erase({int32_t(layer_index), incoming}))
          ++stat_prefetch_hit_;
        continue;
      }
      const uint64_t bytes = layer.config.expert_bytes;
      const uint64_t load = uint64_t(layer.cold[incoming]);
      // Load-only into the least recently freed empty slot whose previous
      // expert (saved asynchronously) is not routed here; its ghost is dropped.
      auto pick = layer.free_slots.end();
      for (auto f = layer.free_slots.begin(); f != layer.free_slots.end();
           ++f) {
        auto g = std::find_if(layer.ghost.begin(), layer.ghost.end(),
                              [&](const auto& v) { return v.second == *f; });
        if (g == layer.ghost.end() || !wanted.count(g->first)) {
          pick = f;
          break;
        }
      }
      if (pick != layer.free_slots.end()) {
        const int32_t slot = *pick;
        layer.free_slots.erase(pick);
        for (auto g = layer.ghost.begin(); g != layer.ghost.end(); ++g)
          if (g->second == slot) {
            prefetched_.erase({int32_t(layer_index), g->first});
            layer.ghost.erase(g);
            break;
          }
        copies.push_back({slot, incoming, -1, load, kSharedNoOffset});
        inputs.push_back({load, bytes});
        ++stat_free_;
        continue;
      }
      if (pool && pool->empty())
        throw std::runtime_error("shared expert exchange pool exhausted");
      ++stat_exchange_;
      int32_t victim = -1;
      for (int32_t expert = 0; expert < layer.config.experts; ++expert) {
        if (layer.slot[expert] < 0 || wanted.count(expert) ||
            evicting.count(expert))
          continue;
        if (victim < 0 || layer.use[expert] < layer.use[victim])
          victim = expert;
      }
      if (victim < 0)
        throw std::runtime_error("no evictable shared expert slot");
      // In-place exchange (VLLM_V41_SWAP_INPLACE): devices write the victim
      // into the incoming expert's own cold allocation after every rank has
      // read it (device-side barrier), so finish() needs no host compaction.
      uint64_t save;
      if (pool) {
        save = pool->back();
        pool->pop_back();
      } else {
        save = inplace_ ? load : exchange_offset_ + copies.size() * bytes;
      }
      copies.push_back({layer.slot[victim], incoming, victim, load, save});
      inputs.push_back({load, bytes});
      outputs.push_back({save, bytes});
      evicting.insert(victim);
    }
    // Refill this layer's empty slots in the background: LRU victims leave
    // the GPU table with this plan's commit; every rank's job lane then saves
    // its exclusive bytes with copy-engine DMA (no SM interference) and the
    // slot becomes free once all ranks report completion.
    evicts_.clear();
    if (async_free_ > 0 && pool) {
      size_t have = layer.free_slots.size() + layer.evicting.size();
      while (have < size_t(async_free_) && pool->size() > kEvictReserve &&
             jobs_[0].size() + jobs_[1].size() + evicts_.size() <
                 kMaxPendingJobs &&
             copies.size() + evicts_.size() < kMaxPlanOps) {
        int32_t victim = -1;
        for (int32_t expert = 0; expert < layer.config.experts; ++expert) {
          if (layer.slot[expert] < 0 || wanted.count(expert) ||
              evicting.count(expert))
            continue;
          if (victim < 0 || layer.use[expert] < layer.use[victim])
            victim = expert;
        }
        if (victim < 0) break;
        evicts_.push_back(
            {layer.slot[victim], -1, victim, kSharedNoOffset, pool->back()});
        ++stat_evict_;
        pool->pop_back();
        evicting.insert(victim);
        ++have;
      }
    }
    fence_.begin(sequence, inputs, outputs, now_ns, timeout_ns);
    ++clock_;
    for (auto expert : routes)
      if (expert != kSkipRoute) layer.use[expert] = clock_;
    pending_ = true;
    pending_layer_ = layer_index;
    copies_ = copies;
    return copies;
  }

  // Dropped experts of the last plan to load in the background.
  std::vector<int32_t> take_background() {
    std::lock_guard<std::mutex> guard(mutex_);
    return std::exchange(background_, {});
  }

  // Evictions published with the pending plan (after its copies).
  std::vector<SharedExpertCopy> evicts() {
    std::lock_guard<std::mutex> guard(mutex_);
    return evicts_;
  }

  // Jobs created by the last finish(), to publish to the job lanes.
  std::vector<SharedExpertJob> take_jobs() {
    std::lock_guard<std::mutex> guard(mutex_);
    return std::exchange(new_jobs_, {});
  }

  // Every rank completed jobs of this kind with ordinal < done.
  void complete_jobs(int kind, uint64_t done) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto& queue = jobs_[kind];
    while (!queue.empty() && queue.front().ordinal < done) {
      const auto job = queue.front();
      queue.pop_front();
      auto& layer = layers_.at(job.layer);
      if (kind == 1 && demand_.erase(job.ordinal))
        continue;  // committed by its plan
      if (kind == 1) {
        // Prefetch landed on every rank: the slot is free, holding a ghost
        // of the expert while its host copy is still canonical.
        auto it = layer.prefetching.find(job.expert);
        if (it != layer.prefetching.end() && it->second == job.ordinal)
          layer.prefetching.erase(it);
        layer.free_slots.push_back(job.slot);
        if (layer.slot[job.expert] < 0 &&
            layer.cold[job.expert] == int64_t(job.offset) &&
            !layer.ghost.count(job.expert) &&
            !layer.prefetching.count(job.expert)) {
          layer.ghost[job.expert] = job.slot;
          prefetched_.insert({job.layer, job.expert});
        }
        continue;
      }
      auto it = layer.evicting.find(job.expert);
      if (it != layer.evicting.end() && it->second.job == job.id) {
        layer.cold[job.expert] = int64_t(job.offset);
        layer.evicting.erase(it);
        // A demand load that fell back to the SM copy may still land later;
        // its slot stays out of the free pool until that DMA completes.
        deferred_.push_back({job.layer, job.slot, job.expert, job.offset});
      } else {
        // Rescued while saving: the copy is stale, recycle its allocation.
        free_.at(layer.config.expert_bytes).push_back(job.offset);
      }
    }
    for (size_t i = 0; i < deferred_.size();) {
      const auto d = deferred_[i];
      const bool pending =
          std::any_of(jobs_[1].begin(), jobs_[1].end(), [&](const auto& j) {
            return j.layer == d.layer && j.slot == d.slot;
          });
      if (pending) {
        ++i;
        continue;
      }
      auto& layer = layers_.at(d.layer);
      layer.free_slots.push_back(d.slot);
      if (layer.slot[d.expert] < 0 &&
          layer.cold[d.expert] == int64_t(d.offset) &&
          !layer.ghost.count(d.expert) && !layer.prefetching.count(d.expert))
        layer.ghost[d.expert] = d.slot;
      deferred_[i] = deferred_.back();
      deferred_.pop_back();
    }
  }

  // Demand loads by copy-engine DMA (VLLM_V41_DMA_DEMAND): the pending
  // plan's free-slot loads become lane jobs. Returns, per copy, the job's
  // (ring id, ordinal), or (~0, ~0) for copies the GPU performs itself.
  std::vector<std::pair<uint64_t, uint64_t>> demand_jobs() {
    std::lock_guard<std::mutex> guard(mutex_);
    std::vector<std::pair<uint64_t, uint64_t>> out(
        copies_.size(), {~uint64_t{0}, ~uint64_t{0}});
    if (!pending_ ||
        jobs_[0].size() + jobs_[1].size() + copies_.size() >= kMaxPendingJobs)
      return out;
    for (size_t i = 0; i < copies_.size(); ++i) {
      const auto& c = copies_[i];
      if (c.victim >= 0 || c.load_offset == kSharedNoOffset) continue;
      const SharedExpertJob job{next_job_++,
                                1,
                                int32_t(pending_layer_),
                                c.slot,
                                c.incoming,
                                c.load_offset,
                                next_ordinal_[1]++};
      demand_.insert(job.ordinal);
      jobs_[1].push_back(job);
      new_jobs_.push_back(job);
      out[i] = {job.id, job.ordinal};
    }
    return out;
  }

  uint64_t pending_jobs() {
    std::lock_guard<std::mutex> guard(mutex_);
    return jobs_[0].size() + jobs_[1].size();
  }

  // Predictive prefetch (between plans): copy predicted cold experts of a
  // later layer into its empty slots with the job lanes. Candidates are in
  // priority order. A landed prefetch becomes a ghost the plan reinstates
  // without a copy; a wrong prediction only spends uplink bandwidth.
  size_t prefetch(size_t layer_index, const std::vector<int32_t>& candidates,
                  size_t budget, size_t max_inflight) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (pending_) return 0;
    auto& layer = layers_.at(layer_index);
    size_t issued = 0;
    for (auto e : candidates) {
      if (issued >= budget || jobs_[1].size() >= max_inflight ||
          jobs_[0].size() + jobs_[1].size() >= kMaxPendingJobs)
        break;
      if (e < 0 || e >= layer.config.experts || layer.slot[e] >= 0 ||
          layer.cold[e] < 0 || layer.evicting.count(e) ||
          layer.ghost.count(e) || layer.prefetching.count(e))
        continue;
      // Oldest free slot; a prefetched ghost is only displaced by a newer one.
      auto pick = layer.free_slots.end();
      for (auto f = layer.free_slots.begin(); f != layer.free_slots.end();
           ++f) {
        auto g = std::find_if(layer.ghost.begin(), layer.ghost.end(),
                              [&](const auto& v) { return v.second == *f; });
        if (g == layer.ghost.end() ||
            !prefetched_.count({int32_t(layer_index), g->first})) {
          pick = f;
          break;
        }
      }
      if (pick == layer.free_slots.end()) break;
      const int32_t slot = *pick;
      layer.free_slots.erase(pick);
      for (auto g = layer.ghost.begin(); g != layer.ghost.end(); ++g)
        if (g->second == slot) {
          prefetched_.erase({int32_t(layer_index), g->first});
          layer.ghost.erase(g);
          break;
        }
      const SharedExpertJob job{next_job_++,
                                1,
                                int32_t(layer_index),
                                slot,
                                e,
                                uint64_t(layer.cold[e]),
                                next_ordinal_[1]++};
      layer.prefetching[e] = job.ordinal;
      jobs_[1].push_back(job);
      new_jobs_.push_back(job);
      ++stat_prefetch_;
      ++issued;
    }
    return issued;
  }

  // In-flight prefetches (ring id, ordinal) of experts routed here, which
  // the coordinator may wait for before planning.
  std::vector<std::pair<uint64_t, uint64_t>> inflight_for(
      size_t layer_index, const std::vector<int32_t>& routes) {
    std::lock_guard<std::mutex> guard(mutex_);
    std::vector<std::pair<uint64_t, uint64_t>> out;
    auto& layer = layers_.at(layer_index);
    for (auto e : routes) {
      auto it = layer.prefetching.find(e);
      if (it == layer.prefetching.end()) continue;
      for (const auto& job : jobs_[1])
        if (job.ordinal == it->second) {
          out.push_back({job.id, job.ordinal});
          break;
        }
    }
    return out;
  }

  // Requests resolved entirely on the GPU (every route resident) still count
  // as uses, so LRU victims stay the least recently routed experts.
  void touch(size_t layer_index, const int32_t* routes, size_t count) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto& layer = layers_.at(layer_index);
    if (clock_ == std::numeric_limits<uint64_t>::max()) return;
    ++clock_;
    for (size_t i = 0; i < count; ++i)
      if (routes[i] >= 0 && routes[i] < layer.config.experts)
        layer.use[routes[i]] = clock_;
  }

  void acknowledge(unsigned rank, uint64_t sequence, uint64_t now_ns) {
    fence_.acknowledge(rank, sequence, now_ns);
  }

  bool ready(uint64_t sequence, uint64_t now_ns) {
    return fence_.ready(sequence, now_ns);
  }

  void finish(uint64_t sequence, uint64_t now_ns) {
    std::lock_guard<std::mutex> guard(mutex_);
    fence_.finish(sequence, now_ns);
    auto& layer = layers_.at(pending_layer_);
    const auto bytes = layer.config.expert_bytes;
    for (auto copy : copies_) {
      // Every input lease and rank completion is resolved before overwriting
      // the old canonical input, including overlapping TP boundary readers.
      if (free_mode_) {
        if (copy.victim >= 0) {
          layer.cold[copy.victim] = int64_t(copy.save_offset);
          layer.slot[copy.victim] = -1;
        }
        if (copy.load_offset != kSharedNoOffset)
          free_.at(bytes).push_back(copy.load_offset);
        else
          layer.evicting.erase(copy.incoming);  // rescued (or a ghost)
        layer.cold[copy.incoming] = -1;
        layer.slot[copy.incoming] = copy.slot;
        continue;
      }
      if (copy.save_offset != copy.load_offset)
        std::memcpy(arena_ + copy.load_offset, arena_ + copy.save_offset,
                    bytes);
      layer.cold[copy.victim] = layer.cold[copy.incoming];
      layer.cold[copy.incoming] = -1;
      layer.slot[copy.incoming] = copy.slot;
      layer.slot[copy.victim] = -1;
    }
    // A rescued ghost's saved host copy is no longer canonical.
    for (const auto& [expert, at] : ghost_rescues_)
      if (at >= 0) free_.at(bytes).push_back(uint64_t(at));
    ghost_rescues_.clear();
    for (const auto& ev : evicts_) {
      const SharedExpertJob job{
          next_job_++,       0,         int32_t(pending_layer_),
          ev.slot,           ev.victim, ev.save_offset,
          next_ordinal_[0]++};
      layer.slot[ev.victim] = -1;
      layer.evicting[ev.victim] = {ev.slot, job.id};
      jobs_[0].push_back(job);
      new_jobs_.push_back(job);
    }
    evicts_.clear();
    copies_.clear();
    pending_ = false;
  }

  // Cold (needs a transfer): not resident, not reclaimable without a copy.
  bool cold(size_t layer_index, int32_t expert) {
    std::lock_guard<std::mutex> guard(mutex_);
    const auto& layer = layers_.at(layer_index);
    return layer.slot.at(expert) < 0 && !layer.ghost.count(expert) &&
           !layer.evicting.count(expert) && !layer.prefetching.count(expert);
  }

  int32_t slot(size_t layer, int32_t expert) {
    std::lock_guard<std::mutex> guard(mutex_);
    return layers_.at(layer).slot.at(expert);
  }

  uint64_t cold_offset(size_t layer, int32_t expert) {
    std::lock_guard<std::mutex> guard(mutex_);
    return cold_offset_locked(layer, expert);
  }

  std::pair<uint64_t, uint64_t> acquire(size_t layer, int32_t expert) {
    std::lock_guard<std::mutex> guard(mutex_);
    const uint64_t offset = cold_offset_locked(layer, expert);
    const uint64_t lease =
        fence_.acquire({offset, layers_.at(layer).config.expert_bytes});
    return {lease, offset};
  }

  std::vector<SharedExpertLayer> configs() {
    std::lock_guard<std::mutex> guard(mutex_);
    std::vector<SharedExpertLayer> out;
    for (const auto& layer : layers_) out.push_back(layer.config);
    return out;
  }

  void release(uint64_t lease) { fence_.release(lease); }
  void stop() { fence_.stop(); }

 private:
  uint64_t cold_offset_locked(size_t index, int32_t expert) {
    const auto& layer = layers_.at(index);
    const auto at = layer.cold.at(expert);
    if (at < 0) throw std::invalid_argument("expert is resident, not cold");
    return uint64_t(at);
  }

  struct Layer {
    SharedExpertLayer config;
    uint64_t offset;
    std::vector<int32_t> slot;
    std::vector<int64_t> cold;
    std::vector<uint64_t> use;
    struct Evicting {
      int32_t slot;
      uint64_t job;
    };
    std::vector<int32_t> free_slots;  // empty, saved on all ranks
    std::unordered_map<int32_t, Evicting> evicting;  // dropped, save in flight
    std::unordered_map<int32_t, int32_t> ghost;  // saved, still in a free slot
    std::unordered_map<int32_t, uint64_t>
        prefetching;  // load ordinal in flight
  };
  uint8_t* const arena_;
  const bool inplace_ = [] {
    const char* v = std::getenv("VLLM_V41_SWAP_INPLACE");
    return v != nullptr && v[0] == '1';
  }();
  const bool free_mode_ = [] {
    const char* v = std::getenv("VLLM_V41_SWAP_FREE");
    return v != nullptr && v[0] == '1';
  }();
  // VLLM_V41_ASYNC_EVICT=<n>: empty GPU slots kept per layer (free mode only).
  const int32_t async_free_ = [] {
    const char* v = std::getenv("VLLM_V41_ASYNC_EVICT");
    const char* f = std::getenv("VLLM_V41_SWAP_FREE");
    return v && f && f[0] == '1' ? std::atoi(v) : 0;
  }();
  static constexpr size_t kEvictReserve =
      4;  // host allocations kept for exchange
  static constexpr size_t kMaxPendingJobs = 160;  // < job ring entries
  static constexpr size_t kMaxPlanOps = 44;       // control page op capacity
  std::deque<SharedExpertJob> jobs_[2];
  std::unordered_set<uint64_t> demand_;
  struct Deferred {
    int32_t layer, slot, expert;
    uint64_t offset;
  };
  std::vector<Deferred> deferred_;
  uint64_t next_ordinal_[2] = {0, 0};
  struct PairHash {
    size_t operator()(const std::pair<int32_t, int32_t>& v) const {
      return std::hash<int64_t>()((int64_t(v.first) << 32) |
                                  uint32_t(v.second));
    }
  };
  std::unordered_set<std::pair<int32_t, int32_t>, PairHash> prefetched_;
  std::vector<SharedExpertJob> new_jobs_;
  std::vector<SharedExpertCopy> evicts_;
  std::vector<int32_t> background_;
  uint64_t next_job_ = 0;
  std::vector<std::pair<int32_t, int64_t>> ghost_rescues_;

 public:
  uint64_t stat_free_ = 0, stat_exchange_ = 0, stat_rescue_ = 0,
           stat_ghost_ = 0, stat_evict_ = 0, stat_prefetch_ = 0,
           stat_prefetch_hit_ = 0, stat_adopt_ = 0;

 private:
  std::unordered_map<uint64_t, std::vector<uint64_t>> free_;
  const unsigned max_routes_;
  static constexpr int32_t kNoTrunc = 1 << 20;

 public:
  static constexpr int32_t kSkipRoute = 0x7FFFFFFF;
  static constexpr int32_t kDeferFlag = 1 << 30;

 private:
  int32_t trunc_ = kNoTrunc;
  uint64_t round_ = 0;
  uint64_t exchange_offset_ = 0, clock_ = 0;
  size_t pending_layer_ = 0;
  bool pending_ = false;
  std::mutex mutex_;
  SharedExpertSwapState fence_;
  std::vector<Layer> layers_;
  std::vector<SharedExpertCopy> copies_;
};

}  // namespace vllm::gguf
