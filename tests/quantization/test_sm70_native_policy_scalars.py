# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile the actual native header; exercise libc dialects and nested owners."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_prepared_native_scalar_dialects_and_scope_lifetime(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    assert compiler is not None
    source = tmp_path / "policy.cpp"
    source.write_text(r"""#include "sm70_policy.h"
#include <cassert>
#include <thread>
using namespace vllm::sm70;
std::string token(const std::string& raw) {
  std::string result = "sm70:1:";
  for (size_t i = 0; i < policy_size; ++i)
    result += std::to_string(raw.size()) + ":" + raw;
  return result;
}
int main() {
  const auto field = PolicyField::fp8_qpn8_m16;
  for (const auto& raw : {"", "0", "1", "2", "true", " -5junk", "01", "1 ", "\x1f"}) {
    const bool unset = std::string(raw) == "\x1f";
    const auto encoded = token(raw);
    prepare_native_policy(encoded);
    setenv("VLLM_SM70_FP8_QPN8_M16", "not the captured value", 1);
    {
      PolicyScope outer(encoded);
      assert(policy_atoi(field, -5, true) == (unset ? -5 : std::atoi(raw)));
      assert(policy_exact_one(field, true, true) == (unset || std::string(raw) == "1"));
      {
        PolicyScope inner(token("7"));
        assert(policy_atoi(field) == 7);
      }
      assert(policy_atoi(field, -5) == (unset ? -5 : std::atoi(raw)));
      std::thread worker([&] {
        PolicyScope other(token("12"));
        assert(policy_atoi(field) == 12);
      });
      worker.join();
      assert(policy_atoi(field, -5) == (unset ? -5 : std::atoi(raw)));
    }
    assert(active_policy == nullptr && active_prepared_policy == nullptr);
  }
  auto targets = parse_gemm_targets(
      "no-separator;shape|bad;shape|16x128x32:12:0:1@kernel;"
      "other|8,256,64,10,0,0 name");
  assert(targets.size() == 3 && !targets[0].valid);
  assert(targets[1].valid && targets[1].descriptor == "shape");
  assert(targets[1].cta_m == 16 && targets[1].splits == 12);
  assert(targets[1].require_mgroup == 1 && targets[1].name_contains == "kernel");
  assert(targets[2].name_contains == "name");
  assert(parse_dispatch_override("") == DispatchOverride::Unset);
  assert(parse_dispatch_override(" default") == DispatchOverride::Invalid);
  assert(parse_dispatch_override("reuse") == DispatchOverride::Reuse);
  // An AOT slot rebinds to the current engine, even with identical computation
  // fingerprints and a different diagnostic budget. Nested owners restore TLS.
  auto first = std::make_shared<RuntimeState>();
  auto second = std::make_shared<RuntimeState>();
  const std::string slot = "sm70:slot:kernel_config.sm70_fp8.native";
  first->policies[slot] = &prepared_policy(token("1"));
  second->policies[slot] = &prepared_policy(token("2"));
  enter_runtime(first);
  assert(++diagnostic_counter("same") == 1);
  { PolicyScope scope(slot); assert(policy_atoi(field) == 1); }
  enter_runtime(second);
  assert(++diagnostic_counter("same") == 1);
  { PolicyScope scope(slot); assert(policy_atoi(field) == 2); }
  exit_runtime(second);
  assert(++diagnostic_counter("same") == 2);
  exit_runtime(first);
  first->close();
  enter_runtime(second);
  assert(++diagnostic_counter("same") == 2);
  exit_runtime(second);
  second->close();
  bool closed = false;
  try { enter_runtime(first); } catch (const std::runtime_error&) { closed = true; }
  assert(closed && active_runtime == nullptr);
  // The independent compatibility entry keeps its former per-call behavior.
  setenv("VLLM_SM70_FP8_QPN8_M16", "1", 1);
  assert(policy_exact_one(field, true, true));
  setenv("VLLM_SM70_FP8_QPN8_M16", "", 1);
  assert(!policy_exact_one(field, true, true));
}
""")
    root = Path(__file__).parents[2]
    binary = tmp_path / "policy"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-pthread",
            "-I",
            str(root / "csrc"),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True)


def test_marlin_prepared_parser_and_engine_binding(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    assert compiler is not None
    source = tmp_path / "marlin.cpp"
    source.write_text(r"""#include "sm70_policy.h"
#include <cassert>
using namespace vllm::sm70;
int main() {
  for (const auto& raw : {"", "1", " +08", "1 ", "2junk", "0", "16",
                          "4294967297", "\x1f"}) {
    MarlinOverrides parsed;
    parsed.parse("DENSE", "\x1f", raw, "\x1f");
    if (std::string(raw) == "" || std::string(raw) == "\x1f") {
      assert(!parsed.active && !parsed.split_k);
    } else {
      char* end = nullptr;
      int value = static_cast<int>(std::strtol(raw, &end, 10));
      bool valid = end != raw && *end == '\0' &&
                   (value == 1 || value == 2 || value == 4 || value == 8);
      assert(parsed.errors[1].empty() == valid);
      if (valid) assert(*parsed.split_k == value);
    }
  }
  MarlinOverrides good, bad;
  good.parse("MOE", " +32x128x32x4x32x32x32", "8", "lane_vectors");
  assert(good.geometry && (*good.geometry)[0] == 32 && *good.split_k == 8);
  assert(good.vector_words && !*good.vector_words);
  bad.parse("DENSE", "bad", "3", "false");
  assert(!bad.errors[0].empty() && !bad.errors[1].empty() && !bad.errors[2].empty());
  assert(bad.errors[1] == "Invalid SM70_MARLIN_DENSE_SPLIT_K value '3'. "
                          "Expected one of 1, 2, 4, or 8.");
  assert(bound_marlin_policy(false) == nullptr);
  PreparedPolicy first{}, second{};
  first.marlin_dense = good;
  second.marlin_dense = bad;
  auto a = std::make_shared<RuntimeState>(), b = std::make_shared<RuntimeState>();
  const std::string slot = "sm70:slot:kernel_config.sm70_marlin";
  a->policies[slot] = &first;
  b->policies[slot] = &second;
  enter_runtime(a);
  setenv("SM70_MARLIN_DENSE_SPLIT_K", "unread", 1);
  assert(bound_marlin_policy(false)->split_k == 8);
  enter_runtime(b);
  assert(!bound_marlin_policy(false)->errors[0].empty());
  exit_runtime(b);
  assert(bound_marlin_policy(false)->split_k == 8);
  exit_runtime(a);
}
""")
    root = Path(__file__).resolve().parents[2]
    binary = tmp_path / "marlin"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-pthread",
            "-I",
            str(root / "csrc"),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
