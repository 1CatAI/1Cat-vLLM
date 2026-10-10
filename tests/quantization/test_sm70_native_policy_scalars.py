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
