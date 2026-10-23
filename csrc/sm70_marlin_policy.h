// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstdlib>
#include <optional>
#include <string>

namespace vllm::sm70 {
// Preserve Marlin's strtol -> int conversion, including whitespace/sign and
// overflow behavior. Both prepared and standalone entrypoints use this parser.
inline bool marlin_parse_int_component(const char*& cursor, int& value) {
  char* end = nullptr;
  const long parsed = std::strtol(cursor, &end, 10);
  if (end == cursor) return false;
  value = static_cast<int>(parsed);
  cursor = end;
  return true;
}

inline bool marlin_parse_geometry(const char* value,
                                  std::array<int, 7>& fields) {
  if (!value || !value[0]) return false;
  const char* cursor = value;
  for (int i = 0; i < 7; ++i) {
    if (!marlin_parse_int_component(cursor, fields[i])) return false;
    if (i < 6 && *cursor++ != 'x') return false;
  }
  return *cursor == '\0';
}

struct MarlinOverrides {
  std::optional<std::array<int, 7>> geometry;
  std::optional<int> split_k;
  std::optional<bool> vector_words;
  std::array<std::string, 3> errors;
  bool active = false;

  void parse(const char* domain, const std::string& raw_geometry,
             const std::string& raw_split, const std::string& raw_metadata) {
    const auto present = [](const std::string& value) {
      return value != "\x1f" && !value.empty();
    };
    const std::string prefix = std::string("SM70_MARLIN_") + domain + "_";
    if (present(raw_geometry)) {
      active = true;
      std::array<int, 7> fields{};
      if (marlin_parse_geometry(raw_geometry.c_str(), fields)) {
        geometry = fields;
      } else {
        errors[0] = "Invalid " + prefix + "CTA_GEOMETRY value '" +
                    raw_geometry +
                    "'. Expected "
                    "{CTA_M}x{CTA_N}x{CTA_K}x{Warps}x{WarpM}x{WarpN}x{WarpK}.";
      }
    }
    if (present(raw_split)) {
      active = true;
      const char* cursor = raw_split.c_str();
      int value = 0;
      if (marlin_parse_int_component(cursor, value) && *cursor == '\0' &&
          (value == 1 || value == 2 || value == 4 || value == 8)) {
        split_k = value;
      } else {
        errors[1] = "Invalid " + prefix + "SPLIT_K value '" + raw_split +
                    "'. Expected one of 1, 2, 4, or 8.";
      }
    }
    if (present(raw_metadata)) {
      active = true;
      if (raw_metadata == "vector_words" || raw_metadata == "lane_vectors") {
        vector_words = raw_metadata == "vector_words";
      } else {
        errors[2] = "Invalid " + prefix + "METADATA_CACHE value '" +
                    raw_metadata + "'. Expected vector_words or lane_vectors.";
      }
    }
  }
};
}  // namespace vllm::sm70
