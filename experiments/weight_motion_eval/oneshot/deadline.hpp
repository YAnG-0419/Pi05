// Per-command deadline guard for the isolated Pi05 controller overlay.
// No ROS dependencies: the same guard is exercised by native unit tests.
#pragma once
#include <charconv>
#include <cstdint>
#include <string>
#include <string_view>

namespace pi05 {
struct Envelope {
  std::string run;
  std::uint64_t sequence{0};
  std::int64_t expiry{0};
};

inline bool decode(std::string_view value, Envelope& out) {
  constexpr std::string_view prefix = "pi05v1/";
  if (value.substr(0, prefix.size()) != prefix) return false;
  value.remove_prefix(prefix.size());
  const auto slash = value.find('/');
  if (slash != 32) return false;
  const auto run = value.substr(0, slash);
  if (run.find_first_not_of("0123456789abcdef") != std::string_view::npos) return false;
  value.remove_prefix(slash + 1);
  const auto next = value.find('/');
  if (next == std::string_view::npos || next == 0) return false;
  const auto sequence = value.substr(0, next);
  value.remove_prefix(next + 1);
  if (value.empty()) return false;
  const auto s = std::from_chars(sequence.data(), sequence.data() + sequence.size(), out.sequence);
  const auto e = std::from_chars(value.data(), value.data() + value.size(), out.expiry);
  if (s.ec != std::errc{} || s.ptr != sequence.data() + sequence.size() ||
      e.ec != std::errc{} || e.ptr != value.data() + value.size()) return false;
  out.run = std::string(run);
  return out.expiry > 0;
}

inline bool fresh(std::int64_t created, std::int64_t expiry, std::int64_t now) {
  // Subtract only after checking ordering, to avoid signed overflow.
  return created > 0 && expiry > created && now >= created && now < expiry &&
      expiry - created <= 20'000'001;
}
}  // namespace pi05
