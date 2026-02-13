#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <string_view>

namespace qerasure {

// Enumerates supported noise channels used by simulator and wrappers.
enum class NoiseChannel : std::size_t {
  kSingleQubitDepolarize = 0,
  kTwoQubitDepolarize,
  kMeasurementError,
  kSingleQubitErasure,
  kTwoQubitErasure,
  kErasureCheckError,
  kCount,
};

// Stores all noise probabilities as a fixed-size vector indexed by NoiseChannel.
//
// This avoids string-hash lookup on the simulator hot path while still exposing
// string-based helpers for Python/user convenience.
class NoiseParams {
 public:
  NoiseParams();

  // Typed channel accessors used by C++ simulation code.
  void set(NoiseChannel channel, double prob);
  double get(NoiseChannel channel) const;

  // String compatibility accessors (primarily for external wrappers).
  void set(const std::string& channel, double prob);
  double get(const std::string& channel) const;

  std::array<double, static_cast<std::size_t>(NoiseChannel::kCount)> values() const {
    return probabilities_;
  }

  static std::string_view to_string(NoiseChannel channel);
  static NoiseChannel from_string(const std::string& channel);

 private:
  // Probability vector indexed by NoiseChannel values.
  std::array<double, static_cast<std::size_t>(NoiseChannel::kCount)> probabilities_{};

  static std::size_t to_index(NoiseChannel channel);
  static void validate_probability(double prob);
};

}  // namespace qerasure
