#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <string_view>

namespace qerasure {

enum class NoiseChannel : std::size_t {
  kSingleQubitDepolarize = 0,
  kTwoQubitDepolarize,
  kMeasurementError,
  kSingleQubitErasure,
  kTwoQubitErasure,
  kErasureCheckError,
  kCount,
};

class NoiseParams {
 public:
  NoiseParams();

  void set(NoiseChannel channel, double prob);
  double get(NoiseChannel channel) const;

  // Compatibility helpers for Python/legacy code paths.
  void set(const std::string& channel, double prob);
  double get(const std::string& channel) const;

  std::array<double, static_cast<std::size_t>(NoiseChannel::kCount)> values() const {
    return probabilities_;
  }

  static std::string_view to_string(NoiseChannel channel);
  static NoiseChannel from_string(const std::string& channel);

 private:
  std::array<double, static_cast<std::size_t>(NoiseChannel::kCount)> probabilities_{};

  static std::size_t to_index(NoiseChannel channel);
  static void validate_probability(double prob);
};

}  // namespace qerasure
