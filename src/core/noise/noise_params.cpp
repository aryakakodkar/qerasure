#include "qerasure/core/noise/noise_params.h"

#include <array>
#include <stdexcept>

namespace qerasure {
namespace {

// Canonical text keys used by Python wrappers and user-facing configs.
constexpr std::array<std::string_view, static_cast<std::size_t>(NoiseChannel::kCount)> kChannelNames = {
    "p_single_qubit_depolarize",
    "p_two_qubit_depolarize",
    "p_measurement_error",
    "p_single_qubit_erasure",
    "p_two_qubit_erasure",
    "p_erasure_check_error",
};

}  // namespace

NoiseParams::NoiseParams() {
  // Start from a physically "clean" default model unless caller opts in.
  probabilities_.fill(0.0);
}

void NoiseParams::validate_probability(double prob) {
  // Probabilities are interpreted literally in Bernoulli sampling.
  if (prob < 0.0 || prob > 1.0) {
    throw std::invalid_argument("Probability must be between 0 and 1");
  }
}

std::size_t NoiseParams::to_index(NoiseChannel channel) {
  return static_cast<std::size_t>(channel);
}

void NoiseParams::set(NoiseChannel channel, double prob) {
  // Centralized validation keeps C++ and Python-facing APIs consistent.
  validate_probability(prob);
  probabilities_[to_index(channel)] = prob;
}

double NoiseParams::get(NoiseChannel channel) const {
  // O(1) typed lookup; used in simulation hot path.
  return probabilities_[to_index(channel)];
}

std::string_view NoiseParams::to_string(NoiseChannel channel) {
  const std::size_t idx = to_index(channel);
  if (idx >= kChannelNames.size()) {
    throw std::invalid_argument("Invalid noise channel enum value");
  }
  return kChannelNames[idx];
}

NoiseChannel NoiseParams::from_string(const std::string& channel) {
  // Linear scan is acceptable: channel count is fixed and tiny (6).
  for (std::size_t i = 0; i < kChannelNames.size(); ++i) {
    if (channel == kChannelNames[i]) {
      return static_cast<NoiseChannel>(i);
    }
  }
  throw std::invalid_argument("Invalid noise parameter channel: " + channel);
}

void NoiseParams::set(const std::string& channel, double prob) {
  // String path is a convenience shim for wrappers/config parsing.
  set(from_string(channel), prob);
}

double NoiseParams::get(const std::string& channel) const {
  // Avoids duplicating conversion/validation logic.
  return get(from_string(channel));
}

}  // namespace qerasure
