#include "qerasure/core/noise/noise_params.h"

#include <array>
#include <stdexcept>

namespace qerasure {
namespace {

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
  probabilities_.fill(0.0);
}

void NoiseParams::validate_probability(double prob) {
  if (prob < 0.0 || prob > 1.0) {
    throw std::invalid_argument("Probability must be between 0 and 1");
  }
}

std::size_t NoiseParams::to_index(NoiseChannel channel) {
  return static_cast<std::size_t>(channel);
}

void NoiseParams::set(NoiseChannel channel, double prob) {
  validate_probability(prob);
  probabilities_[to_index(channel)] = prob;
}

double NoiseParams::get(NoiseChannel channel) const {
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
  for (std::size_t i = 0; i < kChannelNames.size(); ++i) {
    if (channel == kChannelNames[i]) {
      return static_cast<NoiseChannel>(i);
    }
  }
  throw std::invalid_argument("Invalid noise parameter channel: " + channel);
}

void NoiseParams::set(const std::string& channel, double prob) {
  set(from_string(channel), prob);
}

double NoiseParams::get(const std::string& channel) const {
  return get(from_string(channel));
}

}  // namespace qerasure
