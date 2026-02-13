#include "qerasure/core/sim/erasure_simulator.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "internal/fast_rng.h"

namespace qerasure {

ErasureSimulator::ErasureSimulator(ErasureSimParams params)
    : params_(std::move(params)),
      // Fixed fallback seed keeps behavior deterministic when seed is omitted.
      rng_state_(params_.seed.has_value() ? static_cast<std::uint64_t>(*params_.seed)
                                          : 0xD1B54A32D192ED03ULL) {
  validate_params();
  // Done once so inner simulation loops can iterate only participating qubits.
  precompute_active_qubits();
}

void ErasureSimulator::validate_params() const {
  if (params_.shots == 0) {
    throw std::invalid_argument("Number of shots must be greater than 0");
  }
  if (params_.qec_rounds == 0) {
    throw std::invalid_argument("Number of QEC rounds must be greater than 0");
  }

  // Conservative memory guard to avoid accidentally allocating huge sparse outputs.
  const double p_two = params_.noise.get(NoiseChannel::kTwoQubitErasure);
  const std::size_t expected_events_per_shot =
      static_cast<std::size_t>(p_two * params_.code.num_qubits() * params_.qec_rounds * 8);

  if (expected_events_per_shot * params_.shots > 10000000ULL) {
    throw std::invalid_argument(
        "Number of shots is likely to occupy a large proportion of memory");
  }
}

std::size_t ErasureSimulator::estimate_events_per_shot(const ErasureSimParams& params) {
  // Reserve heuristic tuned to reduce reallocations without overcommitting memory.
  return (params.code.num_qubits() * params.qec_rounds * 15) / 100;
}

std::uint64_t ErasureSimulator::next_random_u64() {
  // splitmix64 is fast and statistically adequate for Monte Carlo sampling here.
  return internal::splitmix64_next(&rng_state_);
}

// Convert probability in [0,1] into integer comparison threshold for branch-efficient sampling.
std::uint64_t ErasureSimulator::probability_to_threshold(double p) {
  if (p <= 0.0) {
    return 0;
  }
  if (p >= 1.0) {
    return std::numeric_limits<std::uint64_t>::max();
  }
  return static_cast<std::uint64_t>(p * static_cast<long double>(std::numeric_limits<std::uint64_t>::max()));
}

void ErasureSimulator::precompute_active_qubits() {
  const std::size_t num_qubits = params_.code.num_qubits();
  const std::vector<std::size_t>& partner_map = params_.code.partner_map();

  // Build per-step qubit lists that actually participate in a gate.
  for (std::size_t step = 0; step < 4; ++step) {
    const std::size_t base = step * num_qubits;
    auto& active = active_qubits_per_step_[step];
    active.clear();
    active.reserve(num_qubits);

    for (std::size_t qubit = 0; qubit < num_qubits; ++qubit) {
      // partner_map is flattened by [step][qubit].
      if (partner_map[base + qubit] != kNoPartner) {
        active.push_back(qubit);
      }
    }
  }
}

void ErasureSimulator::apply_step_erasures(std::vector<std::uint8_t>* current_state,
                                           std::vector<ErasureSimEvent>* shot_events,
                                           std::size_t* num_erasure_events,
                                           std::size_t step,
                                           std::uint64_t two_qubit_threshold) {
  if (two_qubit_threshold == 0) {
    return;
  }

  std::vector<std::uint8_t>& state = *current_state;
  std::vector<ErasureSimEvent>& events = *shot_events;

  // A qubit can only be newly erased if it is currently in state 0.
  for (const std::size_t qubit : active_qubits_per_step_[step]) {
    // Each active qubit receives one Bernoulli trial for two-qubit erasure.
    if (state[qubit] == 0 && next_random_u64() <= two_qubit_threshold) {
      state[qubit] = 1;
      events.push_back({qubit, EventType::ERASURE});
      ++(*num_erasure_events);
    }
  }
}

void ErasureSimulator::apply_check_and_reset(std::vector<std::uint8_t>* current_state,
                                             std::vector<ErasureSimEvent>* shot_events,
                                             std::size_t* num_erasure_events,
                                             std::uint64_t check_error_threshold) {
  std::vector<std::uint8_t>& state = *current_state;
  std::vector<ErasureSimEvent>& events = *shot_events;

  // End-of-round logic: either check fails, or successful check resets erased qubits.
  for (std::size_t qubit = 0; qubit < state.size(); ++qubit) {
    // Check-error event dominates reset: if check fails, we record CHECK_ERROR
    // and intentionally skip reset for that qubit at this round boundary.
    if (check_error_threshold != 0 && next_random_u64() <= check_error_threshold) {
      events.push_back({qubit, EventType::CHECK_ERROR});
      ++(*num_erasure_events);
    } else if (state[qubit] == 1) {
      events.push_back({qubit, EventType::RESET});
      state[qubit] = 0;
      ++(*num_erasure_events);
    }
  }
}

ErasureSimResult ErasureSimulator::simulate() {
  const std::size_t num_qubits = params_.code.num_qubits();
  const std::size_t num_timesteps = params_.qec_rounds * 4 + 1;
  const std::size_t estimated_events_per_shot = estimate_events_per_shot(params_);

  const std::uint64_t p_two_threshold =
      probability_to_threshold(params_.noise.get(NoiseChannel::kTwoQubitErasure));
  const std::uint64_t p_check_threshold =
      probability_to_threshold(params_.noise.get(NoiseChannel::kErasureCheckError));

  ErasureSimResult result;
  result.sparse_erasures.resize(params_.shots);
  result.erasure_timestep_offsets.resize(params_.shots);

  // state[q] = 1 means qubit q is currently erased and awaiting check/reset.
  std::vector<std::uint8_t> current_state(num_qubits, 0);

  for (std::size_t shot = 0; shot < params_.shots; ++shot) {
    std::size_t num_erasure_events = 0;
    std::vector<ErasureSimEvent>& shot_events = result.sparse_erasures[shot];
    std::vector<std::size_t>& offsets = result.erasure_timestep_offsets[shot];

    shot_events.clear();
    shot_events.reserve(estimated_events_per_shot);
    offsets.assign(num_timesteps + 1, 0);

    std::fill(current_state.begin(), current_state.end(), 0);

    for (std::size_t round = 0; round < params_.qec_rounds; ++round) {
      // Four schedule steps per round (matching rotated-surface extraction order).
      for (std::size_t step = 0; step < 4; ++step) {
        apply_step_erasures(&current_state, &shot_events, &num_erasure_events, step, p_two_threshold);

        // offsets[t+1] stores cumulative events after timestep t.
        offsets[round * 4 + step + 1] = num_erasure_events;

        if (step == 3) {
          // Round boundary: apply check/reset stage.
          apply_check_and_reset(&current_state, &shot_events, &num_erasure_events, p_check_threshold);
        }
      }
    }

    offsets[num_timesteps] = num_erasure_events;
  }

  return result;
}

}  // namespace qerasure
