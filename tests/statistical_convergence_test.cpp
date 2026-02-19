#include "qerasure/core/code/rotated_surface_code.h"
#include "qerasure/core/lowering/lowering.h"
#include "qerasure/core/noise/noise_params.h"
#include "qerasure/core/sim/erasure_simulator.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {

std::size_t parse_or_default(char* arg, std::size_t fallback) {
  if (arg == nullptr) {
    return fallback;
  }
  return static_cast<std::size_t>(std::stoull(arg));
}

std::uint32_t parse_seed_or_default(char* arg, std::uint32_t fallback) {
  if (arg == nullptr) {
    return fallback;
  }
  return static_cast<std::uint32_t>(std::stoul(arg));
}

double max_double(double a, double b) { return a > b ? a : b; }

}  // namespace

int main(int argc, char* argv[]) {
  using namespace qerasure;

  const std::size_t shots = parse_or_default(argc > 1 ? argv[1] : nullptr, 2000);
  const std::size_t distance = parse_or_default(argc > 2 ? argv[2] : nullptr, 7);
  const std::size_t rounds = parse_or_default(argc > 3 ? argv[3] : nullptr, 1);
  const std::uint32_t seed = parse_seed_or_default(argc > 4 ? argv[4] : nullptr, 12345U);

  // Keep probabilities moderate so convergence is fast and stable in CI/local runs.
  constexpr double kPTwoQubitErasure = 0.08;
  constexpr double kPCheckError = 0.03;
  constexpr double kPLoweringSpread = 0.25;

  RotatedSurfaceCode code(distance);
  NoiseParams noise;
  noise.set(NoiseChannel::kTwoQubitErasure, kPTwoQubitErasure);
  noise.set(NoiseChannel::kErasureCheckError, kPCheckError);

  ErasureSimParams sim_params(code, noise, rounds, shots, seed, ErasureQubitSelection::DATA_QUBITS);
  ErasureSimulator simulator(sim_params);
  ErasureSimResult sim_result = simulator.simulate();

  // Simplistic lowering model: only one spread channel on X_1, no reset-induced lowering.
  SpreadProgram program;
  program.add_error_channel(kPLoweringSpread, {{PauliError::X_ERROR, PartnerSlot::X_1}});
  LoweredErrorParams reset_none{PauliError::NO_ERROR, 0.0};
  LoweringParams lowering_params(program, reset_none);
  Lowerer lowerer(code, lowering_params);
  LoweringResult lowering_result = lowerer.lower(sim_result);

  // Aggregate observed event counts from simulation and lowering outputs.
  std::size_t observed_erasures = 0;
  std::size_t observed_check_errors = 0;
  for (const auto& shot_events : sim_result.sparse_erasures) {
    for (const ErasureSimEvent& event : shot_events) {
      if (event.event_type == EventType::ERASURE) {
        ++observed_erasures;
      } else if (event.event_type == EventType::CHECK_ERROR) {
        ++observed_check_errors;
      }
    }
  }

  std::size_t observed_lowering = 0;
  for (const auto& shot_events : lowering_result.sparse_cliffords) {
    observed_lowering += shot_events.size();
  }

  // Build expected value model from code structure + configured probabilities.
  const std::size_t num_qubits = code.num_qubits();
  const std::size_t num_data_qubits = code.x_anc_offset();
  const std::vector<std::size_t>& partner_map = code.partner_map();
  const std::vector<std::pair<std::size_t, std::size_t>>& x_slots = code.data_to_x_ancilla_slots();

  double per_shot_expected_erasures = 0.0;
  double per_shot_var_erasures = 0.0;

  double per_shot_expected_lowering = 0.0;
  double per_shot_var_lowering = 0.0;

  for (std::size_t data_idx = 0; data_idx < num_data_qubits; ++data_idx) {
    // Data-qubit erasure probability for a round if the qubit starts not erased:
    // 1 - (1-p)^(#active gate opportunities).
    std::size_t active_steps = 0;
    for (std::size_t step = 0; step < 4; ++step) {
      if (partner_map[step * num_qubits + data_idx] != kNoPartner) {
        ++active_steps;
      }
    }

    const double p_erase_once =
        1.0 - std::pow(1.0 - kPTwoQubitErasure, static_cast<double>(active_steps));

    // Lowering here targets only X_1 slot, so expected lowering uses
    // probability qubit is erased by the timestep where X_1 interaction occurs.
    const std::size_t x1_partner = x_slots[data_idx].first;
    if (x1_partner == kNoPartner) {
      continue;
    }

    std::size_t x1_step = 0;
    bool found_x1_step = false;
    for (std::size_t step = 0; step < 4; ++step) {
      if (partner_map[step * num_qubits + data_idx] == x1_partner) {
        x1_step = step;
        found_x1_step = true;
        break;
      }
    }
    if (!found_x1_step) {
      continue;
    }

    std::size_t active_until_x1 = 0;
    for (std::size_t step = 0; step <= x1_step; ++step) {
      if (partner_map[step * num_qubits + data_idx] != kNoPartner) {
        ++active_until_x1;
      }
    }

    const double p_erased_by_x1 =
        1.0 - std::pow(1.0 - kPTwoQubitErasure, static_cast<double>(active_until_x1));

    // Round-by-round two-state model:
    // p_erased_start = probability qubit is already erased at start of round.
    // CHECK_ERROR can block reset, so erased state can persist across rounds.
    double p_erased_start = 0.0;
    for (std::size_t r = 0; r < rounds; ++r) {
      // Erasure events only happen when entering the round not erased.
      const double p_erasure_event_round = (1.0 - p_erased_start) * p_erase_once;
      per_shot_expected_erasures += p_erasure_event_round;
      per_shot_var_erasures += p_erasure_event_round * (1.0 - p_erasure_event_round);

      // Lowering on X_1 can happen if erased at X_1 step:
      // - already erased at round start, or
      // - newly erased before/at X_1 this round.
      const double p_erased_at_x1 =
          p_erased_start + (1.0 - p_erased_start) * p_erased_by_x1;
      const double p_lowering_round = p_erased_at_x1 * kPLoweringSpread;
      per_shot_expected_lowering += p_lowering_round;
      per_shot_var_lowering += p_lowering_round * (1.0 - p_lowering_round);

      // End-of-round erased-state transition:
      // if erased by end of round and CHECK_ERROR happens, erasure persists.
      const double p_erased_end_before_check =
          p_erased_start + (1.0 - p_erased_start) * p_erase_once;
      p_erased_start = p_erased_end_before_check * kPCheckError;
    }
  }

  const double expected_erasures = static_cast<double>(shots) * per_shot_expected_erasures;
  const double expected_check_errors =
      static_cast<double>(shots) * static_cast<double>(rounds) * static_cast<double>(num_qubits) *
      kPCheckError;
  const double expected_lowering = static_cast<double>(shots) * per_shot_expected_lowering;

  const double sigma_erasures = std::sqrt(static_cast<double>(shots) * per_shot_var_erasures);
  const double sigma_check_errors =
      std::sqrt(static_cast<double>(shots) * static_cast<double>(rounds) * static_cast<double>(num_qubits) * kPCheckError *
                (1.0 - kPCheckError));
  const double sigma_lowering = std::sqrt(static_cast<double>(shots) * per_shot_var_lowering);

  // Tolerance is max(5 sigma, 3% relative), balancing probabilistic noise and robustness.
  const double tol_erasures = max_double(5.0 * sigma_erasures, 0.03 * expected_erasures);
  const double tol_check = max_double(5.0 * sigma_check_errors, 0.03 * expected_check_errors);
  const double tol_lowering = max_double(5.0 * sigma_lowering, 0.03 * expected_lowering);

  const double diff_erasures = std::abs(static_cast<double>(observed_erasures) - expected_erasures);
  const double diff_check = std::abs(static_cast<double>(observed_check_errors) - expected_check_errors);
  const double diff_lowering = std::abs(static_cast<double>(observed_lowering) - expected_lowering);

  std::cout << "Shots: " << shots << " Distance: " << distance << " Rounds: " << rounds << "\n";
  std::cout << "Erasures: observed=" << observed_erasures << " expected=" << expected_erasures
            << " tolerance=" << tol_erasures << "\n";
  std::cout << "CheckErrors: observed=" << observed_check_errors << " expected=" << expected_check_errors
            << " tolerance=" << tol_check << "\n";
  std::cout << "Lowering: observed=" << observed_lowering << " expected=" << expected_lowering
            << " tolerance=" << tol_lowering << "\n";

  // Fail fast when observed frequencies drift outside expected probability bands.
  if (diff_erasures > tol_erasures) {
    throw std::runtime_error("Erasure count did not converge to expected probability band");
  }
  if (diff_check > tol_check) {
    throw std::runtime_error("Check-error count did not converge to expected probability band");
  }
  if (diff_lowering > tol_lowering) {
    throw std::runtime_error("Lowering count did not converge to expected probability band");
  }

  return 0;
}
