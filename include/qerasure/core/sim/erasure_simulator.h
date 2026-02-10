#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "qerasure/core/code/rotated_surface_code.h"
#include "qerasure/core/noise/noise_params.h"

namespace qerasure {

struct ErasureSimParams {
  RotatedSurfaceCode code;
  NoiseParams noise;
  std::size_t qec_rounds;
  std::size_t shots;
  std::optional<std::uint32_t> seed;

  ErasureSimParams(RotatedSurfaceCode code_, NoiseParams noise_, std::size_t qec_rounds_,
                   std::size_t shots_, std::optional<std::uint32_t> seed_ = std::nullopt)
      : code(std::move(code_)),
        noise(std::move(noise_)),
        qec_rounds(qec_rounds_),
        shots(shots_),
        seed(seed_) {}
};

enum class EventType : std::uint8_t {
  ERASURE = 0,
  RESET = 1,
  CHECK_ERROR = 2,
};

struct ErasureSimEvent {
  std::size_t qubit_idx;
  EventType event_type;
};

struct ErasureSimResult {
  std::vector<std::vector<ErasureSimEvent>> sparse_erasures;
  std::vector<std::vector<std::size_t>> erasure_timestep_offsets;
};

class ErasureSimulator {
 public:
  explicit ErasureSimulator(ErasureSimParams params);
  ErasureSimResult simulate();

 private:
  ErasureSimParams params_;
  std::uint64_t rng_state_;
  std::array<std::vector<std::size_t>, 4> active_qubits_per_step_;

  void validate_params() const;
  void precompute_active_qubits();

  void apply_step_erasures(std::vector<std::uint8_t>* current_state,
                           std::vector<ErasureSimEvent>* shot_events,
                           std::size_t* num_erasure_events,
                           std::size_t step,
                           std::uint64_t two_qubit_threshold);

  void apply_check_and_reset(std::vector<std::uint8_t>* current_state,
                             std::vector<ErasureSimEvent>* shot_events,
                             std::size_t* num_erasure_events,
                             std::uint64_t check_error_threshold);

  static std::size_t estimate_events_per_shot(const ErasureSimParams& params);

  std::uint64_t next_random_u64();
  static std::uint64_t probability_to_threshold(double p);
};

}  // namespace qerasure
