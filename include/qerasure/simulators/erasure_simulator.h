#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

#include "qerasure/code/code.h"
#include "qerasure/noise/noise.h"

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
  std::mt19937 gen_;
  std::uniform_real_distribution<double> dist_;
  std::array<std::vector<std::size_t>, 4> active_qubits_per_step_;

  void validate_params() const;
  void precompute_active_qubits();

  void apply_step_erasures(std::vector<std::uint8_t>* current_state,
                           std::vector<ErasureSimEvent>* shot_events,
                           std::size_t* num_erasure_events,
                           std::size_t step) ;

  void apply_check_and_reset(std::vector<std::uint8_t>* current_state,
                             std::vector<ErasureSimEvent>* shot_events,
                             std::size_t* num_erasure_events);

  static std::size_t estimate_events_per_shot(const ErasureSimParams& params);
};

}  // namespace qerasure
