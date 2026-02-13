#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "qerasure/core/code/rotated_surface_code.h"
#include "qerasure/core/noise/noise_params.h"

namespace qerasure {

// Immutable configuration object for one simulator run.
struct ErasureSimParams {
  // Geometry + schedule metadata (number of qubits, gate partners per step, etc.).
  RotatedSurfaceCode code;

  // Noise model probabilities used during stochastic evolution.
  NoiseParams noise;

  // Number of QEC rounds; each round has 4 stabilizer interaction steps.
  std::size_t qec_rounds;

  // Number of independent Monte Carlo repetitions.
  std::size_t shots;

  // Optional deterministic seed for reproducible Monte Carlo output.
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
  // Qubit transitions from non-erased to erased during a gate step.
  ERASURE = 0,

  // Previously erased qubit is cleared at end-of-round check.
  RESET = 1,

  // Erasure-check measurement itself is flagged as erroneous.
  CHECK_ERROR = 2,
};

// Sparse event record at one qubit and one timestep slice.
struct ErasureSimEvent {
  std::size_t qubit_idx;
  EventType event_type;
};

// Sparse simulator output:
// - per-shot event vectors
// - per-shot timestep offsets into each event vector
struct ErasureSimResult {
  // For each shot: list of sparse events in chronological order.
  std::vector<std::vector<ErasureSimEvent>> sparse_erasures;

  // For each shot: prefix offsets so events at timestep t are
  // sparse_erasures[shot][offsets[t] : offsets[t+1]].
  std::vector<std::vector<std::size_t>> erasure_timestep_offsets;
};

// Monte Carlo simulator for erasure evolution over repeated QEC rounds.
class ErasureSimulator {
 public:
  explicit ErasureSimulator(ErasureSimParams params);
  ErasureSimResult simulate();

 private:
  // Immutable run-time input copied at construction.
  ErasureSimParams params_;

  // Internal RNG state for fast splitmix64 draws.
  std::uint64_t rng_state_;

  // Active qubits per schedule step to avoid scanning qubits with no gate partner.
  std::array<std::vector<std::size_t>, 4> active_qubits_per_step_;

  void validate_params() const;
  void precompute_active_qubits();

  // Apply stochastic two-qubit erasure events for one syndrome step.
  // `current_state[q]` acts as a one-bit latch for "currently erased".
  void apply_step_erasures(std::vector<std::uint8_t>* current_state,
                           std::vector<ErasureSimEvent>* shot_events,
                           std::size_t* num_erasure_events,
                           std::size_t step,
                           std::uint64_t two_qubit_threshold);

  // Apply erasure-check failures and reset successful erasures.
  // Called at end of each round (after step 3).
  void apply_check_and_reset(std::vector<std::uint8_t>* current_state,
                             std::vector<ErasureSimEvent>* shot_events,
                             std::size_t* num_erasure_events,
                             std::uint64_t check_error_threshold);

  // Heuristic pre-reserve size for sparse event vectors.
  static std::size_t estimate_events_per_shot(const ErasureSimParams& params);

  // One uniform draw in [0, 2^64-1].
  std::uint64_t next_random_u64();

  // Convert p in [0,1] to integer cutoff for branch-efficient sampling.
  static std::uint64_t probability_to_threshold(double p);
};

}  // namespace qerasure
