#pragma once

#include <cstdint>
#include <tuple>
#include <vector>

#include "core/circuit/rail_surface_compile.h"
#include "stim/circuit/circuit.h"

namespace qerasure::simulator {

// Calibration batch that includes latent onset-op index per check event.
// `latent_onset_ops[shot * num_checks + check_event]` is -1 when no erasure is
// active at that check, otherwise the onset op index of the active erasure.
struct RailCalibrationSampleBatch {
  uint32_t num_shots = 0;
  uint32_t num_detectors = 0;
  uint32_t num_observables = 0;
  uint32_t num_checks = 0;

  std::vector<uint8_t> detector_samples;
  std::vector<uint8_t> observable_flips;
  std::vector<uint8_t> check_flags;
  std::vector<int32_t> latent_onset_ops;
};

class RailCalibrationSampler {
 public:
  explicit RailCalibrationSampler(const circuit::RailSurfaceCompiledProgram& program)
      : program_(program) {}

  RailCalibrationSampleBatch sample_syndromes(
      uint32_t num_shots,
      uint32_t seed,
      uint32_t num_threads = 1);

  std::tuple<stim::Circuit, std::vector<uint8_t>, std::vector<int32_t>> sample_exact_shot(
      uint32_t seed,
      uint32_t shot) const;

 private:
  const circuit::RailSurfaceCompiledProgram& program_;
};

}  // namespace qerasure::simulator
