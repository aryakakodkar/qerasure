#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "core/circuit/rail_surface_compile.h"
#include "core/simulator/stream_sampler.h"
#include "stim/circuit/circuit.h"

namespace qerasure::simulator {

class RailStreamSampler {
 public:
  explicit RailStreamSampler(const circuit::RailSurfaceCompiledProgram& program)
      : program_(program) {}

  SyndromeSampleBatch sample_syndromes(
      uint32_t num_shots,
      uint32_t seed,
      uint32_t num_threads = 1);

  std::pair<stim::Circuit, std::vector<uint8_t>> sample_exact_shot(
      uint32_t seed,
      uint32_t shot) const;

 private:
  const circuit::RailSurfaceCompiledProgram& program_;
};

}  // namespace qerasure::simulator
