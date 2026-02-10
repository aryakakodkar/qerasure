#include "qerasure/code/code.h"
#include "qerasure/noise/noise.h"
#include "qerasure/simulators/erasure_simulator.h"

#include <stdexcept>

int main() {
  qerasure::RotatedSurfaceCode code(3);
  qerasure::NoiseParams noise;
  noise.set(qerasure::NoiseChannel::kTwoQubitErasure, 0.0);
  noise.set(qerasure::NoiseChannel::kErasureCheckError, 0.0);

  qerasure::ErasureSimParams params(code, noise, 2, 3, 7);
  qerasure::ErasureSimulator simulator(params);
  qerasure::ErasureSimResult result = simulator.simulate();

  if (result.sparse_erasures.size() != 3) {
    throw std::runtime_error("Unexpected number of shots in simulation result");
  }
  if (result.erasure_timestep_offsets.size() != 3) {
    throw std::runtime_error("Unexpected number of offset vectors in simulation result");
  }

  return 0;
}
