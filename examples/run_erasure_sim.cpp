#include "qerasure/core/code/rotated_surface_code.h"
#include "qerasure/core/noise/noise_params.h"
#include "qerasure/core/sim/erasure_simulator.h"

#include <iostream>

int main() {
  qerasure::RotatedSurfaceCode code(5);
  qerasure::NoiseParams noise;
  noise.set(qerasure::NoiseChannel::kTwoQubitErasure, 0.01);
  noise.set(qerasure::NoiseChannel::kErasureCheckError, 0.02);

  qerasure::ErasureSimParams params(code, noise, 4, 100, 12345);
  qerasure::ErasureSimulator simulator(params);
  qerasure::ErasureSimResult result = simulator.simulate();

  std::cout << "Shots: " << params.shots << ", events in first shot: "
            << result.sparse_erasures.front().size() << "\n";
  return 0;
}
