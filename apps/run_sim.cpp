#include "qerasure/code/code.h"
#include "qerasure/noise/noise.h"
#include "qerasure/simulators/erasure_simulator.h"

#include <iostream>

int main() {
  qerasure::RotatedSurfaceCode code(3);
  qerasure::NoiseParams noise;

  noise.set(qerasure::NoiseChannel::kTwoQubitErasure, 0.01);
  noise.set(qerasure::NoiseChannel::kErasureCheckError, 0.05);

  qerasure::ErasureSimParams params(code, noise, 10, 1000, 12345);
  qerasure::ErasureSimulator simulator(params);
  qerasure::ErasureSimResult result = simulator.simulate();

  std::size_t total_events = 0;
  for (const auto& shot : result.sparse_erasures) {
    total_events += shot.size();
  }

  std::cout << "Simulated shots: " << params.shots << "\n";
  std::cout << "Total sparse events: " << total_events << "\n";
  return 0;
}
