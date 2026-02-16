#include "qerasure/core/code/rotated_surface_code.h"
#include "qerasure/core/noise/noise_params.h"
#include "qerasure/core/sim/erasure_simulator.h"

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

  // With p=1 and data-qubits-only erasure selection, every ERASURE event must be on data qubits.
  qerasure::NoiseParams noisy;
  noisy.set(qerasure::NoiseChannel::kTwoQubitErasure, 1.0);
  noisy.set(qerasure::NoiseChannel::kErasureCheckError, 0.0);
  qerasure::ErasureSimParams data_only_params(
      code, noisy, 1, 1, 7, qerasure::ErasureQubitSelection::DATA_QUBITS);
  qerasure::ErasureSimulator data_only_sim(data_only_params);
  qerasure::ErasureSimResult data_only_result = data_only_sim.simulate();

  if (data_only_result.sparse_erasures.empty()) {
    throw std::runtime_error("Expected one shot in data-only simulation");
  }
  for (const auto& event : data_only_result.sparse_erasures[0]) {
    if (event.event_type == qerasure::EventType::ERASURE &&
        event.qubit_idx >= code.x_anc_offset()) {
      throw std::runtime_error("Found non-data erasure while in DATA_QUBITS selection mode");
    }
  }

  return 0;
}
