#include "qerasure/core/code/rotated_surface_code.h"
#include "qerasure/core/lowering/lowering.h"
#include "qerasure/core/noise/noise_params.h"
#include "qerasure/core/sim/erasure_simulator.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

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

}  // namespace

int main(int argc, char* argv[]) {
  const std::size_t shots = parse_or_default(argc > 1 ? argv[1] : nullptr, 10000);
  const std::size_t distance = parse_or_default(argc > 2 ? argv[2] : nullptr, 15);
  const std::size_t rounds = parse_or_default(argc > 3 ? argv[3] : nullptr, 15);
  const std::uint32_t seed = parse_seed_or_default(argc > 4 ? argv[4] : nullptr, 12345U);

  qerasure::RotatedSurfaceCode code(distance);
  qerasure::NoiseParams noise;
  noise.set(qerasure::NoiseChannel::kTwoQubitErasure, 0.01);
  noise.set(qerasure::NoiseChannel::kErasureCheckError, 0.05);

  qerasure::ErasureSimParams sim_params(code, noise, rounds, shots, seed);
  qerasure::ErasureSimulator simulator(sim_params);

  qerasure::SpreadProgram default_data_program;
  default_data_program.add_error_channel(
      0.5, {{qerasure::PauliError::X_ERROR, qerasure::PartnerSlot::X_1}});
  default_data_program.add_error_channel(
      0.5, {{qerasure::PauliError::X_ERROR, qerasure::PartnerSlot::X_2}});
  default_data_program.add_error_channel(
      0.5, {{qerasure::PauliError::X_ERROR, qerasure::PartnerSlot::Z_1}});
  default_data_program.add_error_channel(
      0.5, {{qerasure::PauliError::X_ERROR, qerasure::PartnerSlot::Z_2}});
  qerasure::LoweredErrorParams reset_params{qerasure::PauliError::Z_ERROR, 1.0};
  qerasure::LoweringParams lowering_params(default_data_program, reset_params);

  qerasure::Lowerer lowerer(code, lowering_params);

  const auto start = std::chrono::steady_clock::now();
  const qerasure::ErasureSimResult sim_result = simulator.simulate();
  const qerasure::LoweringResult lowering_result = lowerer.lower(sim_result);
  const auto end = std::chrono::steady_clock::now();

  (void)sim_result;
  (void)lowering_result;
  const std::chrono::duration<double> elapsed = end - start;

  std::cout << "ElapsedSeconds: " << elapsed.count() << "\n";
  std::cout << "Shots/sec: " << (static_cast<double>(shots) / elapsed.count()) << "\n";
  return 0;
}
