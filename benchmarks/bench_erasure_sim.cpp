#include "qerasure/core/code/rotated_surface_code.h"
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

  qerasure::ErasureSimParams params(code, noise, rounds, shots, seed);
  qerasure::ErasureSimulator simulator(params);

  const auto start = std::chrono::steady_clock::now();
  const qerasure::ErasureSimResult result = simulator.simulate();
  const auto end = std::chrono::steady_clock::now();

  (void)result;
  const std::chrono::duration<double> elapsed = end - start;
  const double shots_per_sec = static_cast<double>(shots) / elapsed.count();

  std::cout << "ElapsedSeconds: " << elapsed.count() << "\n";
  std::cout << "Shots/sec: " << shots_per_sec << "\n";
  return 0;
}
