#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <thread>

#include "core/circuit/compile.h"
#include "core/circuit/erasure_model.h"
#include "core/gen/surf.h"
#include "core/model/pauli_channel.h"
#include "core/simulator/stream_sampler.h"

int main(int argc, char** argv) {
  using namespace qerasure::circuit;    // NOLINT
  using namespace qerasure::gen;        // NOLINT
  using namespace qerasure::simulator;  // NOLINT

  constexpr uint32_t kDistance = 15;
  constexpr uint32_t kRounds = 15;
  constexpr double kErasureProb = 0.01;
  constexpr uint32_t kShots = 10'000;
  constexpr uint32_t kSeed = 12345;
  uint32_t threads = std::thread::hardware_concurrency();
  if (threads == 0) {
    threads = 1;
  }
  if (argc > 1) {
    threads = static_cast<uint32_t>(std::strtoul(argv[1], nullptr, 10));
  }

  SurfaceCodeRotated generator(kDistance);
  const ErasureCircuit erasure_circuit =
      generator.build_circuit(kRounds, kErasureProb, /*erasable_qubits=*/"ALL");

  ErasureModel model(
      /*max_persistence=*/3,
      /*onset=*/PauliChannel(0.18, 0.06, 0.06),
      /*reset=*/PauliChannel(0.04, 0.03, 0.03),
      /*spread=*/TQGSpreadModel(PauliChannel(0.12, 0.04, 0.04), PauliChannel(0.10, 0.05, 0.05)));
  model.check_false_negative_prob = 0.02;
  model.check_false_positive_prob = 0.01;

  const CompiledErasureProgram compiled(erasure_circuit, model);
  StreamSampler sampler(compiled);

  const auto t0 = std::chrono::steady_clock::now();
  sampler.sample_with_callback(
      kShots, kSeed,
      [](const stim::Circuit&, const std::vector<uint8_t>&) {
        // Intentionally empty callback for pure sampling+injection throughput measurement.
      },
      threads);
  const auto t1 = std::chrono::steady_clock::now();

  const double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
  const double shots_per_s = elapsed_s > 0.0 ? static_cast<double>(kShots) / elapsed_s : 0.0;

  std::cout << "bench_stream_sampler\n";
  std::cout << "distance: " << kDistance << "\n";
  std::cout << "rounds: " << kRounds << "\n";
  std::cout << "erasure_prob: " << kErasureProb << "\n";
  std::cout << "shots: " << kShots << "\n";
  std::cout << "threads: " << threads << "\n";
  std::cout << "elapsed_s: " << elapsed_s << "\n";
  std::cout << "shots_per_s: " << shots_per_s << "\n";

  return EXIT_SUCCESS;
}
