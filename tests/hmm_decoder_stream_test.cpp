#include <chrono>
#include <atomic>
#include <cstdint>
#include <iostream>
#include <stdexcept>

#include "core/circuit/compile.h"
#include "core/circuit/erasure_model.h"
#include "core/decode/surf_hmm_decoder.h"
#include "core/gen/surf.h"
#include "core/model/pauli_channel.h"
#include "core/simulator/stream_sampler.h"

int main() {
  using namespace qerasure::circuit;    // NOLINT
  using namespace qerasure::decode;     // NOLINT
  using namespace qerasure::gen;        // NOLINT
  using namespace qerasure::simulator;  // NOLINT

  constexpr uint32_t kDistance = 5;
  constexpr uint32_t kRounds = 5;
  constexpr double kErasureProb = 0.02;
  constexpr uint32_t kShots = 10'000;
  constexpr uint32_t kSeed = 12345;
  constexpr uint32_t kThreads = 4;

  SurfaceCodeRotated generator(kDistance);
  const ErasureCircuit erasure_circuit =
      generator.build_circuit(kRounds, kErasureProb, /*erasable_qubits=*/"ALL");

  ErasureModel model(
      /*max_persistence=*/2,
      /*onset=*/PauliChannel(0.18, 0.06, 0.06),
      /*reset=*/PauliChannel(0.04, 0.03, 0.03),
      /*spread=*/TQGSpreadModel(PauliChannel(0.12, 0.04, 0.04), PauliChannel(0.10, 0.05, 0.05)));
  model.check_false_negative_prob = 0.02;
  model.check_false_positive_prob = 0.01;

  const CompiledErasureProgram compiled(erasure_circuit, model);
  StreamSampler sampler(compiled);
  SurfHMMDecoder decoder(compiled);

  std::atomic<uint32_t> shots_seen{0};
  std::atomic<uint64_t> total_flagged{0};
  const auto t0 = std::chrono::steady_clock::now();

  sampler.sample(
      kShots, kSeed,
      [&](const stim::Circuit& circuit, const std::vector<uint8_t>& check_results) {
        uint64_t local_flagged = 0;
        for (uint8_t bit : check_results) {
          if (bit == 1) {
            ++local_flagged;
          }
        }
        total_flagged.fetch_add(local_flagged, std::memory_order_relaxed);
        const stim::Circuit injected = decoder.decode(circuit, &check_results, /*print_posteriors=*/false);
        (void)injected;
        shots_seen.fetch_add(1, std::memory_order_relaxed);
      },
      kThreads);
  const auto t1 = std::chrono::steady_clock::now();
  const double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
  const double shots_per_s = elapsed_s > 0.0 ? static_cast<double>(kShots) / elapsed_s : 0.0;

  const uint32_t shots_seen_value = shots_seen.load(std::memory_order_relaxed);
  const uint64_t total_flagged_value = total_flagged.load(std::memory_order_relaxed);

  if (shots_seen_value != kShots) {
    throw std::runtime_error("HmmDecoder stream test did not process expected number of shots");
  }

  std::cout << "hmm_decoder_stream_test\n";
  std::cout << "shots_seen: " << shots_seen_value << "\n";
  std::cout << "total_flagged_mappings: " << total_flagged_value << "\n";
  std::cout << "elapsed_s: " << elapsed_s << "\n";
  std::cout << "shots_per_s: " << shots_per_s << "\n";
  return 0;
}
