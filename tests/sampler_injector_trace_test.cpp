#include <iostream>
#include <stdexcept>

#include "core/circuit/compile.h"
#include "core/circuit/erasure_model.h"
#include "core/gen/surf.h"
#include "core/model/pauli_channel.h"
#include "core/simulator/erasure_sampler.h"
#include "core/simulator/injector.h"

int main() {
  using namespace qerasure::circuit;    // NOLINT
  using namespace qerasure::gen;        // NOLINT
  using namespace qerasure::simulator;  // NOLINT

  SurfaceCodeRotated generator(3);
  const ErasureCircuit erasure_circuit =
      generator.build_circuit(/*rounds=*/3, /*erasure_prob=*/0.3, /*erasable_qubits=*/"ALL");

  ErasureModel model(
      /*max_persistence=*/3,
      /*onset=*/PauliChannel(0.25, 0.25, 0.25),
      /*reset=*/PauliChannel(0.25, 0.25, 0.25),
      /*spread=*/TQGSpreadModel(PauliChannel(0.5, 0.0, 0.0), PauliChannel(0.0, 0.0, 0.5)));
  model.check_false_negative_prob = 0.02;
  model.check_false_positive_prob = 0.01;

  const CompiledErasureProgram compiled(erasure_circuit, model);
  ErasureSampler sampler(compiled);

  SamplerParams params{};
  params.shots = 1;
  params.seed = 12345;
  const SampledBatch sampled = sampler.sample(params);
  if (sampled.shots.empty()) {
    throw std::runtime_error("Sampler returned no shots.");
  }

  Injector injector;
  const stim::Circuit injected = injector.inject(sampled, /*shot_index=*/0);

  std::cout << "=== INJECTED STIM CIRCUIT ===\n";
  std::cout << injected << "\n";
  std::cout << "=== SAMPLED SHOT TRACE ===\n";
  std::cout << sampled.shots[0].to_string() << "\n";

  return 0;
}
