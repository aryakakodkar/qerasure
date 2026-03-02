#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "core/circuit/circuit.h"
#include "core/circuit/compile.h"
#include "core/circuit/erasure_model.h"
#include "core/model/pauli_channel.h"
#include "core/simulator/erasure_sampler.h"

namespace {

std::string resolve_circuit_path(int argc, char** argv) {
  if (argc > 1) {
    return argv[1];
  }
  // Running from repo root.
  const std::string root_relative = "benchmarks/erasure_sampler_bench.circuit";
  // Running from build directory.
  const std::string build_relative = "../benchmarks/erasure_sampler_bench.circuit";

  {
    qerasure::circuit::ErasureCircuit probe;
    try {
      probe.from_file(root_relative);
      return root_relative;
    } catch (const std::exception&) {
      // fall through
    }
  }
  return build_relative;
}

}  // namespace

int main(int argc, char** argv) {
  using namespace qerasure::circuit;    // NOLINT
  using namespace qerasure::simulator;  // NOLINT

  const std::string circuit_path = resolve_circuit_path(argc, argv);

  ErasureCircuit circuit;
  try {
    circuit.from_file(circuit_path);
  } catch (const std::exception& ex) {
    std::cerr << "Failed to load circuit file '" << circuit_path << "': " << ex.what() << "\n";
    return EXIT_FAILURE;
  }

  ErasureModel model(
      /*max_persistence=*/3,
      /*onset=*/PauliChannel(0.18, 0.06, 0.06),
      /*reset=*/PauliChannel(0.04, 0.03, 0.03),
      /*spread=*/TQGSpreadModel(PauliChannel(0.12, 0.04, 0.04), PauliChannel(0.10, 0.05, 0.05)));
  model.check_false_negative_prob = 0.02;
  model.check_false_positive_prob = 0.01;

  const CompiledErasureProgram compiled(circuit, model);
  ErasureSampler sampler(compiled);

  SamplerParams params{};
  params.shots = 10'000;
  params.seed = 12345;

  const auto t0 = std::chrono::steady_clock::now();
  const SampledBatch batch = sampler.sample(params);
  const auto t1 = std::chrono::steady_clock::now();

  std::uint64_t total_onsets = 0;
  std::uint64_t total_spreads = 0;
  std::uint64_t total_checks = 0;
  std::uint64_t total_resets = 0;
  for (const auto& shot : batch.shots) {
    for (const auto& group : shot.operation_groups) {
      total_onsets += group.onsets.size();
      total_spreads += group.spreads.size();
      total_checks += group.checks.size();
      total_resets += group.resets.size();
    }
  }

  const double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
  const double shots_per_s = elapsed_s > 0.0 ? static_cast<double>(params.shots) / elapsed_s : 0.0;

  std::cout << "erasure_sampler_bench\n";
  std::cout << "circuit_file: " << circuit_path << "\n";
  std::cout << "ops: " << compiled.operation_groups.size() << "\n";
  std::cout << "shots: " << params.shots << "\n";
  std::cout << "elapsed_s: " << elapsed_s << "\n";
  std::cout << "shots_per_s: " << shots_per_s << "\n";
  std::cout << "sampled_onsets: " << total_onsets << "\n";
  std::cout << "sampled_spreads: " << total_spreads << "\n";
  std::cout << "sampled_checks: " << total_checks << "\n";
  std::cout << "sampled_resets: " << total_resets << "\n";

  return EXIT_SUCCESS;
}
