#include <cstdint>
#include <stdexcept>
#include <vector>

#include "core/circuit/erasure_model.h"
#include "core/circuit/rail_surface_compile.h"
#include "core/decode/rail_surface_dem_builder.h"
#include "core/gen/surf.h"
#include "core/model/pauli_channel.h"
#include "core/simulator/rail_stream_sampler.h"

namespace {

void expect(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

}  // namespace

int main() {
  using qerasure::circuit::ErasureModel;
  using qerasure::circuit::RailSurfaceCompiledProgram;
  using qerasure::circuit::TQGSpreadModel;
  using qerasure::decode::RailSurfaceDemBuilder;
  using qerasure::gen::SurfaceCodeRotated;
  using qerasure::simulator::RailStreamSampler;

  constexpr uint32_t kDistance = 3;
  constexpr uint32_t kRounds = 3;
  constexpr double kErasureProb = 0.02;

  SurfaceCodeRotated generator(kDistance);
  const auto circuit = generator.build_circuit(
      kRounds,
      kErasureProb,
      "ALL",
      0.0,
      false,
      true,
      0.0,
      2);

  ErasureModel model(
      /*max_persistence=*/2,
      /*onset=*/::PauliChannel(0.25, 0.25, 0.25),
      /*reset=*/::PauliChannel(0.25, 0.25, 0.25),
      /*spread=*/TQGSpreadModel(
          ::PauliChannel(0.25, 0.25, 0.25),
          ::PauliChannel(0.25, 0.25, 0.25)));
  model.check_false_negative_prob = 0.02;
  model.check_false_positive_prob = 0.02;

  RailSurfaceCompiledProgram rail_program(circuit, model, kDistance, kRounds);
  RailStreamSampler rail_sampler(rail_program);
  auto sampled = rail_sampler.sample_syndromes(/*num_shots=*/8, /*seed=*/1234, /*num_threads=*/1);
  expect(sampled.num_checks == rail_program.base_program().num_checks(), "num_checks mismatch");
  expect(sampled.num_detectors > 0, "expected nonzero detector count");
  expect(sampled.check_flags.size() == sampled.num_shots * sampled.num_checks, "check buffer mismatch");

  int32_t picked_event = -1;
  uint32_t picked_qubit = 0;
  uint32_t picked_round = 0;
  for (uint32_t i = 0; i < rail_program.check_event_to_qubit().size(); ++i) {
    const uint32_t q = rail_program.check_event_to_qubit()[i];
    if (!rail_program.is_data_qubit(q)) {
      continue;
    }
    const int32_t round = rail_program.op_round(rail_program.check_event_to_op_index()[i]);
    if (round >= 1) {
      picked_event = static_cast<int32_t>(i);
      picked_qubit = q;
      picked_round = static_cast<uint32_t>(round);
      break;
    }
  }
  expect(picked_event >= 0, "failed to find data-qubit check event for evidence test");

  std::vector<uint8_t> check_results(rail_program.base_program().num_checks(), 0);
  check_results[static_cast<size_t>(picked_event)] = 1;
  std::vector<uint8_t> detector_samples(rail_program.num_detectors(), 0);

  const auto slots = rail_program.data_z_ancilla_slots(picked_qubit);
  const int32_t preferred_slot = slots.first >= 0 ? slots.first : slots.second;
  const int32_t secondary_slot = slots.first >= 0 && slots.second >= 0 ? slots.second : -1;
  expect(preferred_slot >= 0, "picked data qubit has no adjacent Z ancilla");

  const uint32_t start_round = picked_round == 0 ? 0 : (picked_round - 1);
  for (uint32_t round = start_round; round <= picked_round; ++round) {
    const int32_t pref_detector = rail_program.round_detector_index(
        round, static_cast<uint32_t>(preferred_slot));
    if (pref_detector >= 0) {
      detector_samples[static_cast<size_t>(pref_detector)] = 1;
    }
    if (secondary_slot >= 0) {
      const int32_t sec_detector = rail_program.round_detector_index(
          round, static_cast<uint32_t>(secondary_slot));
      if (sec_detector >= 0) {
        detector_samples[static_cast<size_t>(sec_detector)] = 0;
      }
    }
  }

  RailSurfaceDemBuilder rail_builder(rail_program);
  const auto calibration_rows = rail_builder.calibration_rows(&check_results, &detector_samples);
  expect(!calibration_rows.empty(), "expected non-empty calibration rows for flagged data check");
  auto buckets = rail_builder.compute_spread_injections_with_evidence(
      &check_results, &detector_samples, /*verbose=*/false, /*skippable_reweights=*/nullptr);

  double preferred_mass = 0.0;
  double secondary_mass = 0.0;
  for (const auto& bucket : buckets) {
    for (const auto& event : bucket) {
      if (event.target_qubit == static_cast<uint32_t>(preferred_slot)) {
        preferred_mass += event.p_x;
      } else if (secondary_slot >= 0 &&
                 event.target_qubit == static_cast<uint32_t>(secondary_slot)) {
        secondary_mass += event.p_x;
      }
    }
  }
  expect(preferred_mass > 0.0, "expected rail-conditioned X mass on preferred Z ancilla");
  if (secondary_slot >= 0) {
    expect(preferred_mass > secondary_mass, "preferred rail should dominate under matching evidence");
  }

  return 0;
}
