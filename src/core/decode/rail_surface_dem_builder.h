#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/circuit/rail_surface_compile.h"
#include "core/decode/surf_dem_builder.h"
#include "stim/circuit/circuit.h"

namespace qerasure::decode {

class RailSurfaceDemBuilder {
 public:
  explicit RailSurfaceDemBuilder(const circuit::RailSurfaceCompiledProgram& program);

  SpreadInjectionBuckets compute_spread_injections_with_evidence(
      const std::vector<uint8_t>* check_results,
      const std::vector<uint8_t>* detector_samples,
      bool verbose = false,
      SkippableReweightMap* skippable_reweights = nullptr) const;

  stim::Circuit build_decoded_circuit(
      const std::vector<uint8_t>* check_results,
      const std::vector<uint8_t>* detector_samples,
      bool verbose = false) const;

  std::string build_decoded_circuit_text(
      const std::vector<uint8_t>* check_results,
      const std::vector<uint8_t>* detector_samples,
      bool verbose = false) const;

 private:
  struct RailChoiceWeight {
    int32_t z_ancilla;
    double weight;
  };

  struct OnsetBranch {
    uint32_t op_index;
    double probability;
    uint32_t survived_checks;
  };

  struct BranchEvidence {
    double total_likelihood;
    std::vector<RailChoiceWeight> choice_weights;
  };

  BranchEvidence compute_branch_evidence(
      uint32_t data_qubit,
      uint32_t onset_op_index,
      uint32_t check_round,
      const std::vector<uint8_t>* detector_samples) const;

  double compute_no_erasure_evidence(
      uint32_t data_qubit,
      uint32_t check_round,
      const std::vector<uint8_t>* detector_samples) const;

  void append_rail_events_for_branch(
      uint32_t data_qubit,
      uint32_t onset_op_index,
      uint32_t check_round,
      double branch_posterior,
      const std::vector<RailChoiceWeight>& choice_weights,
      SpreadInjectionBuckets* buckets) const;

  const circuit::RailSurfaceCompiledProgram& rail_program_;
  const circuit::CompiledErasureProgram& program_;
  SurfDemBuilder base_builder_;
  std::vector<uint32_t> op_to_emit_op_index_;
};

}  // namespace qerasure::decode
