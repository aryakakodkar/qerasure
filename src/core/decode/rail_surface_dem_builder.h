#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "core/circuit/rail_surface_compile.h"
#include "core/decode/surf_dem_builder.h"
#include "stim/circuit/circuit.h"

namespace qerasure::decode {

class RailSurfaceDemBuilder {
 public:
  static constexpr size_t kNumScheduleBuckets = 2;
  static constexpr size_t kNumConditionBuckets = 4;
  static constexpr size_t kNumOnsetBins = 8;

  struct CalibrationRow {
    uint32_t check_event_index;
    uint32_t data_qubit;
    uint32_t check_op_index;
    uint32_t check_round;
    uint32_t onset_op_index;
    double prior_mass;
    double evidence_likelihood;
    double posterior_mass;
    int32_t schedule_type;
    bool boundary_data_qubit;
  };

  explicit RailSurfaceDemBuilder(const circuit::RailSurfaceCompiledProgram& program);

  // Configure calibrated onset priors P(onset_bin | condition, schedule_type).
  // Expected shape is [2][4][8] where:
  // schedule bucket 0 := XZZX interior, 1 := ZXXZ interior;
  // condition bucket 0 := 00, 1 := 10, 2 := 01, 3 := 11.
  // If `boost_nonzero_with_pe2` is true, each strictly-positive entry is
  // incremented by p_e^2 and renormalized per [schedule][condition] row.
  void set_calibrated_onset_posteriors(
      double erasure_probability,
      const std::vector<std::vector<std::vector<double>>>& posteriors,
      bool boost_nonzero_with_pe2 = true);

  // Configure calibrated onset priors used only for flagged checks on the
  // final round (check_round == rounds-1). Falls back to the default table
  // when unset.
  void set_final_round_calibrated_onset_posteriors(
      double erasure_probability,
      const std::vector<std::vector<std::vector<double>>>& posteriors,
      bool boost_nonzero_with_pe2 = true);

  void clear_calibrated_onset_posteriors();

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

  std::vector<CalibrationRow> calibration_rows(
      const std::vector<uint8_t>* check_results,
      const std::vector<uint8_t>* detector_samples) const;

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

  using CalibratedPosteriorCube = std::array<
      std::array<std::array<double, kNumOnsetBins>, kNumConditionBuckets>,
      kNumScheduleBuckets>;

  BranchEvidence compute_branch_evidence(
      uint32_t data_qubit,
      uint32_t onset_op_index,
      uint32_t check_round,
      const std::vector<uint8_t>* detector_samples) const;

  double compute_no_erasure_evidence(
      uint32_t data_qubit,
      uint32_t check_round,
      const std::vector<uint8_t>* detector_samples) const;

  bool pair_inconsistency_in_round(
      uint32_t data_qubit,
      uint32_t round_index,
      const std::vector<uint8_t>* detector_samples) const;

  int classify_two_round_condition(
      uint32_t data_qubit,
      uint32_t check_round,
      const std::vector<uint8_t>* detector_samples) const;

  bool apply_calibrated_branch_priors(
      uint32_t data_qubit,
      uint32_t check_round,
      const std::vector<uint8_t>* detector_samples,
      std::vector<OnsetBranch>* branches,
      double* no_erasure_prior) const;

  void subtract_standard_spread_for_check(
      uint32_t check_event_index,
      SpreadInjectionBuckets* buckets) const;

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
  double evidence_mismatch_floor_;
  bool calibrated_posteriors_enabled_;
  CalibratedPosteriorCube calibrated_posteriors_;
  bool calibrated_final_round_posteriors_enabled_;
  CalibratedPosteriorCube calibrated_final_round_posteriors_;
};

}  // namespace qerasure::decode
