#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "qerasure/core/code/rotated_surface_code.h"
#include "stim/circuit/circuit.h"

namespace qerasure::translation_internal {

struct CircuitBuildContext {
  std::size_t num_qubits = 0;
  std::size_t num_data = 0;
  std::size_t num_x_anc = 0;
  std::size_t num_z_anc = 0;
  std::size_t num_anc = 0;

  std::vector<uint32_t> data_qubits_u32;
  std::vector<uint32_t> x_ancillas_u32;
  std::vector<uint32_t> ancillas_u32;

  std::vector<std::vector<uint32_t>> cx_targets_by_step;
  std::vector<std::vector<std::size_t>> z_ancilla_supports;
  std::vector<std::size_t> logical_x_data_qubits;
};

void append_index_op(stim::Circuit* circuit, const char* op, const std::vector<uint32_t>& indices);
void append_detector_lookbacks(stim::Circuit* circuit, std::vector<uint32_t>* rec_targets,
                               const std::vector<uint32_t>& rec_lookbacks);
CircuitBuildContext build_context(const RotatedSurfaceCode& code);
void append_round_detectors(stim::Circuit* circuit, const CircuitBuildContext& ctx, std::size_t round,
                            std::vector<uint32_t>* detector_lookbacks,
                            std::vector<uint32_t>* detector_targets);
void append_final_readout_detectors_and_observable(stim::Circuit* circuit, const CircuitBuildContext& ctx,
                                                   std::vector<uint32_t>* detector_lookbacks,
                                                   std::vector<uint32_t>* detector_targets);

// Shared extraction-round skeleton with injection hooks used by translation pipelines.
template <typename PreStepHook, typename PostStepHook, typename PreMeasureHook>
void append_extraction_round(stim::Circuit* circuit, const CircuitBuildContext& ctx,
                             std::size_t round_index, PreStepHook&& pre_step_hook,
                             PostStepHook&& post_step_hook, PreMeasureHook&& pre_measure_hook,
                             std::vector<uint32_t>* detector_lookbacks,
                             std::vector<uint32_t>* detector_targets) {
  append_index_op(circuit, "H", ctx.x_ancillas_u32);
  for (std::size_t step = 0; step < 4; ++step) {
    pre_step_hook(round_index, step);
    circuit->safe_append_u("CX", ctx.cx_targets_by_step[step]);
    post_step_hook(round_index, step);
  }
  append_index_op(circuit, "H", ctx.x_ancillas_u32);
  pre_measure_hook(round_index);
  append_index_op(circuit, "MR", ctx.ancillas_u32);
  append_round_detectors(circuit, ctx, round_index, detector_lookbacks, detector_targets);
}

}  // namespace qerasure::translation_internal
