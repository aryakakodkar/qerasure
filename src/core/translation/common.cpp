#include "common.h"

#include <algorithm>

#include "stim/circuit/gate_target.h"

namespace qerasure::translation_internal {

namespace {

std::vector<uint32_t> as_u32_targets(const std::vector<std::size_t>& indices) {
  std::vector<uint32_t> out;
  out.reserve(indices.size());
  for (const std::size_t q : indices) {
    out.push_back(static_cast<uint32_t>(q));
  }
  return out;
}

}  // namespace

void append_index_op(stim::Circuit* circuit, const char* op, const std::vector<uint32_t>& indices) {
  if (!indices.empty()) {
    circuit->safe_append_u(op, indices);
  }
}

void append_detector_lookbacks(stim::Circuit* circuit, std::vector<uint32_t>* rec_targets,
                               const std::vector<uint32_t>& rec_lookbacks) {
  rec_targets->clear();
  rec_targets->reserve(rec_lookbacks.size());
  for (const uint32_t lookback : rec_lookbacks) {
    rec_targets->push_back(lookback | stim::TARGET_RECORD_BIT);
  }
  circuit->safe_append_u("DETECTOR", *rec_targets);
}

CircuitBuildContext build_context(const RotatedSurfaceCode& code) {
  CircuitBuildContext ctx;
  ctx.num_qubits = code.num_qubits();
  ctx.num_data = code.x_anc_offset();
  const std::size_t x_anc_offset = code.x_anc_offset();
  const std::size_t z_anc_offset = code.z_anc_offset();
  ctx.num_x_anc = z_anc_offset - x_anc_offset;
  ctx.num_z_anc = ctx.num_qubits - z_anc_offset;
  ctx.num_anc = ctx.num_x_anc + ctx.num_z_anc;

  std::vector<std::size_t> data_qubits;
  data_qubits.reserve(ctx.num_data);
  for (std::size_t q = 0; q < ctx.num_data; ++q) {
    data_qubits.push_back(q);
  }

  std::vector<std::size_t> x_ancillas;
  x_ancillas.reserve(ctx.num_x_anc);
  for (std::size_t q = x_anc_offset; q < z_anc_offset; ++q) {
    x_ancillas.push_back(q);
  }

  std::vector<std::size_t> z_ancillas;
  z_ancillas.reserve(ctx.num_z_anc);
  for (std::size_t q = z_anc_offset; q < ctx.num_qubits; ++q) {
    z_ancillas.push_back(q);
  }

  std::vector<std::size_t> ancillas = x_ancillas;
  ancillas.insert(ancillas.end(), z_ancillas.begin(), z_ancillas.end());

  ctx.data_qubits_u32 = as_u32_targets(data_qubits);
  ctx.x_ancillas_u32 = as_u32_targets(x_ancillas);
  ctx.ancillas_u32 = as_u32_targets(ancillas);

  const std::vector<Gate>& gates = code.gates();
  const std::size_t gates_per_step = code.gates_per_step();
  ctx.cx_targets_by_step.assign(4, {});
  for (std::size_t step = 0; step < 4; ++step) {
    std::vector<uint32_t>& cx_targets = ctx.cx_targets_by_step[step];
    cx_targets.reserve(gates_per_step * 2);
    const std::size_t step_start = step * gates_per_step;
    for (std::size_t i = 0; i < gates_per_step; ++i) {
      const Gate& gate = gates[step_start + i];
      cx_targets.push_back(static_cast<uint32_t>(gate.first));
      cx_targets.push_back(static_cast<uint32_t>(gate.second));
    }
  }

  const std::vector<std::size_t>& partner_map = code.partner_map();
  ctx.z_ancilla_supports.assign(ctx.num_z_anc, {});
  for (std::size_t zi = 0; zi < ctx.num_z_anc; ++zi) {
    const std::size_t z_anc = z_ancillas[zi];
    std::vector<std::size_t>& support = ctx.z_ancilla_supports[zi];
    support.reserve(4);
    for (std::size_t step = 0; step < 4; ++step) {
      const std::size_t partner = partner_map[step * ctx.num_qubits + z_anc];
      if (partner != kNoPartner && partner < ctx.num_data) {
        support.push_back(partner);
      }
    }
    std::sort(support.begin(), support.end());
    support.erase(std::unique(support.begin(), support.end()), support.end());
  }

  const auto& coords = code.index_to_coord();
  for (const std::size_t q : data_qubits) {
    if (coords[q].second == 1) {
      ctx.logical_x_data_qubits.push_back(q); // TODO_ARYA: incorrectly named
    }
  }

  return ctx;
}

void append_round_detectors(stim::Circuit* circuit, const CircuitBuildContext& ctx, std::size_t round,
                            std::vector<uint32_t>* detector_lookbacks,
                            std::vector<uint32_t>* detector_targets) {
  for (std::size_t zi = 0; zi < ctx.num_z_anc; ++zi) {
    const std::size_t ancilla_position = ctx.num_x_anc + zi;
    const uint32_t current_lookback = static_cast<uint32_t>(ctx.num_anc - ancilla_position);
    detector_lookbacks->clear();
    detector_lookbacks->push_back(current_lookback);
    if (round > 0) {
      const uint32_t previous_lookback = static_cast<uint32_t>(2 * ctx.num_anc - ancilla_position);
      detector_lookbacks->push_back(previous_lookback);
    }
    append_detector_lookbacks(circuit, detector_targets, *detector_lookbacks);
  }
}

void append_final_readout_detectors_and_observable(stim::Circuit* circuit, const CircuitBuildContext& ctx,
                                                   std::vector<uint32_t>* detector_lookbacks,
                                                   std::vector<uint32_t>* detector_targets) {
  append_index_op(circuit, "M", ctx.data_qubits_u32);

  for (std::size_t zi = 0; zi < ctx.num_z_anc; ++zi) {
    detector_lookbacks->clear();
    const std::size_t ancilla_position = ctx.num_x_anc + zi;
    const uint32_t ancilla_lookback_after_data =
        static_cast<uint32_t>(ctx.num_data + (ctx.num_anc - ancilla_position));
    detector_lookbacks->push_back(ancilla_lookback_after_data);
    for (const std::size_t data_q : ctx.z_ancilla_supports[zi]) {
      detector_lookbacks->push_back(static_cast<uint32_t>(ctx.num_data - data_q));
    }
    append_detector_lookbacks(circuit, detector_targets, *detector_lookbacks);
  }

  std::vector<uint32_t> logical_targets;
  logical_targets.reserve(ctx.logical_x_data_qubits.size());
  for (const std::size_t data_q : ctx.logical_x_data_qubits) {
    logical_targets.push_back((static_cast<uint32_t>(ctx.num_data - data_q)) | stim::TARGET_RECORD_BIT);
  }
  circuit->safe_append_ua("OBSERVABLE_INCLUDE", logical_targets, 0.0);
}

}  // namespace qerasure::translation_internal
