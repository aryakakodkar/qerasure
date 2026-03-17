#include "core/decode/rail_surface_dem_builder.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "core/circuit/instruction.h"
#include "core/simulator/sim_internal_utils.h"

namespace qerasure::decode {

namespace {

uint64_t make_op_qubit_key(uint32_t op_index, uint32_t qubit) {
  return (static_cast<uint64_t>(op_index) << 32) | qubit;
}

uint32_t find_start_qubit_op_offset(
    const circuit::CompiledErasureProgram& program,
    uint32_t qubit,
    int32_t reset_op_after_lookback) {
  const std::vector<uint32_t>& qubit_ops = program.qubit_operation_indices.at(qubit);
  if (qubit_ops.empty()) {
    throw std::logic_error("qubit_operation_indices entry is unexpectedly empty");
  }
  if (reset_op_after_lookback < 0) {
    return 0;
  }
  const uint32_t reset_op = static_cast<uint32_t>(reset_op_after_lookback);
  const auto it = std::lower_bound(qubit_ops.begin(), qubit_ops.end(), reset_op);
  if (it == qubit_ops.end()) {
    throw std::logic_error("reset op index not found in qubit_operation_indices");
  }
  return static_cast<uint32_t>(it - qubit_ops.begin());
}

uint32_t find_end_qubit_op_offset(
    const circuit::CompiledErasureProgram& program,
    uint32_t qubit,
    uint32_t check_op_index) {
  const std::vector<uint32_t>& qubit_ops = program.qubit_operation_indices.at(qubit);
  const auto it = std::lower_bound(qubit_ops.begin(), qubit_ops.end(), check_op_index);
  if (it == qubit_ops.end() || *it != check_op_index) {
    throw std::logic_error("check op index not found in qubit_operation_indices");
  }
  return static_cast<uint32_t>(it - qubit_ops.begin());
}

}  // namespace

RailSurfaceDemBuilder::RailSurfaceDemBuilder(
    const circuit::RailSurfaceCompiledProgram& program)
    : rail_program_(program),
      program_(program.base_program()),
      base_builder_(program.base_program()),
      op_to_emit_op_index_(program_.operation_groups.size(), 0) {
  uint32_t last_emit = 0;
  for (uint32_t op_index = 0; op_index < program_.operation_groups.size(); ++op_index) {
    if (program_.operation_groups[op_index].stim_instruction.has_value()) {
      last_emit = op_index;
    }
    op_to_emit_op_index_[op_index] = last_emit;
  }
}

RailSurfaceDemBuilder::BranchEvidence RailSurfaceDemBuilder::compute_branch_evidence(
    uint32_t data_qubit,
    uint32_t onset_op_index,
    uint32_t check_round,
    const std::vector<uint8_t>* detector_samples) const {
  const auto slots = rail_program_.data_z_ancilla_slots(data_qubit);
  std::vector<RailChoiceWeight> hypotheses;
  if (slots.first >= 0 && slots.second >= 0) {
    hypotheses.push_back({slots.first, 0.5});
    hypotheses.push_back({slots.second, 0.5});
  } else if (slots.first >= 0 || slots.second >= 0) {
    const int32_t slot = slots.first >= 0 ? slots.first : slots.second;
    hypotheses.push_back({slot, 0.5});
    hypotheses.push_back({-1, 0.5});
  } else {
    hypotheses.push_back({-1, 1.0});
  }

  const uint32_t start_round = check_round == 0 ? 0 : (check_round - 1);
  constexpr double kEpsilon = 0.05;
  std::vector<double> weighted_likelihoods(hypotheses.size(), 0.0);
  double total = 0.0;
  for (size_t i = 0; i < hypotheses.size(); ++i) {
    const int32_t selected_z_ancilla = hypotheses[i].z_ancilla;
    double likelihood = 1.0;
    for (uint32_t round = start_round; round <= check_round; ++round) {
      for (const int32_t z_ancilla : {slots.first, slots.second}) {
        if (z_ancilla < 0) {
          continue;
        }
        const int32_t det_index =
            rail_program_.round_detector_index(round, static_cast<uint32_t>(z_ancilla));
        if (det_index < 0 ||
            det_index >= static_cast<int32_t>(detector_samples->size())) {
          continue;
        }
        const int32_t interaction_op = rail_program_.interaction_op_for_data_z_ancilla(
            data_qubit, static_cast<uint32_t>(z_ancilla), round);
        const bool active_before_interaction =
            interaction_op >= 0 && onset_op_index < static_cast<uint32_t>(interaction_op);
        const bool predicted_flag =
            selected_z_ancilla == z_ancilla && active_before_interaction;
        const uint8_t observed = (*detector_samples)[det_index];
        const double p_obs = predicted_flag
            ? (observed ? (1.0 - kEpsilon) : kEpsilon)
            : (observed ? kEpsilon : (1.0 - kEpsilon));
        likelihood *= p_obs;
      }
    }
    weighted_likelihoods[i] = hypotheses[i].weight * likelihood;
    total += weighted_likelihoods[i];
  }

  BranchEvidence out;
  out.total_likelihood = total;
  if (total <= 0.0) {
    return out;
  }
  out.choice_weights.reserve(hypotheses.size());
  for (size_t i = 0; i < hypotheses.size(); ++i) {
    const double posterior = weighted_likelihoods[i] / total;
    if (posterior <= 0.0) {
      continue;
    }
    out.choice_weights.push_back({hypotheses[i].z_ancilla, posterior});
  }
  return out;
}

double RailSurfaceDemBuilder::compute_no_erasure_evidence(
    uint32_t data_qubit,
    uint32_t check_round,
    const std::vector<uint8_t>* detector_samples) const {
  const auto slots = rail_program_.data_z_ancilla_slots(data_qubit);
  const uint32_t start_round = check_round == 0 ? 0 : (check_round - 1);
  constexpr double kEpsilon = 0.05;
  double likelihood = 1.0;
  for (uint32_t round = start_round; round <= check_round; ++round) {
    for (const int32_t z_ancilla : {slots.first, slots.second}) {
      if (z_ancilla < 0) {
        continue;
      }
      const int32_t det_index =
          rail_program_.round_detector_index(round, static_cast<uint32_t>(z_ancilla));
      if (det_index < 0 ||
          det_index >= static_cast<int32_t>(detector_samples->size())) {
        continue;
      }
      const uint8_t observed = (*detector_samples)[det_index];
      likelihood *= observed ? kEpsilon : (1.0 - kEpsilon);
    }
  }
  return likelihood;
}

void RailSurfaceDemBuilder::append_rail_events_for_branch(
    uint32_t data_qubit,
    uint32_t onset_op_index,
    uint32_t check_round,
    double branch_posterior,
    const std::vector<RailChoiceWeight>& choice_weights,
    SpreadInjectionBuckets* buckets) const {
  if (branch_posterior <= 0.0 || buckets == nullptr) {
    return;
  }
  const uint32_t start_round = check_round == 0 ? 0 : (check_round - 1);
  for (const RailChoiceWeight& choice : choice_weights) {
    if (choice.z_ancilla < 0 || choice.weight <= 0.0) {
      continue;
    }
    const uint32_t z_ancilla = static_cast<uint32_t>(choice.z_ancilla);
    for (uint32_t round = start_round; round <= check_round; ++round) {
      const int32_t interaction_op =
          rail_program_.interaction_op_for_data_z_ancilla(data_qubit, z_ancilla, round);
      if (interaction_op < 0) {
        continue;
      }
      const uint32_t op_index = static_cast<uint32_t>(interaction_op);
      if (onset_op_index >= op_index) {
        continue;
      }
      const uint32_t emit_op_index = op_to_emit_op_index_[op_index];
      const double p_x = branch_posterior * choice.weight;
      bool merged = false;
      for (SpreadInjectionEvent& event : (*buckets)[emit_op_index]) {
        if (event.target_qubit == z_ancilla) {
          const double remaining = std::max(0.0, 1.0 - (event.p_x + event.p_y + event.p_z));
          event.p_x += std::min(p_x, remaining);
          merged = true;
          break;
        }
      }
      if (!merged) {
        (*buckets)[emit_op_index].push_back({emit_op_index, z_ancilla, std::min(1.0, p_x), 0.0, 0.0});
      }
    }
  }
}

SpreadInjectionBuckets RailSurfaceDemBuilder::compute_spread_injections_with_evidence(
    const std::vector<uint8_t>* check_results,
    const std::vector<uint8_t>* detector_samples,
    bool verbose,
    SkippableReweightMap* skippable_reweights) const {
  if (check_results == nullptr || detector_samples == nullptr) {
    throw std::invalid_argument(
        "RailSurfaceDemBuilder requires check_results and detector_samples");
  }
  if (check_results->size() != program_.num_checks()) {
    throw std::invalid_argument("RailSurfaceDemBuilder check_results size mismatch");
  }

  SpreadInjectionBuckets buckets = base_builder_.compute_spread_injections(
      check_results, verbose, skippable_reweights);
  if (program_.max_persistence() != 2) {
    return buckets;
  }

  for (uint32_t check_event_index = 0; check_event_index < check_results->size(); ++check_event_index) {
    if ((*check_results)[check_event_index] != 1) {
      continue;
    }
    const uint32_t qubit = rail_program_.check_event_to_qubit().at(check_event_index);
    if (!rail_program_.is_data_qubit(qubit)) {
      continue;
    }
    const uint32_t check_op = rail_program_.check_event_to_op_index().at(check_event_index);
    const int32_t check_round = rail_program_.op_round(check_op);
    if (check_round < 0) {
      continue;
    }

    const circuit::CheckLookbackLink& link = program_.check_lookback_links.at(check_event_index);
    const uint32_t start_offset = find_start_qubit_op_offset(
        program_, qubit, link.reset_op_after_lookback);
    const uint32_t end_offset = find_end_qubit_op_offset(program_, qubit, check_op);
    if (start_offset > end_offset) {
      continue;
    }

    double p_unerased = 1.0;
    double no_erasure_check_likelihood = 1.0;
    std::vector<OnsetBranch> branches;
    branches.reserve(static_cast<size_t>(end_offset - start_offset + 1));
    const std::vector<uint32_t>& qubit_ops = program_.qubit_operation_indices.at(qubit);
    for (uint32_t offset = start_offset; offset <= end_offset; ++offset) {
      const uint32_t op_index = qubit_ops[offset];
      const circuit::OperationGroup& op_group = program_.operation_groups[op_index];

      for (const auto& onset : op_group.onsets) {
        if (onset.qubit_index != qubit) {
          continue;
        }
        const double p_erase = p_unerased * onset.probability;
        branches.push_back({op_index, p_erase, 0});
        p_unerased -= p_erase;
      }

      for (const auto& onset_pair : op_group.onset_pairs) {
        if (onset_pair.qubit_index1 != qubit && onset_pair.qubit_index2 != qubit) {
          continue;
        }
        const double p_erase = p_unerased * (0.5 * onset_pair.probability);
        branches.push_back({op_index, p_erase, 0});
        p_unerased -= p_erase;
      }

      for (const auto& check : op_group.checks) {
        if (check.qubit_index != qubit) {
          continue;
        }
        const bool is_flagged_check = op_index == check_op;
        for (OnsetBranch& branch : branches) {
          if (!is_flagged_check) {
            branch.probability *= check.false_negative_probability;
            branch.survived_checks++;
          } else {
            const bool forced_detection = branch.survived_checks >= 1;
            const double flag_prob = forced_detection ? 1.0 : (1.0 - check.false_negative_probability);
            branch.probability *= flag_prob;
          }
        }
        no_erasure_check_likelihood *= is_flagged_check
            ? check.false_positive_probability
            : (1.0 - check.false_positive_probability);
        break;
      }
    }

    double branch_mass = 0.0;
    std::vector<BranchEvidence> branch_evidence;
    branch_evidence.reserve(branches.size());
    for (const OnsetBranch& branch : branches) {
      BranchEvidence evidence = compute_branch_evidence(
          qubit, branch.op_index, static_cast<uint32_t>(check_round), detector_samples);
      branch_mass += branch.probability * evidence.total_likelihood;
      branch_evidence.push_back(std::move(evidence));
    }

    const double no_erasure_mass = p_unerased * no_erasure_check_likelihood *
        compute_no_erasure_evidence(
            qubit, static_cast<uint32_t>(check_round), detector_samples);
    const double normalizer = branch_mass + no_erasure_mass;
    if (normalizer <= 0.0) {
      continue;
    }

    for (size_t i = 0; i < branches.size(); ++i) {
      const OnsetBranch& branch = branches[i];
      const BranchEvidence& evidence = branch_evidence[i];
      if (evidence.total_likelihood <= 0.0) {
        continue;
      }
      const double posterior =
          (branch.probability * evidence.total_likelihood) / normalizer;
      append_rail_events_for_branch(
          qubit,
          branch.op_index,
          static_cast<uint32_t>(check_round),
          posterior,
          evidence.choice_weights,
          &buckets);
    }
  }

  return buckets;
}

std::vector<RailSurfaceDemBuilder::CalibrationRow> RailSurfaceDemBuilder::calibration_rows(
    const std::vector<uint8_t>* check_results,
    const std::vector<uint8_t>* detector_samples) const {
  if (check_results == nullptr || detector_samples == nullptr) {
    throw std::invalid_argument(
        "RailSurfaceDemBuilder calibration_rows requires check_results and detector_samples");
  }
  if (check_results->size() != program_.num_checks()) {
    throw std::invalid_argument("RailSurfaceDemBuilder calibration_rows check_results size mismatch");
  }
  std::vector<CalibrationRow> rows;
  if (program_.max_persistence() != 2) {
    return rows;
  }

  for (uint32_t check_event_index = 0; check_event_index < check_results->size(); ++check_event_index) {
    if ((*check_results)[check_event_index] != 1) {
      continue;
    }
    const uint32_t qubit = rail_program_.check_event_to_qubit().at(check_event_index);
    if (!rail_program_.is_data_qubit(qubit)) {
      continue;
    }
    const uint32_t check_op = rail_program_.check_event_to_op_index().at(check_event_index);
    const int32_t check_round = rail_program_.op_round(check_op);
    if (check_round < 0) {
      continue;
    }

    const circuit::CheckLookbackLink& link = program_.check_lookback_links.at(check_event_index);
    const uint32_t start_offset = find_start_qubit_op_offset(
        program_, qubit, link.reset_op_after_lookback);
    const uint32_t end_offset = find_end_qubit_op_offset(program_, qubit, check_op);
    if (start_offset > end_offset) {
      continue;
    }

    double p_unerased = 1.0;
    double no_erasure_check_likelihood = 1.0;
    std::vector<OnsetBranch> branches;
    branches.reserve(static_cast<size_t>(end_offset - start_offset + 1));
    const std::vector<uint32_t>& qubit_ops = program_.qubit_operation_indices.at(qubit);
    for (uint32_t offset = start_offset; offset <= end_offset; ++offset) {
      const uint32_t op_index = qubit_ops[offset];
      const circuit::OperationGroup& op_group = program_.operation_groups[op_index];

      for (const auto& onset : op_group.onsets) {
        if (onset.qubit_index != qubit) {
          continue;
        }
        const double p_erase = p_unerased * onset.probability;
        branches.push_back({op_index, p_erase, 0});
        p_unerased -= p_erase;
      }

      for (const auto& onset_pair : op_group.onset_pairs) {
        if (onset_pair.qubit_index1 != qubit && onset_pair.qubit_index2 != qubit) {
          continue;
        }
        const double p_erase = p_unerased * (0.5 * onset_pair.probability);
        branches.push_back({op_index, p_erase, 0});
        p_unerased -= p_erase;
      }

      for (const auto& check : op_group.checks) {
        if (check.qubit_index != qubit) {
          continue;
        }
        const bool is_flagged_check = op_index == check_op;
        for (OnsetBranch& branch : branches) {
          if (!is_flagged_check) {
            branch.probability *= check.false_negative_probability;
            branch.survived_checks++;
          } else {
            const bool forced_detection = branch.survived_checks >= 1;
            const double flag_prob = forced_detection ? 1.0 : (1.0 - check.false_negative_probability);
            branch.probability *= flag_prob;
          }
        }
        no_erasure_check_likelihood *= is_flagged_check
            ? check.false_positive_probability
            : (1.0 - check.false_positive_probability);
        break;
      }
    }

    std::vector<BranchEvidence> branch_evidence;
    branch_evidence.reserve(branches.size());
    double branch_mass = 0.0;
    for (const OnsetBranch& branch : branches) {
      BranchEvidence evidence = compute_branch_evidence(
          qubit, branch.op_index, static_cast<uint32_t>(check_round), detector_samples);
      branch_mass += branch.probability * evidence.total_likelihood;
      branch_evidence.push_back(std::move(evidence));
    }

    const double no_erasure_mass = p_unerased * no_erasure_check_likelihood *
        compute_no_erasure_evidence(
            qubit, static_cast<uint32_t>(check_round), detector_samples);
    const double normalizer = branch_mass + no_erasure_mass;
    if (normalizer <= 0.0) {
      continue;
    }

    for (size_t i = 0; i < branches.size(); ++i) {
      const OnsetBranch& branch = branches[i];
      const BranchEvidence& evidence = branch_evidence[i];
      if (evidence.total_likelihood <= 0.0) {
        continue;
      }
      const double posterior =
          (branch.probability * evidence.total_likelihood) / normalizer;
      rows.push_back(
          {check_event_index,
           qubit,
           check_op,
           static_cast<uint32_t>(check_round),
           branch.op_index,
           branch.probability,
           evidence.total_likelihood,
           posterior,
           rail_program_.data_qubit_schedule_type(qubit),
           rail_program_.data_qubit_is_boundary(qubit)});
    }
  }

  return rows;
}

stim::Circuit RailSurfaceDemBuilder::build_decoded_circuit(
    const std::vector<uint8_t>* check_results,
    const std::vector<uint8_t>* detector_samples,
    bool verbose) const {
  SkippableReweightMap skippable_reweights;
  SpreadInjectionBuckets buckets = compute_spread_injections_with_evidence(
      check_results, detector_samples, verbose, &skippable_reweights);

  stim::Circuit injected;
  for (uint32_t op_index = 0; op_index < program_.operation_groups.size(); ++op_index) {
    const circuit::OperationGroup& op_group = program_.operation_groups[op_index];
    if (op_group.stim_instruction.has_value()) {
      const circuit::Instruction& instr = *op_group.stim_instruction;
      const bool should_reweight =
          circuit::is_erasure_skippable_op(instr.op) && circuit::is_probabilistic_op(instr.op);
      if (!should_reweight) {
        simulator::internal::append_mapped_stim_instruction(instr, &injected);
      } else {
        const char* op_name = circuit::opcode_name(instr.op);
        for (const uint32_t target : instr.targets) {
          double p_unerased = 1.0;
          const auto it = skippable_reweights.find(make_op_qubit_key(op_index, target));
          if (it != skippable_reweights.end()) {
            p_unerased = it->second;
          }
          const double reweighted_prob = std::clamp(instr.arg * p_unerased, 0.0, 1.0);
          if (reweighted_prob <= 0.0) {
            continue;
          }
          injected.safe_append_ua(op_name, {target}, reweighted_prob);
        }
      }
    }

    for (const SpreadInjectionEvent& event : buckets[op_index]) {
      const double p_x = std::clamp(event.p_x, 0.0, 1.0);
      const double p_y = std::clamp(event.p_y, 0.0, 1.0);
      const double p_z = std::clamp(event.p_z, 0.0, 1.0);
      if (p_x <= 0.0 && p_y <= 0.0 && p_z <= 0.0) {
        continue;
      }
      injected.safe_append_u("PAULI_CHANNEL_1", {event.target_qubit}, {p_x, p_y, p_z});
    }
  }
  return injected;
}

std::string RailSurfaceDemBuilder::build_decoded_circuit_text(
    const std::vector<uint8_t>* check_results,
    const std::vector<uint8_t>* detector_samples,
    bool verbose) const {
  SkippableReweightMap skippable_reweights;
  SpreadInjectionBuckets buckets = compute_spread_injections_with_evidence(
      check_results, detector_samples, verbose, &skippable_reweights);

  std::ostringstream out;
  bool first_line = true;
  for (uint32_t op_index = 0; op_index < program_.operation_groups.size(); ++op_index) {
    const circuit::OperationGroup& op_group = program_.operation_groups[op_index];
    if (op_group.stim_instruction.has_value()) {
      const auto& instr = *op_group.stim_instruction;
      const bool should_reweight =
          circuit::is_erasure_skippable_op(instr.op) && circuit::is_probabilistic_op(instr.op);
      if (!should_reweight) {
        if (!first_line) {
          out << "\n";
        }
        first_line = false;
        out << circuit::opcode_name(instr.op);
        if (circuit::is_probabilistic_op(instr.op)) {
          out << "(" << instr.arg << ")";
        } else if (instr.op == circuit::OpCode::OBSERVABLE_INCLUDE) {
          out << "(0)";
        }
        for (const uint32_t target : instr.targets) {
          if (circuit::uses_measurement_record_targets(instr.op)) {
            out << " rec[-" << target << "]";
          } else {
            out << " " << target;
          }
        }
      } else {
        for (const uint32_t target : instr.targets) {
          double p_unerased = 1.0;
          const auto it = skippable_reweights.find(make_op_qubit_key(op_index, target));
          if (it != skippable_reweights.end()) {
            p_unerased = it->second;
          }
          const double reweighted_prob = std::clamp(instr.arg * p_unerased, 0.0, 1.0);
          if (reweighted_prob <= 0.0) {
            continue;
          }
          if (!first_line) {
            out << "\n";
          }
          first_line = false;
          out << circuit::opcode_name(instr.op) << "(" << reweighted_prob << ") " << target;
        }
      }
    }

    for (const SpreadInjectionEvent& event : buckets[op_index]) {
      const double p_x = std::clamp(event.p_x, 0.0, 1.0);
      const double p_y = std::clamp(event.p_y, 0.0, 1.0);
      const double p_z = std::clamp(event.p_z, 0.0, 1.0);
      if (p_x <= 0.0 && p_y <= 0.0 && p_z <= 0.0) {
        continue;
      }
      if (!first_line) {
        out << "\n";
      }
      first_line = false;
      out << "PAULI_CHANNEL_1(" << p_x << ", " << p_y << ", " << p_z << ") "
          << event.target_qubit;
    }
  }
  return out.str();
}

}  // namespace qerasure::decode
