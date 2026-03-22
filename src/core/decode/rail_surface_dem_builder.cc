#include "core/decode/rail_surface_dem_builder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
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

// Returns the other qubit in an ERASE2_ANY onset pair touching `data_qubit`
// at `onset_op_index`, or -1 when no such onset-pair branch exists.
int32_t find_onset_pair_partner_qubit(
    const circuit::CompiledErasureProgram& program,
    uint32_t data_qubit,
    uint32_t onset_op_index) {
  if (onset_op_index >= program.operation_groups.size()) {
    return -1;
  }
  const circuit::OperationGroup& op_group = program.operation_groups[onset_op_index];
  for (const auto& onset_pair : op_group.onset_pairs) {
    if (onset_pair.qubit_index1 == data_qubit) {
      return static_cast<int32_t>(onset_pair.qubit_index2);
    }
    if (onset_pair.qubit_index2 == data_qubit) {
      return static_cast<int32_t>(onset_pair.qubit_index1);
    }
  }
  return -1;
}

int schedule_bucket_for_type(int32_t schedule_type) {
  if (schedule_type == 1) {
    return 0;
  }
  if (schedule_type == 2) {
    return 1;
  }
  return -1;
}

void normalize_pauli_channel_probs(double* p_x, double* p_y, double* p_z) {
  constexpr double kTolerance = 1e-12;
  *p_x = std::clamp(*p_x, 0.0, 1.0);
  *p_y = std::clamp(*p_y, 0.0, 1.0);
  *p_z = std::clamp(*p_z, 0.0, 1.0);
  double total = *p_x + *p_y + *p_z;
  if (total <= 0.0) {
    *p_x = 0.0;
    *p_y = 0.0;
    *p_z = 0.0;
    return;
  }
  if (total > 1.0) {
    if (total <= 1.0 + kTolerance) {
      const double excess = total - 1.0;
      if (*p_x >= *p_y && *p_x >= *p_z) {
        *p_x = std::max(0.0, *p_x - excess);
      } else if (*p_y >= *p_x && *p_y >= *p_z) {
        *p_y = std::max(0.0, *p_y - excess);
      } else {
        *p_z = std::max(0.0, *p_z - excess);
      }
    } else {
      const double scale = 1.0 / total;
      *p_x *= scale;
      *p_y *= scale;
      *p_z *= scale;
    }
  }
}

std::array<std::array<std::array<double, RailSurfaceDemBuilder::kNumOnsetBins>,
                      RailSurfaceDemBuilder::kNumConditionBuckets>,
           RailSurfaceDemBuilder::kNumScheduleBuckets>
normalize_calibrated_posterior_cube(
    double erasure_probability,
    const std::vector<std::vector<std::vector<double>>>& posteriors,
    bool boost_nonzero_with_pe2) {
  if (posteriors.size() != RailSurfaceDemBuilder::kNumScheduleBuckets) {
    throw std::invalid_argument("expected calibrated posterior shape [2][4][8]");
  }
  std::array<std::array<std::array<double, RailSurfaceDemBuilder::kNumOnsetBins>,
                        RailSurfaceDemBuilder::kNumConditionBuckets>,
             RailSurfaceDemBuilder::kNumScheduleBuckets>
      normalized{};
  const double p_e = std::clamp(erasure_probability, 0.0, 1.0);
  const double pe2 = p_e * p_e;
  for (size_t schedule = 0; schedule < RailSurfaceDemBuilder::kNumScheduleBuckets; ++schedule) {
    if (posteriors[schedule].size() != RailSurfaceDemBuilder::kNumConditionBuckets) {
      throw std::invalid_argument("expected calibrated posterior shape [2][4][8]");
    }
    for (size_t condition = 0; condition < RailSurfaceDemBuilder::kNumConditionBuckets; ++condition) {
      if (posteriors[schedule][condition].size() != RailSurfaceDemBuilder::kNumOnsetBins) {
        throw std::invalid_argument("expected calibrated posterior shape [2][4][8]");
      }
      double row_sum = 0.0;
      for (size_t bin = 0; bin < RailSurfaceDemBuilder::kNumOnsetBins; ++bin) {
        const double raw = posteriors[schedule][condition][bin];
        if (!std::isfinite(raw) || raw < 0.0) {
          throw std::invalid_argument("calibrated onset posteriors must be finite and >= 0");
        }
        double value = raw;
        if (boost_nonzero_with_pe2 && value > 0.0) {
          value += pe2;
        }
        normalized[schedule][condition][bin] = value;
        row_sum += value;
      }
      if (row_sum <= 0.0) {
        throw std::invalid_argument(
            "each [schedule][condition] calibrated onset row must contain positive mass");
      }
      for (size_t bin = 0; bin < RailSurfaceDemBuilder::kNumOnsetBins; ++bin) {
        normalized[schedule][condition][bin] /= row_sum;
      }
    }
  }
  return normalized;
}

}  // namespace

RailSurfaceDemBuilder::RailSurfaceDemBuilder(
    const circuit::RailSurfaceCompiledProgram& program)
    : rail_program_(program),
      program_(program.base_program()),
      base_builder_(program.base_program()),
      op_to_emit_op_index_(program_.operation_groups.size(), 0),
      evidence_mismatch_floor_(1e-12),
      calibrated_posteriors_enabled_(false),
      calibrated_posteriors_{},
      calibrated_final_round_posteriors_enabled_(false),
      calibrated_final_round_posteriors_{} {
  double onset_probability_scale = 0.0;
  for (const circuit::OperationGroup& group : program_.operation_groups) {
    for (const auto& onset : group.onsets) {
      onset_probability_scale = std::max(onset_probability_scale, onset.probability);
    }
    for (const auto& onset_pair : group.onset_pairs) {
      onset_probability_scale = std::max(onset_probability_scale, onset_pair.probability);
    }
  }
  if (onset_probability_scale <= 0.0) {
    onset_probability_scale = 1e-6;
  }
  // Keep impossible branches at second-order scale instead of hard-zero.
  evidence_mismatch_floor_ = std::clamp(
      onset_probability_scale * onset_probability_scale, 1e-12, 0.25);

  uint32_t last_emit = 0;
  for (uint32_t op_index = 0; op_index < program_.operation_groups.size(); ++op_index) {
    if (program_.operation_groups[op_index].stim_instruction.has_value()) {
      last_emit = op_index;
    }
    op_to_emit_op_index_[op_index] = last_emit;
  }
}

void RailSurfaceDemBuilder::set_calibrated_onset_posteriors(
    double erasure_probability,
    const std::vector<std::vector<std::vector<double>>>& posteriors,
    bool boost_nonzero_with_pe2) {
  calibrated_posteriors_ = normalize_calibrated_posterior_cube(
      erasure_probability, posteriors, boost_nonzero_with_pe2);
  calibrated_posteriors_enabled_ = true;
}

void RailSurfaceDemBuilder::set_final_round_calibrated_onset_posteriors(
    double erasure_probability,
    const std::vector<std::vector<std::vector<double>>>& posteriors,
    bool boost_nonzero_with_pe2) {
  calibrated_final_round_posteriors_ = normalize_calibrated_posterior_cube(
      erasure_probability, posteriors, boost_nonzero_with_pe2);
  calibrated_final_round_posteriors_enabled_ = true;
}

void RailSurfaceDemBuilder::clear_calibrated_onset_posteriors() {
  calibrated_posteriors_enabled_ = false;
  calibrated_posteriors_ = {};
  calibrated_final_round_posteriors_enabled_ = false;
  calibrated_final_round_posteriors_ = {};
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
  const int32_t onset_round = rail_program_.op_round(onset_op_index);
  const int32_t onset_partner = find_onset_pair_partner_qubit(
      program_, data_qubit, onset_op_index);
  const bool onset_partner_is_z =
      onset_partner >= static_cast<int32_t>(rail_program_.z_anc_offset());
  const double onset_flip_probability =
      std::clamp(
          program_.model().onset.p_x + program_.model().onset.p_y,
          0.0,
          1.0);

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
        double predicted_flag_probability = predicted_flag ? 1.0 : 0.0;
        // If onset happened via ERASE2_ANY against this Z ancilla, include the
        // onset-channel bit-flip contribution at the onset round.
        if (onset_partner_is_z &&
            onset_round >= 0 &&
            round == static_cast<uint32_t>(onset_round) &&
            z_ancilla == onset_partner) {
          predicted_flag_probability =
              predicted_flag ? (1.0 - onset_flip_probability) : onset_flip_probability;
        }
        const uint8_t observed = (*detector_samples)[det_index];
        const double p_obs_if_flag =
            observed ? (1.0 - evidence_mismatch_floor_) : evidence_mismatch_floor_;
        const double p_obs_if_no_flag =
            observed ? evidence_mismatch_floor_ : (1.0 - evidence_mismatch_floor_);
        const double p_obs =
            predicted_flag_probability * p_obs_if_flag +
            (1.0 - predicted_flag_probability) * p_obs_if_no_flag;
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
      likelihood *= observed ? evidence_mismatch_floor_ : (1.0 - evidence_mismatch_floor_);
    }
  }
  return likelihood;
}

bool RailSurfaceDemBuilder::pair_inconsistency_in_round(
    uint32_t data_qubit,
    uint32_t round_index,
    const std::vector<uint8_t>* detector_samples) const {
  const auto slots = rail_program_.data_z_ancilla_slots(data_qubit);
  if (slots.first < 0 || slots.second < 0) {
    return false;
  }
  const int32_t det0 = rail_program_.round_detector_index(
      round_index, static_cast<uint32_t>(slots.first));
  const int32_t det1 = rail_program_.round_detector_index(
      round_index, static_cast<uint32_t>(slots.second));
  if (det0 < 0 || det1 < 0) {
    return false;
  }
  if (det0 >= static_cast<int32_t>(detector_samples->size()) ||
      det1 >= static_cast<int32_t>(detector_samples->size())) {
    return false;
  }
  return (*detector_samples)[static_cast<size_t>(det0)] !=
      (*detector_samples)[static_cast<size_t>(det1)];
}

int RailSurfaceDemBuilder::classify_two_round_condition(
    uint32_t data_qubit,
    uint32_t check_round,
    const std::vector<uint8_t>* detector_samples) const {
  const bool inc_round1 =
      check_round > 0 &&
      pair_inconsistency_in_round(data_qubit, check_round - 1, detector_samples);
  const bool inc_round2 =
      pair_inconsistency_in_round(data_qubit, check_round, detector_samples);
  if (inc_round1 && inc_round2) {
    return 3;
  }
  if (inc_round1) {
    return 1;
  }
  if (inc_round2) {
    return 2;
  }
  return 0;
}

bool RailSurfaceDemBuilder::apply_calibrated_branch_priors(
    uint32_t data_qubit,
    uint32_t check_round,
    const std::vector<uint8_t>* detector_samples,
    std::vector<OnsetBranch>* branches,
    double* no_erasure_prior) const {
  if (!calibrated_posteriors_enabled_ || branches == nullptr || no_erasure_prior == nullptr ||
      detector_samples == nullptr) {
    return false;
  }
  if (check_round == 0) {
    return false;
  }
  // Calibration tables are learned on full-interior qubits only.
  // Fall back to the base branch priors for boundary/degenerate neighborhoods.
  if (!rail_program_.data_qubit_is_full_interior(data_qubit)) {
    return false;
  }
  const int schedule_bucket = schedule_bucket_for_type(
      rail_program_.data_qubit_schedule_type(data_qubit));
  if (schedule_bucket < 0) {
    return false;
  }
  const int condition_bucket = classify_two_round_condition(
      data_qubit, check_round, detector_samples);
  if (condition_bucket < 0 ||
      condition_bucket >= static_cast<int>(kNumConditionBuckets)) {
    return false;
  }

  std::array<std::vector<size_t>, kNumOnsetBins> bin_to_branch_indices;
  std::array<double, kNumOnsetBins> bin_raw_mass{};
  std::array<uint32_t, kNumOnsetBins> prev_round_ops{};
  std::array<uint32_t, kNumOnsetBins> curr_round_ops{};
  std::vector<double> original_branch_probabilities(branches->size(), 0.0);
  size_t prev_round_count = 0;
  size_t curr_round_count = 0;
  bool has_out_of_window_mass = false;

  for (size_t i = 0; i < branches->size(); ++i) {
    const OnsetBranch& branch = (*branches)[i];
    original_branch_probabilities[i] = std::max(0.0, branch.probability);
    const int32_t onset_round = rail_program_.op_round(branch.op_index);
    if (onset_round == static_cast<int32_t>(check_round - 1)) {
      if (prev_round_count < 4 &&
          std::find(
              prev_round_ops.begin(),
              prev_round_ops.begin() + static_cast<ptrdiff_t>(prev_round_count),
              branch.op_index) ==
              prev_round_ops.begin() + static_cast<ptrdiff_t>(prev_round_count)) {
        prev_round_ops[prev_round_count++] = branch.op_index;
      }
    } else if (onset_round == static_cast<int32_t>(check_round)) {
      if (curr_round_count < 4 &&
          std::find(
              curr_round_ops.begin(),
              curr_round_ops.begin() + static_cast<ptrdiff_t>(curr_round_count),
              branch.op_index) ==
              curr_round_ops.begin() + static_cast<ptrdiff_t>(curr_round_count)) {
        curr_round_ops[curr_round_count++] = branch.op_index;
      }
    } else if (original_branch_probabilities[i] > 0.0) {
      // Calibration tables currently model only two-round onset support:
      // rounds (check_round-1) and check_round. If appreciable posterior mass
      // exists outside this window (e.g. sparse checks), applying the table
      // would incorrectly zero that mass.
      has_out_of_window_mass = true;
    }
  }
  if (has_out_of_window_mass) {
    return false;
  }
  if (prev_round_count != 4 || curr_round_count != 4) {
    return false;
  }
  std::sort(prev_round_ops.begin(), prev_round_ops.begin() + static_cast<ptrdiff_t>(prev_round_count));
  std::sort(curr_round_ops.begin(), curr_round_ops.begin() + static_cast<ptrdiff_t>(curr_round_count));

  for (size_t i = 0; i < branches->size(); ++i) {
    const OnsetBranch& branch = (*branches)[i];
    const int32_t onset_round = rail_program_.op_round(branch.op_index);
    int bin = -1;
    if (onset_round == static_cast<int32_t>(check_round - 1)) {
      const auto it = std::lower_bound(
          prev_round_ops.begin(),
          prev_round_ops.begin() + static_cast<ptrdiff_t>(prev_round_count),
          branch.op_index);
      if (it != prev_round_ops.begin() + static_cast<ptrdiff_t>(prev_round_count) &&
          *it == branch.op_index) {
        bin = static_cast<int>(it - prev_round_ops.begin());
      }
    } else if (onset_round == static_cast<int32_t>(check_round)) {
      const auto it = std::lower_bound(
          curr_round_ops.begin(),
          curr_round_ops.begin() + static_cast<ptrdiff_t>(curr_round_count),
          branch.op_index);
      if (it != curr_round_ops.begin() + static_cast<ptrdiff_t>(curr_round_count) &&
          *it == branch.op_index) {
        bin = static_cast<int>(4 + (it - curr_round_ops.begin()));
      }
    }
    if (bin < 0 || bin >= static_cast<int>(kNumOnsetBins)) {
      continue;
    }
    bin_to_branch_indices[static_cast<size_t>(bin)].push_back(i);
    bin_raw_mass[static_cast<size_t>(bin)] += original_branch_probabilities[i];
  }

  for (OnsetBranch& branch : *branches) {
    branch.probability = 0.0;
  }

  const bool use_final_round_table =
      calibrated_final_round_posteriors_enabled_ &&
      check_round + 1 == rail_program_.rounds();
  const CalibratedPosteriorCube& posterior_table =
      use_final_round_table ? calibrated_final_round_posteriors_ : calibrated_posteriors_;
  // Preserve the model-derived no-erasure prior mass (important when q > 0).
  // Calibrated tables describe onset-bin allocation conditioned on an erasure
  // explanation, not the marginal probability that a flagged check was a
  // false-positive with no erasure in-window.
  const double base_no_erasure_prior = std::clamp(*no_erasure_prior, 0.0, 1.0);
  const double calibrated_branch_total = std::max(0.0, 1.0 - base_no_erasure_prior);

  double assigned_mass = 0.0;
  for (size_t bin = 0; bin < kNumOnsetBins; ++bin) {
    const double target_mass = calibrated_branch_total *
        posterior_table[static_cast<size_t>(schedule_bucket)]
                       [static_cast<size_t>(condition_bucket)][bin];
    if (target_mass <= 0.0 || bin_to_branch_indices[bin].empty()) {
      continue;
    }
    assigned_mass += target_mass;
    const double raw_mass = bin_raw_mass[bin];
    if (raw_mass > 0.0) {
      for (size_t branch_index : bin_to_branch_indices[bin]) {
        const double raw = original_branch_probabilities[branch_index];
        const double weight = raw / raw_mass;
        (*branches)[branch_index].probability = target_mass * weight;
      }
    } else {
      const double share =
          target_mass / static_cast<double>(bin_to_branch_indices[bin].size());
      for (size_t branch_index : bin_to_branch_indices[bin]) {
        (*branches)[branch_index].probability = share;
      }
    }
  }

  *no_erasure_prior = std::max(0.0, 1.0 - assigned_mass);
  return true;
}

void RailSurfaceDemBuilder::subtract_standard_spread_for_check(
    uint32_t check_event_index,
    SpreadInjectionBuckets* buckets) const {
  if (buckets == nullptr) {
    return;
  }
  const uint32_t qubit = rail_program_.check_event_to_qubit().at(check_event_index);
  const uint32_t check_op = rail_program_.check_event_to_op_index().at(check_event_index);
  const circuit::CheckLookbackLink& link = program_.check_lookback_links.at(check_event_index);
  const uint32_t start_offset = find_start_qubit_op_offset(
      program_, qubit, link.reset_op_after_lookback);
  const uint32_t end_offset = find_end_qubit_op_offset(program_, qubit, check_op);
  if (start_offset > end_offset) {
    return;
  }

  double p_unerased = 1.0;
  std::vector<OnsetBranch> branches;
  branches.reserve(static_cast<size_t>(end_offset - start_offset + 1));
  const std::vector<uint32_t>& qubit_ops = program_.qubit_operation_indices.at(qubit);

  for (uint32_t offset = start_offset; offset <= end_offset; ++offset) {
    const uint32_t op_index = qubit_ops[offset];
    const circuit::OperationGroup& op_group = program_.operation_groups[op_index];
    double p_onset_at_op = 0.0;

    for (const auto& onset : op_group.onsets) {
      if (onset.qubit_index != qubit) {
        continue;
      }
      const double p_erase = p_unerased * onset.probability;
      branches.push_back({op_index, p_erase, 0});
      p_unerased -= p_erase;
      p_onset_at_op += p_erase;
    }

    for (const auto& onset_pair : op_group.onset_pairs) {
      if (onset_pair.qubit_index1 != qubit && onset_pair.qubit_index2 != qubit) {
        continue;
      }
      const double p_erase = p_unerased * (0.5 * onset_pair.probability);
      branches.push_back({op_index, p_erase, 0});
      p_unerased -= p_erase;
      p_onset_at_op += p_erase;
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
      break;
    }

    double p_erased_by_op = 0.0;
    for (const OnsetBranch& branch : branches) {
      p_erased_by_op += branch.probability;
    }
    const uint32_t emit_op_index = op_to_emit_op_index_[op_index];

    auto subtract_from_event = [&buckets, emit_op_index](uint32_t target,
                                                         double p_x,
                                                         double p_y,
                                                         double p_z) {
      if (p_x <= 0.0 && p_y <= 0.0 && p_z <= 0.0) {
        return;
      }
      if (emit_op_index >= buckets->size()) {
        return;
      }
      for (SpreadInjectionEvent& event : (*buckets)[emit_op_index]) {
        if (event.target_qubit != target) {
          continue;
        }
        event.p_x = std::max(0.0, event.p_x - p_x);
        event.p_y = std::max(0.0, event.p_y - p_y);
        event.p_z = std::max(0.0, event.p_z - p_z);
        return;
      }
    };

    if (p_onset_at_op > 0.0) {
      for (const auto& spread : op_group.onset_spreads) {
        if (spread.source_qubit_index != qubit) {
          continue;
        }
        subtract_from_event(
            spread.aff_qubit_index,
            p_onset_at_op * spread.spread_probability_channel.p_x,
            p_onset_at_op * spread.spread_probability_channel.p_y,
            p_onset_at_op * spread.spread_probability_channel.p_z);
      }
    }

    if (p_erased_by_op > 0.0) {
      for (const auto& spread : op_group.persistent_spreads) {
        if (spread.source_qubit_index != qubit) {
          continue;
        }
        subtract_from_event(
            spread.aff_qubit_index,
            p_erased_by_op * spread.spread_probability_channel.p_x,
            p_erased_by_op * spread.spread_probability_channel.p_y,
            p_erased_by_op * spread.spread_probability_channel.p_z);
      }
    }
  }
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
          normalize_pauli_channel_probs(&event.p_x, &event.p_y, &event.p_z);
          merged = true;
          break;
        }
      }
      if (!merged) {
        double new_px = std::min(1.0, p_x);
        double new_py = 0.0;
        double new_pz = 0.0;
        normalize_pauli_channel_probs(&new_px, &new_py, &new_pz);
        (*buckets)[emit_op_index].push_back({emit_op_index, z_ancilla, new_px, new_py, new_pz});
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
    if (!rail_program_.data_qubit_is_full_interior(qubit)) {
      continue;
    }
    // Remove standard onset/persistent spread from this interior flagged-check
    // branch before injecting rail-conditioned replacement spread.
    subtract_standard_spread_for_check(check_event_index, &buckets);
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

    double no_erasure_prior = p_unerased * no_erasure_check_likelihood;
    const bool used_calibrated_priors = apply_calibrated_branch_priors(
        qubit,
        static_cast<uint32_t>(check_round),
        detector_samples,
        &branches,
        &no_erasure_prior);

    double branch_mass = 0.0;
    std::vector<BranchEvidence> branch_evidence;
    branch_evidence.reserve(branches.size());
    for (const OnsetBranch& branch : branches) {
      BranchEvidence evidence = compute_branch_evidence(
          qubit, branch.op_index, static_cast<uint32_t>(check_round), detector_samples);
      if (used_calibrated_priors) {
        branch_mass += branch.probability;
      } else {
        branch_mass += branch.probability * evidence.total_likelihood;
      }
      branch_evidence.push_back(std::move(evidence));
    }

    const double no_erasure_mass = used_calibrated_priors
        ? std::max(0.0, no_erasure_prior)
        : no_erasure_prior * compute_no_erasure_evidence(
              qubit, static_cast<uint32_t>(check_round), detector_samples);
    const double normalizer = branch_mass + no_erasure_mass;
    if (normalizer <= 0.0) {
      continue;
    }

    for (size_t i = 0; i < branches.size(); ++i) {
      const OnsetBranch& branch = branches[i];
      const BranchEvidence& evidence = branch_evidence[i];
      if (branch.probability <= 0.0) {
        continue;
      }
      if (!used_calibrated_priors && evidence.total_likelihood <= 0.0) {
        continue;
      }
      if (used_calibrated_priors && evidence.choice_weights.empty()) {
        continue;
      }
      const double posterior = used_calibrated_priors
          ? (branch.probability / normalizer)
          : ((branch.probability * evidence.total_likelihood) / normalizer);
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
      double p_x = event.p_x;
      double p_y = event.p_y;
      double p_z = event.p_z;
      normalize_pauli_channel_probs(&p_x, &p_y, &p_z);
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
  out << std::setprecision(17);
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
      double p_x = event.p_x;
      double p_y = event.p_y;
      double p_z = event.p_z;
      normalize_pauli_channel_probs(&p_x, &p_y, &p_z);
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
