#include "core/decode/surf_dem_builder.h"
#include "core/circuit/compile.h"
#include "core/simulator/sim_internal_utils.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace qerasure::decode {

namespace {

// Methods for finding start and end indices for iteration over qubit operations
uint32_t find_start_qubit_op_offset(const circuit::CompiledErasureProgram& program,
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

uint32_t find_end_qubit_op_offset(const circuit::CompiledErasureProgram& program,
									uint32_t qubit,
									uint32_t check_op_index) {
	
	const std::vector<uint32_t>& qubit_ops = program.qubit_operation_indices.at(qubit);

	const auto it = std::lower_bound(qubit_ops.begin(), qubit_ops.end(), check_op_index);

	if (it == qubit_ops.end() || *it != check_op_index) {
		throw std::logic_error("check op index not found in qubit_operation_indices");
	}

	return static_cast<uint32_t>(it - qubit_ops.begin());
}

struct OnsetBranch {
	uint32_t op_index;
	double probability;
	uint32_t survived_checks;
	bool from_onset_pair;
	uint32_t onset_pair_target;
};

struct LocalTargetChannelAccum {
	uint32_t target_qubit;
	double p_x;
	double p_y;
	double p_z;
};

uint64_t make_op_qubit_key(uint32_t op_index, uint32_t qubit) {
	return (static_cast<uint64_t>(op_index) << 32) | qubit;
}

}  // namespace

SurfDemBuilder::SurfDemBuilder(const circuit::CompiledErasureProgram& program) : program_(program) {

	check_event_to_qubit_.reserve(program_.num_checks());
	check_event_to_op_index_.reserve(program_.num_checks());
	op_to_emit_op_index_.resize(program_.operation_groups.size(), 0);

	uint32_t last_emit_op = 0;
	for (uint32_t op_index = 0; op_index < program_.operation_groups.size(); ++op_index) {
		if (program_.operation_groups[op_index].stim_instruction.has_value()) {
			last_emit_op = op_index;
		}
		op_to_emit_op_index_[op_index] = last_emit_op;
	}

	for (uint32_t op_index = 0; op_index < program_.operation_groups.size(); ++op_index) {
	const circuit::OperationGroup& group = program_.operation_groups[op_index];
	for (const circuit::ErasureCheck& check : group.checks) {
		check_event_to_qubit_.push_back(check.qubit_index);
		check_event_to_op_index_.push_back(op_index);
	}
	}
	qubit_check_events_.assign(program_.max_qubit_index() + 1, {});
	for (uint32_t check_event_index = 0; check_event_index < check_event_to_qubit_.size();
		 ++check_event_index) {
		const uint32_t qubit = check_event_to_qubit_[check_event_index];
		qubit_check_events_[qubit].push_back(check_event_index);
	}

	if (check_event_to_qubit_.size() != program_.num_checks()) {
		throw std::logic_error("SurfDemBuilder check-event map size mismatch with CompiledErasureProgram");
	}
	if (program_.check_lookback_links.size() != program_.num_checks()) {
		throw std::logic_error("SurfDemBuilder expected check_lookback_links to match num_checks");
	}
	for (uint32_t qubit = 0; qubit < qubit_check_events_.size(); ++qubit) {
		if (qubit_check_events_[qubit].size() !=
			program_.qubit_check_operation_indices.at(qubit).size()) {
			throw std::logic_error(
				"SurfDemBuilder per-qubit check-event mapping size mismatch with compiled check op indices");
		}
	}
}

void SurfDemBuilder::add_tail_hidden_injections(
	const std::vector<uint8_t>* check_results,
	uint32_t qubit,
	uint32_t start_op,
	uint32_t final_meas_op,
	SpreadInjectionBuckets* buckets,
	SkippableReweightMap* skippable_reweights) const {
	if (buckets == nullptr) {
		throw std::invalid_argument("tail hidden injections require non-null buckets");
	}
	const uint32_t max_persistence = program_.max_persistence();
	if (max_persistence == 0) {
		throw std::logic_error("max_persistence must be positive");
	}

	double p_unerased = 1.0;
	std::vector<double> erased_mass(max_persistence + 1, 0.0);

	const std::vector<uint32_t>& qubit_ops = program_.qubit_operation_indices.at(qubit);
	const std::vector<uint32_t>& local_check_events = qubit_check_events_.at(qubit);
	const std::vector<uint32_t>& local_check_ops = program_.qubit_check_operation_indices.at(qubit);
	const std::vector<uint32_t>& qubit_skippable_ops =
		program_.qubit_skippable_operation_indices.at(qubit);

	auto start_it = std::lower_bound(qubit_ops.begin(), qubit_ops.end(), start_op);
	auto end_it = std::upper_bound(
		start_it, qubit_ops.end(), static_cast<uint32_t>(final_meas_op));
	size_t local_check_cursor = static_cast<size_t>(
		std::lower_bound(local_check_ops.begin(), local_check_ops.end(), start_op) -
		local_check_ops.begin());
	size_t skippable_index = static_cast<size_t>(
		std::lower_bound(qubit_skippable_ops.begin(), qubit_skippable_ops.end(), start_op) -
		qubit_skippable_ops.begin());

	for (auto it = start_it; it != end_it; ++it) {
		const uint32_t op_index = *it;
		const circuit::OperationGroup& op_group = program_.operation_groups[op_index];
		double p_onset_at_op = 0.0;

		auto merge_event = [&](uint32_t emit_op_index, uint32_t target_qubit, double p_x, double p_y,
							   double p_z) {
			if (p_x <= 0.0 && p_y <= 0.0 && p_z <= 0.0) {
				return;
			}
			(*buckets)[emit_op_index].push_back({emit_op_index, target_qubit, p_x, p_y, p_z});
		};

		for (const auto& onset : op_group.onsets) {
			if (onset.qubit_index != qubit) {
				continue;
			}
			std::vector<double> next_erased_mass = erased_mass;
			const double p_onset_from_unerased = p_unerased * onset.probability;
			p_unerased *= (1.0 - onset.probability);
			next_erased_mass[1] += p_onset_from_unerased;
			p_onset_at_op += p_onset_from_unerased;
			for (uint32_t state = 1; state <= max_persistence; ++state) {
				const double fired = erased_mass[state] * onset.probability;
				next_erased_mass[state] -= fired;
				next_erased_mass[std::min(max_persistence, state + 1)] += fired;
				p_onset_at_op += fired;
			}
			erased_mass.swap(next_erased_mass);
		}
		for (const auto& onset_pair : op_group.onset_pairs) {
			if (onset_pair.qubit_index1 != qubit && onset_pair.qubit_index2 != qubit) {
				continue;
			}
			const double p_onset = p_unerased * (0.5 * onset_pair.probability);
			erased_mass[1] += p_onset;
			p_unerased -= p_onset;
			p_onset_at_op += p_onset;
			const uint32_t other_qubit =
				(onset_pair.qubit_index1 == qubit) ? onset_pair.qubit_index2 : onset_pair.qubit_index1;
			const PauliChannel& onset_channel = program_.model().onset;
			merge_event(
				op_to_emit_op_index_[op_index],
				other_qubit,
				p_onset * onset_channel.p_x,
				p_onset * onset_channel.p_y,
				p_onset * onset_channel.p_z);
		}

		if (p_onset_at_op > 0.0) {
			for (const auto& spread : op_group.onset_spreads) {
				if (spread.source_qubit_index != qubit) {
					continue;
				}
				merge_event(
					op_to_emit_op_index_[op_index],
					spread.aff_qubit_index,
					p_onset_at_op * spread.spread_probability_channel.p_x,
					p_onset_at_op * spread.spread_probability_channel.p_y,
					p_onset_at_op * spread.spread_probability_channel.p_z);
			}
		}

		double p_erased_by_op = 0.0;
		for (uint32_t state = 1; state <= max_persistence; ++state) {
			p_erased_by_op += erased_mass[state];
		}

		while (skippable_index < qubit_skippable_ops.size() &&
			   qubit_skippable_ops[skippable_index] < op_index) {
			++skippable_index;
		}
		if (skippable_reweights != nullptr &&
			skippable_index < qubit_skippable_ops.size() &&
			qubit_skippable_ops[skippable_index] == op_index) {
			const double p_unerased_by_op = std::clamp(1.0 - p_erased_by_op, 0.0, 1.0);
			const uint64_t key = make_op_qubit_key(op_index, qubit);
			const auto existing_it = skippable_reweights->find(key);
			if (existing_it == skippable_reweights->end() ||
				p_unerased_by_op < existing_it->second) {
				(*skippable_reweights)[key] = p_unerased_by_op;
			}
		}

		if (p_erased_by_op > 0.0) {
			for (const auto& spread : op_group.persistent_spreads) {
				if (spread.source_qubit_index != qubit) {
					continue;
				}
				merge_event(
					op_to_emit_op_index_[op_index],
					spread.aff_qubit_index,
					p_erased_by_op * spread.spread_probability_channel.p_x,
					p_erased_by_op * spread.spread_probability_channel.p_y,
					p_erased_by_op * spread.spread_probability_channel.p_z);
			}
		}

		for (const auto& check : op_group.checks) {
			if (check.qubit_index != qubit) {
				continue;
			}
			while (local_check_cursor < local_check_ops.size() &&
				   local_check_ops[local_check_cursor] < op_index) {
				++local_check_cursor;
			}
			if (local_check_cursor >= local_check_ops.size() ||
				local_check_ops[local_check_cursor] != op_index) {
				throw std::logic_error("missing check event index for qubit/op in DEM builder");
			}
			const uint8_t observed = (*check_results)[local_check_events[local_check_cursor]];
			++local_check_cursor;

			std::vector<double> next_erased_mass(max_persistence + 1, 0.0);
			if (observed == 0) {
				p_unerased *= (1.0 - check.false_positive_probability);
				for (uint32_t state = 1; state < max_persistence; ++state) {
					next_erased_mass[state + 1] += erased_mass[state] * check.false_negative_probability;
				}
			} else if (observed == 1) {
				// Tail windows start after the most recent flagged check, so any positive check
				// here indicates an inconsistent input window.
				throw std::logic_error("tail hidden injection window unexpectedly contains a flagged check");
			} else {
				throw std::invalid_argument("check_results must be binary");
			}
			erased_mass.swap(next_erased_mass);
			break;
		}

		if (op_group.stim_instruction.has_value() &&
			circuit::is_measurement_op(op_group.stim_instruction->op) &&
			p_erased_by_op > 0.0) {
			const uint32_t pre_emit_op_index =
				(op_index == 0) ? op_to_emit_op_index_[op_index]
								: op_to_emit_op_index_[op_index - 1];
			merge_event(pre_emit_op_index, qubit, 0.5 * p_erased_by_op, 0.0, 0.0);
		}
	}
}

SpreadInjectionBuckets SurfDemBuilder::compute_spread_injections(
	const std::vector<uint8_t>* check_results,
	bool verbose,
	SkippableReweightMap* skippable_reweights) const {
	SpreadInjectionBuckets buckets(program_.operation_groups.size());
	// Reuse per-source merge buffers across flagged checks to avoid repeated
	// full-size allocations and whole-range scans.
	std::vector<std::vector<LocalTargetChannelAccum>> source_emit_channels(
		program_.operation_groups.size());
	std::vector<uint32_t> touched_emit_ops;
	touched_emit_ops.reserve(256);

	if (check_results == nullptr) {
		throw std::invalid_argument(
			"SurfDemBuilder::compute_spread_injections requires non-null check_results pointer");
	}
	if (check_results->size() != program_.num_checks()) {
		throw std::invalid_argument(
			"SurfDemBuilder::compute_spread_injections check_results size mismatch");
	}

	for (uint32_t check_event_index = 0; check_event_index < check_results->size(); ++check_event_index) {
		const uint8_t bit = (*check_results)[check_event_index]; // 0: non-flagged, 1: flagged
		if (bit == 0) {
			continue;
		}
		if (bit != 1) {
			throw std::invalid_argument("SurfDemBuilder::build_decoded_circuit expects binary check results");
		}

		// Look up the qubit and check operation for this flagged check event.
		const circuit::CheckLookbackLink& link = program_.check_lookback_links.at(check_event_index);
		const uint32_t qubit = check_event_to_qubit_[check_event_index];
		const uint32_t check_op = check_event_to_op_index_[check_event_index];
		if (link.qubit_index != qubit || link.check_op_index != check_op) {
			throw std::logic_error("check_lookback_links metadata mismatch with decoder check maps");
		}

		// For every flagged check, apply reset-channel contribution on that same qubit.
		// Runtime sampler semantics: reset noise is applied when reset succeeds.
		// Therefore net injected probabilities are (1 - p_fail) * (p_x, p_y, p_z).
		const circuit::OperationGroup& check_group = program_.operation_groups[check_op];
		for (const auto& reset : check_group.resets) {
			if (reset.qubit_index != qubit) {
				continue;
			}
			const double p_reset_success = 1.0 - reset.reset_failure_probability;
			const double p_x = p_reset_success * reset.reset_probability_channel.p_x;
			const double p_y = p_reset_success * reset.reset_probability_channel.p_y;
			const double p_z = p_reset_success * reset.reset_probability_channel.p_z;
			if (p_x <= 0.0 && p_y <= 0.0 && p_z <= 0.0) {
				continue;
			}
			const uint32_t emit_op_index = op_to_emit_op_index_[check_op];
			buckets[emit_op_index].push_back({emit_op_index, qubit, p_x, p_y, p_z});
		}

		const uint32_t start_offset = find_start_qubit_op_offset(program_, qubit, link.reset_op_after_lookback);
		const uint32_t end_offset = find_end_qubit_op_offset(program_, qubit, check_op);
		if (start_offset > end_offset) {
			throw std::logic_error("decoded lookback operation window is inverted");
		}
		const std::vector<uint32_t>& qubit_skippable_ops =
			program_.qubit_skippable_operation_indices.at(qubit);
		for (uint32_t emit_op_index : touched_emit_ops) {
			source_emit_channels[emit_op_index].clear();
		}
		touched_emit_ops.clear();

		double p_unerased = 1.0;
		// Likelihood of the observed check pattern in this window under a no-erasure path.
		// For fixed false-positive rate x and a checks, this reduces to (1-x)^(a-1) * x.
		double no_erasure_check_likelihood = 1.0;
		std::vector<OnsetBranch> onset_branches;
		onset_branches.reserve(end_offset - start_offset + 1);

		for (uint32_t offset = start_offset; offset <= end_offset; ++offset) {
			const uint32_t op_index = program_.qubit_operation_indices.at(qubit)[offset];
			const circuit::OperationGroup& op_group = program_.operation_groups[op_index];

			for (const auto& onset : op_group.onsets) {
				if (onset.qubit_index == qubit) {
					const double p_erase = p_unerased * onset.probability;
					onset_branches.push_back({op_index, p_erase, 0, false, 0});
					p_unerased -= p_erase;
				}
			}

			for (const auto& onset_pair : op_group.onset_pairs) {
				if (onset_pair.qubit_index1 == qubit || onset_pair.qubit_index2 == qubit) {
					// ERASE2_ANY erases exactly one of the pair uniformly, so this qubit has
					// half of the pair-onset probability mass.
					const double p_pair = onset_pair.probability;
					const double p_erase = p_unerased * (0.5 * p_pair);
					const uint32_t other_qubit =
						(onset_pair.qubit_index1 == qubit) ? onset_pair.qubit_index2 : onset_pair.qubit_index1;
					onset_branches.push_back({op_index, p_erase, 0, true, other_qubit});
					p_unerased -= p_erase;
				}
			}

			// For candidate onset branches:
			// - intermediate checks in the window must be missed (multiply by false-negative prob q)
			// - the flagged check itself multiplies by (1 - q), unless forced-detection applies at max persistence
			for (const auto& check : op_group.checks) {
				if (check.qubit_index == qubit) {
					const double false_negative_prob = check.false_negative_probability;
					const double false_positive_prob = check.false_positive_probability;
					const bool is_flagged_check = (op_index == check_op);
					for (auto& branch : onset_branches) {
						if (!is_flagged_check) {
							branch.probability *= false_negative_prob;
							branch.survived_checks++;
							continue;
						}

						bool forced_detection = false;
						if (program_.max_persistence() > 0) {
							const uint32_t checks_before_forced_detection = program_.max_persistence() - 1;
							forced_detection = branch.survived_checks >= checks_before_forced_detection;
						}
						const double flag_prob = forced_detection ? 1.0 : (1.0 - false_negative_prob);
						branch.probability *= flag_prob;
					}
					// Under no-erasure, non-flagged checks are true-negatives and the flagged
					// check is a false-positive.
					no_erasure_check_likelihood *=
						is_flagged_check ? false_positive_prob : (1.0 - false_positive_prob);
					break;
				}
			}
		}

		double normalizer = 0.0;
		for (const auto& branch : onset_branches) {
			normalizer += branch.probability;
		}
		const double no_erasure_mass = p_unerased * no_erasure_check_likelihood;
		normalizer += no_erasure_mass;
		if (normalizer > 0.0) {
			for (auto& branch : onset_branches) {
				branch.probability /= normalizer;
			}
		}

		if (verbose) {
			std::cout << " [surf_dem_builder] qubit=" << qubit
					  << " check_op=" << check_op
					  << " start_offset=" << program_.qubit_operation_indices.at(qubit)[start_offset]
					  << " end_offset=" << program_.qubit_operation_indices.at(qubit)[end_offset]
					  << " lookback_check=" << link.lookback_check_event_index
					  << " reset_after_lookback=" << link.reset_op_after_lookback
					  << " no_erasure_mass=" << no_erasure_mass << "\n";
			if (onset_branches.empty()) {
				std::cout << "  posterior_onset_probs: (none)\n";
			} else {
				std::cout << std::fixed << std::setprecision(6);
				for (const auto& branch : onset_branches) {
					std::cout << "  op=" << branch.op_index
							  << " posterior=" << branch.probability << "\n";
				}
			}
		}

		if (normalizer <= 0.0) {
			continue;
		}

		double p_erased_by_op = 0.0;
		size_t branch_index = 0;
		size_t skippable_index = 0;
		const uint32_t start_op_index = program_.qubit_operation_indices.at(qubit)[start_offset];
		while (skippable_index < qubit_skippable_ops.size() &&
			   qubit_skippable_ops[skippable_index] < start_op_index) {
			++skippable_index;
		}
		const auto merge_into_emit_bucket =
			[&source_emit_channels, &touched_emit_ops](uint32_t emit_op_index, uint32_t target_qubit, double p_x,
									double p_y, double p_z) {
				if (p_x <= 0.0 && p_y <= 0.0 && p_z <= 0.0) {
					return;
				}
				std::vector<LocalTargetChannelAccum>& bucket = source_emit_channels[emit_op_index];
				for (auto& entry : bucket) {
					if (entry.target_qubit == target_qubit) {
						entry.p_x += p_x;
						entry.p_y += p_y;
						entry.p_z += p_z;
						return;
					}
				}
				if (bucket.empty()) {
					touched_emit_ops.push_back(emit_op_index);
				}
				bucket.push_back({target_qubit, p_x, p_y, p_z});
			};
		for (uint32_t offset = start_offset; offset <= end_offset; ++offset) {
			const uint32_t op_index = program_.qubit_operation_indices.at(qubit)[offset];
			std::vector<LocalTargetChannelAccum> local_channels;
			double p_onset_at_op = 0.0;
			const auto accumulate_channel = [&local_channels](
											 uint32_t target_qubit, double p_x, double p_y, double p_z) {
				if (p_x <= 0.0 && p_y <= 0.0 && p_z <= 0.0) {
					return;
				}
				for (auto& entry : local_channels) {
					if (entry.target_qubit == target_qubit) {
						entry.p_x += p_x;
						entry.p_y += p_y;
						entry.p_z += p_z;
						return;
					}
				}
				local_channels.push_back({target_qubit, p_x, p_y, p_z});
			};

			while (branch_index < onset_branches.size() &&
				   onset_branches[branch_index].op_index <= op_index) {
				const OnsetBranch& branch = onset_branches[branch_index];
				p_erased_by_op += branch.probability;
				if (branch.op_index == op_index) {
					p_onset_at_op += branch.probability;
				}
				if (branch.from_onset_pair) {
					const PauliChannel& onset_channel = program_.model().onset;
					const double onset_px = branch.probability * onset_channel.p_x;
					const double onset_py = branch.probability * onset_channel.p_y;
					const double onset_pz = branch.probability * onset_channel.p_z;
					accumulate_channel(branch.onset_pair_target, onset_px, onset_py, onset_pz);
				}
				branch_index++;
			}
			while (skippable_index < qubit_skippable_ops.size() &&
				   qubit_skippable_ops[skippable_index] < op_index) {
				++skippable_index;
			}
			if (skippable_reweights != nullptr &&
				skippable_index < qubit_skippable_ops.size() &&
				qubit_skippable_ops[skippable_index] == op_index) {
				const double p_unerased_by_op = std::clamp(1.0 - p_erased_by_op, 0.0, 1.0);
				const uint64_t key = make_op_qubit_key(op_index, qubit);
				const auto existing_it = skippable_reweights->find(key);
				if (existing_it == skippable_reweights->end() ||
					p_unerased_by_op < existing_it->second) {
					(*skippable_reweights)[key] = p_unerased_by_op;
				}
			}

			if (p_onset_at_op > 0.0) {
				const circuit::OperationGroup& op_group = program_.operation_groups[op_index];
				for (const auto& spread : op_group.onset_spreads) {
					if (spread.source_qubit_index != qubit) {
						continue;
					}

					const double p_x = p_onset_at_op * spread.spread_probability_channel.p_x;
					const double p_y = p_onset_at_op * spread.spread_probability_channel.p_y;
					const double p_z = p_onset_at_op * spread.spread_probability_channel.p_z;
					accumulate_channel(spread.aff_qubit_index, p_x, p_y, p_z);
				}
			}

			if (p_erased_by_op > 0.0) {
				const circuit::OperationGroup& op_group = program_.operation_groups[op_index];
				for (const auto& spread : op_group.persistent_spreads) {
					if (spread.source_qubit_index != qubit) {
						continue;
					}

					const double p_x = p_erased_by_op *
									   spread.spread_probability_channel.p_x;
					const double p_y = p_erased_by_op *
									   spread.spread_probability_channel.p_y;
					const double p_z = p_erased_by_op *
									   spread.spread_probability_channel.p_z;
					accumulate_channel(spread.aff_qubit_index, p_x, p_y, p_z);
				}
			}

			// Measurement randomization translation:
			// inject X before measuring qubits that may be erased.
			// Requested model: P(X before measurement) = 0.5 * P(erased by that point).
			const circuit::OperationGroup& op_group = program_.operation_groups[op_index];
			if (op_group.stim_instruction.has_value() &&
				circuit::is_measurement_op(op_group.stim_instruction->op) &&
				p_erased_by_op > 0.0) {
				// op_index comes from qubit_operation_indices for this qubit, so measurement
				// targets necessarily include `qubit`.
				const double p_meas_x = 0.5 * p_erased_by_op;
				if (p_meas_x > 0.0) {
					const uint32_t pre_emit_op_index =
						(op_index == 0) ? op_to_emit_op_index_[op_index]
										: op_to_emit_op_index_[op_index - 1];
					merge_into_emit_bucket(pre_emit_op_index, qubit, p_meas_x, 0.0, 0.0);
				}
			}

			const uint32_t emit_op_index = op_to_emit_op_index_[op_index];
			for (const auto& entry : local_channels) {
				merge_into_emit_bucket(
					emit_op_index, entry.target_qubit, entry.p_x, entry.p_y, entry.p_z);
			}
		}

		for (uint32_t emit_op_index : touched_emit_ops) {
			for (const auto& entry : source_emit_channels[emit_op_index]) {
				buckets[emit_op_index].push_back(
					{emit_op_index, entry.target_qubit, entry.p_x, entry.p_y, entry.p_z});
			}
		}
	}

	// Tail correction for missed final checks:
	// estimate per-qubit no-erasure probability up to the final measurement by
	// carrying forward:
	// - no-onset factors (1 - p_onset) over onset opportunities, and
	// - no-erasure check-likelihood factors from observed check bits.
	//
	// This captures cases where a qubit remains erased despite an unflagged final check.
	const uint32_t num_qubits = program_.max_qubit_index() + 1;
	for (uint32_t qubit = 0; qubit < num_qubits; ++qubit) {
		const std::vector<uint32_t>& qubit_ops = program_.qubit_operation_indices.at(qubit);
		if (qubit_ops.empty()) {
			continue;
		}

		// Find final measurement operation touching this qubit.
		int32_t final_meas_op = -1;
		for (auto it = qubit_ops.rbegin(); it != qubit_ops.rend(); ++it) {
			const uint32_t op_index = *it;
			const circuit::OperationGroup& op_group = program_.operation_groups[op_index];
			if (!op_group.stim_instruction.has_value()) {
				continue;
			}
			const circuit::Instruction& instr = op_group.stim_instruction.value();
			if (!circuit::is_measurement_op(instr.op)) {
				continue;
			}
			if (std::find(instr.targets.begin(), instr.targets.end(), qubit) == instr.targets.end()) {
				continue;
			}
			final_meas_op = static_cast<int32_t>(op_index);
			break;
		}
		if (final_meas_op < 0) {
			continue;
		}

		uint32_t start_op = 0;
		const std::vector<uint32_t>& local_check_events = qubit_check_events_[qubit];
		const std::vector<uint32_t>& local_check_ops = program_.qubit_check_operation_indices.at(qubit);
		if (!local_check_events.empty()) {
			const uint32_t last_local = static_cast<uint32_t>(local_check_events.size() - 1);
			uint32_t lookback_local = 0;
			if (program_.max_persistence() <= last_local) {
				lookback_local = last_local - program_.max_persistence() + 1;
			}
			const uint32_t lookback_event = local_check_events[lookback_local];
			uint32_t lookback_op = check_event_to_op_index_[lookback_event];

			int32_t latest_flagged_event = -1;
			for (const uint32_t event_idx : local_check_events) {
				if ((*check_results)[event_idx] == 1) {
					latest_flagged_event = static_cast<int32_t>(event_idx);
				}
			}
			if (latest_flagged_event >= 0) {
				const uint32_t flagged_op =
					check_event_to_op_index_[static_cast<uint32_t>(latest_flagged_event)];
				// Start after the latest flagged check/reset point.
				lookback_op = std::max(lookback_op, flagged_op + 1);
			}
			start_op = lookback_op;
		}

		add_tail_hidden_injections(
			check_results,
			qubit,
			start_op,
			static_cast<uint32_t>(final_meas_op),
			&buckets,
			skippable_reweights);
	}

	return buckets;
}

stim::Circuit SurfDemBuilder::build_decoded_circuit(
	const std::vector<uint8_t>* check_results,
	bool verbose) const {
	SkippableReweightMap skippable_reweights;
	SpreadInjectionBuckets buckets =
		compute_spread_injections(check_results, verbose, &skippable_reweights);

	stim::Circuit injected;
	for (uint32_t op_index = 0; op_index < program_.operation_groups.size(); ++op_index) {
		const circuit::OperationGroup& op_group = program_.operation_groups[op_index];
		if (op_group.stim_instruction.has_value()) {
			const circuit::Instruction& instr = op_group.stim_instruction.value();
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
			if (p_x > 0.0 || p_y > 0.0 || p_z > 0.0) {
				injected.safe_append_u(
					"PAULI_CHANNEL_1", {event.target_qubit}, {p_x, p_y, p_z});
			}
		}
	}

	return injected;
}

std::string SurfDemBuilder::build_decoded_circuit_text(
	const std::vector<uint8_t>* check_results,
	bool verbose) const {
	SkippableReweightMap skippable_reweights;
	SpreadInjectionBuckets buckets =
		compute_spread_injections(check_results, verbose, &skippable_reweights);

	std::ostringstream out;
	bool first_line = true;
	for (uint32_t op_index = 0; op_index < program_.operation_groups.size(); ++op_index) {
		const circuit::OperationGroup& op_group = program_.operation_groups[op_index];
		if (op_group.stim_instruction.has_value()) {
			const auto& instr = op_group.stim_instruction.value();
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
					// Stim requires an observable index argument; qerasure uses index 0.
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
