#include "core/decode/surf_hmm_decoder.h"
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

}  // namespace

SurfHMMDecoder::SurfHMMDecoder(const circuit::CompiledErasureProgram& program) : program_(program) {

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

	if (check_event_to_qubit_.size() != program_.num_checks()) {
		throw std::logic_error("SurfHMMDecoder check-event map size mismatch with CompiledErasureProgram");
	}
	if (program_.check_lookback_links.size() != program_.num_checks()) {
		throw std::logic_error("SurfHMMDecoder expected check_lookback_links to match num_checks");
	}
}

SpreadInjectionBuckets SurfHMMDecoder::compute_spread_injections(
	const std::vector<uint8_t>* check_results,
	bool verbose) const {
	SpreadInjectionBuckets buckets(program_.operation_groups.size());

	if (check_results == nullptr) {
		throw std::invalid_argument(
			"SurfHMMDecoder::compute_spread_injections requires non-null check_results pointer");
	}
	if (check_results->size() != program_.num_checks()) {
		throw std::invalid_argument(
			"SurfHMMDecoder::compute_spread_injections check_results size mismatch");
	}

	for (uint32_t check_event_index = 0; check_event_index < check_results->size(); ++check_event_index) {
		const uint8_t bit = (*check_results)[check_event_index]; // 0: non-flagged, 1: flagged
		if (bit == 0) {
			continue;
		}
		if (bit != 1) {
			throw std::invalid_argument("SurfHMMDecoder::decode expects binary check results");
		}

		const circuit::CheckLookbackLink& link = program_.check_lookback_links.at(check_event_index);
		const uint32_t qubit = check_event_to_qubit_[check_event_index];
		const uint32_t check_op = check_event_to_op_index_[check_event_index];
		if (link.qubit_index != qubit || link.check_op_index != check_op) {
			throw std::logic_error("check_lookback_links metadata mismatch with decoder check maps");
		}

		// For every flagged check, apply reset-channel contribution on that same qubit.
		// Runtime sampler semantics: reset noise is applied only if reset failure occurs.
		// Therefore net injected probabilities are p_fail * (p_x, p_y, p_z).
		const circuit::OperationGroup& check_group = program_.operation_groups[check_op];
		for (const auto& reset : check_group.resets) {
			if (reset.qubit_index != qubit) {
				continue;
			}
			const double p_x = reset.reset_failure_probability * reset.reset_probability_channel.p_x;
			const double p_y = reset.reset_failure_probability * reset.reset_probability_channel.p_y;
			const double p_z = reset.reset_failure_probability * reset.reset_probability_channel.p_z;
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

		double p_unerased = 1.0;
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
					break;
				}
			}
		}

		double normalizer = 0.0;
		for (const auto& branch : onset_branches) {
			normalizer += branch.probability;
		}
		if (normalizer > 0.0) {
			for (auto& branch : onset_branches) {
				branch.probability /= normalizer;
			}
		}

		if (verbose) {
			std::cout << " [surf_hmm] qubit=" << qubit
					  << " check_op=" << check_op
					  << " start_offset=" << program_.qubit_operation_indices.at(qubit)[start_offset]
					  << " end_offset=" << program_.qubit_operation_indices.at(qubit)[end_offset]
					  << " lookback_check=" << link.lookback_check_event_index
					  << " reset_after_lookback=" << link.reset_op_after_lookback << "\n";
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
		std::vector<std::vector<LocalTargetChannelAccum>> source_emit_channels(
			program_.operation_groups.size());
		const auto merge_into_emit_bucket =
			[&source_emit_channels](uint32_t emit_op_index, uint32_t target_qubit, double p_x,
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

			const uint32_t emit_op_index = op_to_emit_op_index_[op_index];
			for (const auto& entry : local_channels) {
				merge_into_emit_bucket(
					emit_op_index, entry.target_qubit, entry.p_x, entry.p_y, entry.p_z);
			}
		}

		for (uint32_t emit_op_index = 0; emit_op_index < source_emit_channels.size(); ++emit_op_index) {
			for (const auto& entry : source_emit_channels[emit_op_index]) {
				buckets[emit_op_index].push_back(
					{emit_op_index, entry.target_qubit, entry.p_x, entry.p_y, entry.p_z});
			}
		}
	}

	return buckets;
}

stim::Circuit SurfHMMDecoder::decode(
	const std::vector<uint8_t>* check_results,
	bool verbose) const {
	SpreadInjectionBuckets buckets =
		compute_spread_injections(check_results, verbose);

	stim::Circuit injected;
	for (uint32_t op_index = 0; op_index < program_.operation_groups.size(); ++op_index) {
		const circuit::OperationGroup& op_group = program_.operation_groups[op_index];
		if (op_group.stim_instruction.has_value()) {
			simulator::internal::append_mapped_stim_instruction(
				op_group.stim_instruction.value(), &injected);
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

std::string SurfHMMDecoder::debug_decoded_circuit_text(
	const std::vector<uint8_t>* check_results,
	bool verbose) const {
	SpreadInjectionBuckets buckets =
		compute_spread_injections(check_results, verbose);

	std::ostringstream out;
	bool first_line = true;
	for (uint32_t op_index = 0; op_index < program_.operation_groups.size(); ++op_index) {
		const circuit::OperationGroup& op_group = program_.operation_groups[op_index];
		if (op_group.stim_instruction.has_value()) {
			const auto& instr = op_group.stim_instruction.value();
			if (!first_line) {
				out << "\n";
			}
			first_line = false;
			out << circuit::opcode_name(instr.op);
			if (circuit::is_probabilistic_op(instr.op)) {
				out << "(" << instr.arg << ")";
			}
			for (const uint32_t target : instr.targets) {
				out << " " << target;
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
