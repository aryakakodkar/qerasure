#include <iostream>
#include <unordered_map>

#include "compile.h"
#include "instruction.h"

namespace qerasure::circuit {

namespace{
    uint64_t probability_to_threshold(double p) {
        if (p < 0.0 || p > 1.0) {
            throw std::invalid_argument("Probability must be between 0 and 1.");
        }
        if (p == 0.0) {
            return 0;
        }
        if (p == 1.0) {
            return std::numeric_limits<std::uint64_t>::max();
        }
        return static_cast<std::uint64_t>(p * static_cast<long double>(std::numeric_limits<std::uint64_t>::max()));
    }

    // For debugging purposes
    double threshold_to_probability(uint64_t threshold) {
        if (threshold == 0) {
            return 0.0;
        }
        if (threshold == std::numeric_limits<std::uint64_t>::max()) {
            return 1.0;
        }
        return static_cast<double>(threshold) / static_cast<long double>(std::numeric_limits<std::uint64_t>::max());
    }

    void check_max_qubit_index(const std::vector<uint32_t>& targets, uint32_t& max_index) {
        for (const auto& target : targets) {
            if (target > max_index) {
                max_index = target;
            }
        }
    }

    void append_unique_op_index(std::vector<std::vector<uint32_t>>* per_qubit_indices,
                                uint32_t qubit,
                                uint32_t op_index) {
        std::vector<uint32_t>& indices = (*per_qubit_indices)[qubit];
        if (indices.empty() || indices.back() != op_index) {
            indices.push_back(op_index);
        }
    }
}

// TODO: Need to find a better way to convey false negative, false positive probabilities for checks.
// Currently requires model to outlast CompiledErasureProgram, which is awkward.
CompiledErasureProgram::CompiledErasureProgram(const ErasureCircuit& circuit, const ErasureModel& model)
    : model_(model), thresholded_onset_(model.onset) {
    operation_groups.resize(circuit.instructions().size());

    std::unordered_map<uint32_t, uint32_t> checks_survived; // maps erased qubit index to number of checks through which erasure has gone undetected
    
    // Pre-compute thresholded channels and probs
    ThresholdedPauliChannel thresholded_onset(model.onset);
    ThresholdedPauliChannel thresholded_reset(model.reset);
    ThresholdedPauliChannel thresholded_control_spread(model.spread.control_spread);
    ThresholdedPauliChannel thresholded_target_spread(model.spread.target_spread);
    const PauliChannel onset_probability_channel = model.onset;
    const PauliChannel reset_probability_channel = model.reset;
    const PauliChannel control_spread_probability_channel = model.spread.control_spread;
    const PauliChannel target_spread_probability_channel = model.spread.target_spread;
    uint64_t check_false_negative_threshold = probability_to_threshold(model.check_false_negative_prob);
    uint64_t check_false_positive_threshold = probability_to_threshold(model.check_false_positive_prob);

    max_persistence_ = model.max_persistence;

    // Pre-scan once to size thin per-qubit index layers.
    for (const auto& instr : circuit.instructions()) {
        if (!uses_measurement_record_targets(instr.op)) {
            check_max_qubit_index(instr.targets, max_qubit_index_);
        }
    }
    qubit_operation_indices.resize(max_qubit_index_ + 1);
    qubit_check_operation_indices.resize(max_qubit_index_ + 1);
    qubit_reset_operation_indices.resize(max_qubit_index_ + 1);
    qubit_last_check_operation_index.assign(max_qubit_index_ + 1, -1);
    std::vector<std::vector<uint32_t>> qubit_check_event_indices(max_qubit_index_ + 1);
    
    // TODO: Need to check if qubits that might be erased are involved in ERROR ops or MEASUREMENTS
    uint32_t op_index = 0;
    for (const auto& instr : circuit.instructions()) {
        OperationGroup& group = operation_groups[op_index]; // group of operations for this timestep
        const auto mark_qubit_operation = [&](uint32_t qubit) {
            append_unique_op_index(&qubit_operation_indices, qubit, op_index);
        };
        const auto mark_qubit_check = [&](uint32_t qubit) {
            append_unique_op_index(&qubit_check_operation_indices, qubit, op_index);
        };
        const auto mark_qubit_reset = [&](uint32_t qubit) {
            append_unique_op_index(&qubit_reset_operation_indices, qubit, op_index);
        };

        if (is_stim_op(instr.op)) {
            group.stim_instruction = instr;
            if (!uses_measurement_record_targets(instr.op)) {
                for (const auto& target : instr.targets) {
                    mark_qubit_operation(target);
                }
            }
            if (is_entangling_op(instr.op)) {
                for (size_t i = 0; i < instr.targets.size(); i += 2) {
                    uint32_t control = instr.targets[i];
                    uint32_t target = instr.targets[i + 1];
                    if (checks_survived.find(control) != checks_survived.end()) {
                        group.persistent_spreads.push_back(
                            {control, target, thresholded_control_spread, control_spread_probability_channel});
                    }
                    if (checks_survived.find(target) != checks_survived.end()) {
                        group.persistent_spreads.push_back(
                            {target, control, thresholded_target_spread, target_spread_probability_channel});
                    }
                }
            }
        } else if (is_single_onset_op(instr.op)) {
            for (const auto& target : instr.targets) {
                group.onsets.push_back({target, probability_to_threshold(instr.arg), instr.arg});
                erasable_qubits_.push_back(target);
                checks_survived[target] = 0;
                mark_qubit_operation(target);
            }
        } else if (is_multi_onset_op(instr.op)) {
            if (instr.op == OpCode::ERASE2) {
                for (size_t i = 0; i < instr.targets.size(); i += 2) {
                    uint32_t target1 = instr.targets[i]; // to be erased
                    uint32_t target2 = instr.targets[i + 1]; // affected by onset spread
                    group.onsets.push_back({target1, probability_to_threshold(instr.arg), instr.arg});
                    erasable_qubits_.push_back(target1);
                    group.onset_spreads.push_back(
                        {target1, target2, thresholded_onset, onset_probability_channel});
                    checks_survived[target1] = 0;
                    mark_qubit_operation(target1);
                    mark_qubit_operation(target2);
                }
            } else if (instr.op == OpCode::ERASE2_ANY) {
                for (size_t i = 0; i < instr.targets.size(); i += 2) {
                    uint32_t target1 = instr.targets[i];
                    uint32_t target2 = instr.targets[i + 1];
                    group.onset_pairs.push_back(
                        {target1, target2, probability_to_threshold(instr.arg), instr.arg});
                    erasable_qubits_.push_back(target1);
                    erasable_qubits_.push_back(target2);
                    checks_survived[target1] = 0;
                    checks_survived[target2] = 0;
                    mark_qubit_operation(target1);
                    mark_qubit_operation(target2);
                }
            }
        } else if (is_erasure_check_op(instr.op)) {
            for (const auto& target : instr.targets) {
                if (checks_survived.find(target) == checks_survived.end()) {
                    continue;
                }
                checks_survived[target]++;
                group.checks.push_back(
                    {target,
                     check_false_negative_threshold,
                     check_false_positive_threshold,
                     model.check_false_negative_prob,
                     model.check_false_positive_prob});
                qubit_last_check_operation_index[target] = static_cast<int32_t>(op_index);
                qubit_check_event_indices[target].push_back(num_checks_);
                num_checks_++;
                mark_qubit_operation(target);
                mark_qubit_check(target);
            }
        // Erasure and reset are not mutually exclusive (e.g. ECR), so both need to be processed if applicable
        } if (is_erasure_reset_op(instr.op)) {
            for (const auto& target : instr.targets) {
                // check if qubit can be erased and was checked since last erase op.
                if (checks_survived.find(target) == checks_survived.end()
                    || checks_survived[target] == 0) {
                    continue;
                } else if (checks_survived[target] >= max_persistence_) {
                    group.resets.push_back(
                        {target, 0, thresholded_reset, 0.0, reset_probability_channel}); // if max persistence exceeded, reset is guaranteed to succeed
                    checks_survived.erase(target);
                    mark_qubit_operation(target);
                    mark_qubit_reset(target);
                    continue;
                }
                group.resets.push_back(
                    {target,
                     probability_to_threshold(instr.arg),
                     thresholded_reset,
                     instr.arg,
                     reset_probability_channel});
                mark_qubit_operation(target);
                mark_qubit_reset(target);
            }
        }
        group.op_num = group.onsets.size() + group.onset_pairs.size() + group.onset_spreads.size() +
                       group.persistent_spreads.size() + group.checks.size() + group.resets.size();
        op_index++;
    }

    // Build per-check lookback links:
    // for local check index n on a qubit, link to (n - max_persistence_) and
    // to the first reset op-index at/after that lookback check op-index.
    check_lookback_links.assign(num_checks_, {0, 0, -1, -1});
    uint32_t global_check_event_index = 0;
    for (uint32_t group_op_index = 0; group_op_index < operation_groups.size(); ++group_op_index) {
        const OperationGroup& group = operation_groups[group_op_index];
        for (const ErasureCheck& check : group.checks) {
            check_lookback_links[global_check_event_index].qubit_index = check.qubit_index;
            check_lookback_links[global_check_event_index].check_op_index = group_op_index;
            global_check_event_index++;
        }
    }

    for (uint32_t qubit = 0; qubit < qubit_check_event_indices.size(); ++qubit) {
        const std::vector<uint32_t>& local_check_events = qubit_check_event_indices[qubit];
        const std::vector<uint32_t>& local_check_ops = qubit_check_operation_indices[qubit];
        const std::vector<uint32_t>& local_reset_ops = qubit_reset_operation_indices[qubit];
        if (local_check_events.empty()) {
            continue;
        }

        for (uint32_t local_idx = 0; local_idx < local_check_events.size(); ++local_idx) {
            const uint32_t check_event = local_check_events[local_idx];
            CheckLookbackLink& link = check_lookback_links[check_event];
            if (max_persistence_ > local_idx) {
                // No (n - max_persistence_) check for this local event.
                continue;
            }

            const uint32_t lookback_local_idx = local_idx - max_persistence_;
            const uint32_t lookback_event = local_check_events[lookback_local_idx];
            const uint32_t lookback_op = local_check_ops[lookback_local_idx];
            link.lookback_check_event_index = static_cast<int32_t>(lookback_event);

            const auto reset_it =
                std::lower_bound(local_reset_ops.begin(), local_reset_ops.end(), lookback_op);
            if (reset_it != local_reset_ops.end()) {
                link.reset_op_after_lookback = static_cast<int32_t>(*reset_it);
            }
        }
    }
}

// Debug helper
void CompiledErasureProgram::print_summary() const {
    std::cout << "Compiled Erasure Program Summary:\n";
    std::cout << "Total Instructions: " << operation_groups.size() << "\n";
    
    for (size_t i = 0; i < operation_groups.size(); ++i) {
        std::cout << "====== INSTRUCTION " << i << " ======\n";
        if (operation_groups[i].stim_instruction.has_value()) {
            const auto& instr = operation_groups[i].stim_instruction.value();
            std::cout << "  Stim Instruction: op=" << static_cast<int>(instr.op)
                      << ", targets=[";
            for (size_t j = 0; j < instr.targets.size(); ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << instr.targets[j];
            }
            std::cout << "], arg=" << instr.arg << "\n";
        }
        for (size_t j = 0; j < operation_groups[i].onsets.size(); ++j) {
            const auto& onset = operation_groups[i].onsets[j];
            std::cout << "  Onset - Qubit: " << onset.qubit_index 
                      << ", Probability: " << onset.probability << "\n";
        }
        for (size_t j = 0; j < operation_groups[i].checks.size(); ++j) {
            const auto& check = operation_groups[i].checks[j];
            std::cout << "  Check - Qubit: " << check.qubit_index 
                      << ", False Negative: " << check.false_negative_probability
                      << ", False Positive: " << check.false_positive_probability << "\n";
        }
        for (size_t j = 0; j < operation_groups[i].resets.size(); ++j) {
            const auto& reset = operation_groups[i].resets[j];
            std::cout << "  Reset - Qubit: " << reset.qubit_index
                      << ", Reset Failure Prob: " << reset.reset_failure_probability
                      << ", Reset Channel: (X: " << reset.reset_probability_channel.p_x
                      << ", Y: " << reset.reset_probability_channel.p_y
                      << ", Z: " << reset.reset_probability_channel.p_z << ")\n";
        }
        for (size_t j = 0; j < operation_groups[i].onset_spreads.size(); ++j) {
            const auto& spread = operation_groups[i].onset_spreads[j];
            std::cout << "  Onset Spread - Source Qubit: " << spread.source_qubit_index
                      << ", Affected Qubit: " << spread.aff_qubit_index
                      << ", Channel: (X: " << spread.spread_probability_channel.p_x
                      << ", Y: " << spread.spread_probability_channel.p_y
                      << ", Z: " << spread.spread_probability_channel.p_z << ")\n";
        }
        for (size_t j = 0; j < operation_groups[i].persistent_spreads.size(); ++j) {
            const auto& spread = operation_groups[i].persistent_spreads[j];
            std::cout << "  Persistent Spread - Source Qubit: " << spread.source_qubit_index
                      << ", Affected Qubit: " << spread.aff_qubit_index
                      << ", Channel: (X: " << spread.spread_probability_channel.p_x
                      << ", Y: " << spread.spread_probability_channel.p_y
                      << ", Z: " << spread.spread_probability_channel.p_z << ")\n";
        }
    }
}

}  // namespace qerasure::circuit
