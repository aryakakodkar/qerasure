#include <cstdint>
#include <algorithm>
#include <optional>
#include <sstream>
#include <string>

#include "core/model/pauli_channel.h"
#include "core/simulator/fast_rng.h"
#include "core/simulator/erasure_sampler.h"
#include "core/circuit/compile.h"
#include "core/simulator/sim_internal_utils.h"

namespace qerasure::simulator {

namespace {

PauliOperation from_internal_pauli_operation(internal::PauliOperation op) {
    switch (op) {
        case internal::PauliOperation::X:
            return PauliOperation::X;
        case internal::PauliOperation::Y:
            return PauliOperation::Y;
        case internal::PauliOperation::Z:
            return PauliOperation::Z;
        case internal::PauliOperation::I:
            return PauliOperation::I;
    }
    return PauliOperation::I;
}

const char* pauli_operation_name(PauliOperation op) {
    switch (op) {
        case PauliOperation::X:
            return "X";
        case PauliOperation::Y:
            return "Y";
        case PauliOperation::Z:
            return "Z";
        case PauliOperation::I:
            return "I";
    }
    return "UNKNOWN";
}

const char* check_outcome_name(CheckOutcome outcome) {
    switch (outcome) {
        case CheckOutcome::TruePositive:
            return "TruePositive";
        case CheckOutcome::FalseNegative:
            return "FalseNegative";
        case CheckOutcome::FalsePositive:
            return "FalsePositive";
        case CheckOutcome::TrueNegative:
            return "TrueNegative";
    }
    return "UNKNOWN";
}

void append_instruction_summary(std::ostringstream* out, const circuit::Instruction& instr) {
    *out << "Stim Instruction: " << circuit::opcode_name(instr.op);
    if (circuit::is_probabilistic_op(instr.op)) {
        *out << "(" << instr.arg << ")";
    }
    for (const uint32_t target : instr.targets) {
        *out << " " << target;
    }
    *out << "\n";
}

}  // namespace

std::string SampledShot::to_string() const {
    std::ostringstream out;
    for (size_t op_index = 0; op_index < operation_groups.size(); ++op_index) {
        const SampledOperationGroup& group = operation_groups[op_index];
        out << "=== OP NUM (" << op_index << ") ===\n";

        if (group.stim_instruction.has_value()) {
            append_instruction_summary(&out, group.stim_instruction.value());
        } else {
            out << "Stim Instruction: (none)\n";
        }

        out << "Onsets:\n";
        if (group.onsets.empty()) {
            out << "  (none)\n";
        } else {
            for (const SampledOnset& onset : group.onsets) {
                out << "  q=" << onset.qubit_index << "\n";
            }
        }

        out << "Spreads:\n";
        if (group.spreads.empty()) {
            out << "  (none)\n";
        } else {
            for (const SampledSpread& spread : group.spreads) {
                out << "  q=" << spread.qubit_index
                    << ", op=" << pauli_operation_name(spread.operation) << "\n";
            }
        }

        out << "Checks:\n";
        if (group.checks.empty()) {
            out << "  (none)\n";
        } else {
            for (const SampledCheck& check : group.checks) {
                out << "  q=" << check.qubit_index
                    << ", outcome=" << check_outcome_name(check.outcome) << "\n";
            }
        }

        out << "Resets:\n";
        if (group.resets.empty()) {
            out << "  (none)\n";
        } else {
            for (const SampledReset& reset : group.resets) {
                out << "  q=" << reset.qubit_index
                    << ", op=" << pauli_operation_name(reset.operation) << "\n";
            }
        }

        if (op_index + 1 < operation_groups.size()) {
            out << "\n";
        }
    }
    return out.str();
}

// TODO: Add Stim instruction parsing to remove error ops on non-erased qubits
SampledBatch ErasureSampler::sample(const SamplerParams& params) {
    SampledBatch batch;
    batch.shots.reserve(params.shots);
    size_t num_ops = program_.operation_groups.size();

    // Shift to uint64_t representation of max persistence for conditional check functionality.
    // Once a qubit reaches this check-survival count, the next check is forced true-positive.
    uint64_t max_persistence = static_cast<uint64_t>(program_.max_persistence());

    FastRng rng_(params.seed);

    // Current erasure state holds the check survival count as well
    // 0: unerased, 1: newly erased, n (1 < n): erased and survived n-1 checks
    std::vector<uint64_t> current_erasure_state(program_.max_qubit_index() + 1, 0);
    
    std::vector<uint8_t> last_check_result(program_.max_qubit_index() + 1, 0); // 0: no check or negative, 1: positive

    for (std::uint64_t shot_idx = 0; shot_idx < params.shots; ++shot_idx) {
        SampledShot shot;
        std::fill(current_erasure_state.begin(), current_erasure_state.end(), 0); // reset erasure state for new shot
        std::fill(last_check_result.begin(), last_check_result.end(), 0); // reset last check results for new shot
        shot.operation_groups.resize(num_ops);

        for (size_t op_index = 0; op_index < num_ops; ++op_index) {
            const circuit::OperationGroup& op_group = program_.operation_groups[op_index];
            SampledOperationGroup& group = shot.operation_groups[op_index];

            if (op_group.stim_instruction.has_value()) {
                group.stim_instruction = op_group.stim_instruction;
            }

            group.onsets.reserve(op_group.onsets.size());
            group.spreads.reserve(op_group.onset_spreads.size() + op_group.persistent_spreads.size());
            group.checks.reserve(op_group.checks.size());
            group.resets.reserve(op_group.resets.size());
            std::vector<uint32_t> onset_qubits_this_op;
            onset_qubits_this_op.reserve(op_group.onsets.size() + op_group.onset_pairs.size());

            for (const auto& onset : op_group.onsets) {
                if (rng_.next_u64() <= onset.prob_threshold) {
                    current_erasure_state[onset.qubit_index] = current_erasure_state[onset.qubit_index] == 0 
                        ? 1 : current_erasure_state[onset.qubit_index] + 1; // if not already erased, mark as erased
                    group.onsets.push_back({onset.qubit_index});
                    onset_qubits_this_op.push_back(onset.qubit_index);
                }
            }

            for (const auto& onset_pair : op_group.onset_pairs) {
                if (rng_.next_u64() <= onset_pair.prob_threshold) {
                    // If either qubit is already erased, skip operation
                    if (current_erasure_state[onset_pair.qubit_index1] != 0 ||
                        current_erasure_state[onset_pair.qubit_index2] != 0) {
                        continue;
                    }
                    uint32_t unerased;
                    if (rng_.next_u64() <= (1ULL << 63)) { // coin-flip for which qubit gets erased
                        current_erasure_state[onset_pair.qubit_index1] = 1;
                        group.onsets.push_back({onset_pair.qubit_index1});
                        onset_qubits_this_op.push_back(onset_pair.qubit_index1);
                        unerased = onset_pair.qubit_index2;
                    } else {
                        current_erasure_state[onset_pair.qubit_index2] = 1;
                        group.onsets.push_back({onset_pair.qubit_index2});
                        onset_qubits_this_op.push_back(onset_pair.qubit_index2);
                        unerased = onset_pair.qubit_index1;
                    }

                    // Sample spread on unerased qubit
                    PauliOperation sampled_op = from_internal_pauli_operation(
                        internal::sample_thresholded_pauli_channel(program_.thresholded_onset_channel(), &rng_));
                    if (sampled_op != PauliOperation::I) {
                        group.spreads.push_back({unerased, sampled_op});
                    }
                }
            }

            for (const auto& spread : op_group.onset_spreads) {
                if (current_erasure_state[spread.aff_qubit_index] != 0 ||
                    std::find(onset_qubits_this_op.begin(), onset_qubits_this_op.end(),
                              spread.source_qubit_index) == onset_qubits_this_op.end()) {
                    continue;
                }
                PauliOperation sampled_op = from_internal_pauli_operation(
                    internal::sample_thresholded_pauli_channel(spread.spread_channel, &rng_));
                if (sampled_op != PauliOperation::I) {
                    group.spreads.push_back({spread.aff_qubit_index, sampled_op});
                }
            }

            for (const auto& spread : op_group.persistent_spreads) {
                if (current_erasure_state[spread.aff_qubit_index] != 0 || current_erasure_state[spread.source_qubit_index] == 0) { // only sample spread if affected qubit is erased and source qubit is erased
                    continue;
                }
                PauliOperation sampled_op = from_internal_pauli_operation(
                    internal::sample_thresholded_pauli_channel(spread.spread_channel, &rng_));
                if (sampled_op != PauliOperation::I) {
                    group.spreads.push_back({spread.aff_qubit_index, sampled_op});
                }
            }

            // TODO: Multiple checks of the same bool. Presumably slows down sampling. Probably not a bottleneck.
            for (const auto& check : op_group.checks) {
                if (current_erasure_state[check.qubit_index] == 0) {
                    bool false_positive = rng_.next_u64() <= check.false_positive_threshold;
                    group.checks.push_back({check.qubit_index, false_positive ? CheckOutcome::FalsePositive : CheckOutcome::TrueNegative});
                    if (false_positive) {
                        last_check_result[check.qubit_index] = 1;
                    } else {
                        last_check_result[check.qubit_index] = 0;
                    }
                } else {    
                    const bool is_final_check_for_qubit =
                        program_.qubit_last_check_operation_index[check.qubit_index] ==
                        static_cast<int32_t>(op_index);
                    const bool force_true_positive =
                        is_final_check_for_qubit ||
                        current_erasure_state[check.qubit_index] >= max_persistence;
                    if (force_true_positive) {
                        group.checks.push_back({check.qubit_index, CheckOutcome::TruePositive});
                        last_check_result[check.qubit_index] = 1;
                    } else {
                        bool false_negative = rng_.next_u64() <= check.false_negative_threshold;
                        group.checks.push_back({check.qubit_index, false_negative ? CheckOutcome::FalseNegative : CheckOutcome::TruePositive});
                        if (false_negative) {
                            current_erasure_state[check.qubit_index]++;
                            last_check_result[check.qubit_index] = 0;
                        } else {
                            last_check_result[check.qubit_index] = 1;
                        }
                    }
                }
            }

            for (const auto& reset : op_group.resets) {
                // Reset is driven only by the most recent check outcome.
                if (last_check_result[reset.qubit_index] == 0) {
                    continue;
                }
                const bool reset_failed = rng_.next_u64() <= reset.reset_failure_threshold;
                if (reset_failed) {
                    last_check_result[reset.qubit_index] = 0; // clear check outcome after attempted reset
                    continue;
                }
                PauliOperation sampled_op = from_internal_pauli_operation(
                    internal::sample_thresholded_pauli_channel(reset.reset_channel, &rng_));
                group.resets.push_back({reset.qubit_index, sampled_op});
                current_erasure_state[reset.qubit_index] = 0; // successful reset clears erasure
                last_check_result[reset.qubit_index] = 0; // clear most recent check outcome after reset
            }
        }
        batch.shots.push_back(std::move(shot));
    }
    return batch;
}

} // namespace qerasure::simulator
