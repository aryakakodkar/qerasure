#include <iostream>
#include <unordered_map>

#include "compile.h"
#include "instruction.h"

namespace qerasure::circuit {

CompiledErasureProgram::CompiledErasureProgram(const ErasureCircuit& circuit, const ErasureModel& model) {
    operation_groups.resize(circuit.instructions().size());

    std::unordered_map<uint32_t, uint32_t> checks_survived; // maps erased qubit index to number of checks through which erasure has gone undetected

    uint32_t max_persistence = model.max_persistence;

    uint32_t op_index = 0;
    for (const auto& instr : circuit.instructions()) {
        OperationGroup& group = operation_groups[op_index]; // group of operations for this timestep
        if (is_stim_op(instr.op)) {
            group.stim_instruction = instr;
            if (is_entangling_op(instr.op)) {
                for (size_t i = 0; i < instr.targets.size(); i += 2) {
                    uint32_t control = instr.targets[i];
                    uint32_t target = instr.targets[i + 1];
                    if (checks_survived.find(control) != checks_survived.end()) {
                        group.spreads.push_back({target, model.spread.control_spread});
                    }
                    if (checks_survived.find(target) != checks_survived.end()) {
                        group.spreads.push_back({control, model.spread.target_spread});
                    }
                }
            }
        } else if (is_single_onset_op(instr.op)) {
            for (const auto& target : instr.targets) {
                group.onsets.push_back({target, instr.arg});
                checks_survived[target] = 0;
            }
        } else if (is_multi_onset_op(instr.op)) {
            if (instr.op == OpCode::ERASE2) {
                for (size_t i = 0; i < instr.targets.size(); i += 2) {
                    uint32_t target1 = instr.targets[i]; // to be erased
                    uint32_t target2 = instr.targets[i + 1]; // affected by onset spread
                    group.onsets.push_back({target1, instr.arg});
                    group.spreads.push_back({target2, model.onset});
                    checks_survived[target1] = 0;
                }
            }
        } else if (is_erasure_check_op(instr.op)) {
            for (const auto& target : instr.targets) {
                if (checks_survived.find(target) == checks_survived.end()) {
                    continue;
                }
                checks_survived[target]++;
                group.checks.push_back({target, model.check_false_negative_prob, model.check_false_positive_prob}); // false negative and false positive probs are set later
            }
        // Erasure and reset are not mutually exclusive (e.g. ECR), so both need to be processed if applicable
        } if (is_erasure_reset_op(instr.op)) {
            for (const auto& target : instr.targets) {
                // check if qubit can be erased and was checked since last erase op.
                if (checks_survived.find(target) == checks_survived.end()
                    || checks_survived[target] == 0) {
                    continue;
                } else if (checks_survived[target] >= max_persistence) {
                    group.resets.push_back({target, 0.0, model.reset}); // if max persistence exceeded, reset is guaranteed to succeed
                    checks_survived.erase(target);
                    continue;
                }
                group.resets.push_back({target, instr.arg, model.reset});
            }
        }
        op_index++;
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
            std::cout << "  Onset - Qubit: " << onset.qubit_index << ", Probability: " << onset.probability << "\n";
        }
        for (size_t j = 0; j < operation_groups[i].checks.size(); ++j) {
            const auto& check = operation_groups[i].checks[j];
            std::cout << "  Check - Qubit: " << check.qubit_index << ", False Negative: " << check.false_negative_prob << ", False Positive: " << check.false_positive_prob << "\n";
        }
        for (size_t j = 0; j < operation_groups[i].resets.size(); ++j) {
            const auto& reset = operation_groups[i].resets[j];
            std::cout << "  Reset - Qubit: " << reset.qubit_index << ", Reset Failure Prob: " << reset.reset_failure_prob 
                      << ", Reset Channel: (X: " << reset.reset_channel.p_x << ", Y: " << reset.reset_channel.p_y << ", Z: " << reset.reset_channel.p_z << ")\n";
        }
        for (size_t j = 0; j < operation_groups[i].spreads.size(); ++j) {
            const auto& spread = operation_groups[i].spreads[j];
            std::cout << "  Spread - Affected Qubit: " << spread.aff_qubit_index << ", Spread Channel: (X: " << spread.spread_channel.p_x << ", Y: " << spread.spread_channel.p_y << ", Z: " << spread.spread_channel.p_z << ")\n";
        }  
    }
}

}  // namespace qerasure::circuit
