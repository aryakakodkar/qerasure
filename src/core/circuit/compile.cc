#include <iostream>
#include <unordered_map>

#include "compile.h"
#include "instruction.h"

CompiledErasureProgram::CompiledErasureProgram(const ErasureCircuit& circuit, const ErasureModel& model) {
    std::unordered_map<uint32_t, uint32_t> possibly_erased_qubits; // maps qubit index to number of reset ops survived
    uint32_t max_persistence = model.max_persistence;

    for (const auto& instr : circuit.instructions()) {
        if (is_stim_op(instr.op)) {
            stim_instructions.push_back(instr);
            if (is_entangling_op(instr.op)) {
                for (size_t i = 0; i < instr.targets.size(); i += 2) {
                    uint32_t control = instr.targets[i];
                    uint32_t target = instr.targets[i + 1];
                    if (possibly_erased_qubits.find(control) != possibly_erased_qubits.end()) {
                        spreads.push_back({target, model.spread.control_spread});
                    }
                    if (possibly_erased_qubits.find(target) != possibly_erased_qubits.end()) {
                        spreads.push_back({control, model.spread.target_spread});
                    }
                }
            }
        } else if (is_single_onset_op(instr.op)) {
            for (const auto& target : instr.targets) {
                onsets.push_back({target, instr.arg});
                possibly_erased_qubits[target] = 0;
            }
        } else if (is_multi_onset_op(instr.op)) {
            if (instr.op == OpCode::ERASE2) {
                for (size_t i = 0; i < instr.targets.size(); i += 2) {
                    uint32_t target1 = instr.targets[i];
                    uint32_t target2 = instr.targets[i + 1];
                    onsets.push_back({target1, instr.arg});
                    spreads.push_back({target2, model.onset});
                }
            }
        } else if (is_erasure_check_op(instr.op)) {
            for (const auto& target : instr.targets) {
                if (possibly_erased_qubits.find(target) == possibly_erased_qubits.end()) {
                    continue;
                }
                possibly_erased_qubits[target]++;
                checks.push_back({target, model.check_false_negative_prob, model.check_false_positive_prob}); // false negative and false positive probs are set later
            }
        } if (is_erasure_reset_op(instr.op)) {
            for (const auto& target : instr.targets) {
                // check if qubit can be erased and was checked since last erase op.
                if (possibly_erased_qubits.find(target) == possibly_erased_qubits.end()
                    || possibly_erased_qubits[target] == 0) {
                    continue;
                } else if (possibly_erased_qubits[target] >= max_persistence) {
                    resets.push_back({target, 0.0, model.reset}); // if max persistence exceeded, reset is guaranteed to succeed
                }
                resets.push_back({target, instr.arg, model.reset});
                possibly_erased_qubits[target]++;
                if (possibly_erased_qubits[target] == max_persistence) {
                    possibly_erased_qubits.erase(target);
                }
            }
        }

        onset_offsets.push_back(static_cast<uint32_t>(onsets.size()));
        check_offsets.push_back(static_cast<uint32_t>(checks.size()));
        reset_offsets.push_back(static_cast<uint32_t>(resets.size()));
        spread_offsets.push_back(static_cast<uint32_t>(spreads.size()));
    }
}

// Debug helper
void CompiledErasureProgram::print_summary() const {
    std::cout << "Compiled Erasure Program Summary:\n";
    std::cout << "Total Stim Instructions: " << stim_instructions.size() << "\n";
    
    for (size_t i = 0; i < onset_offsets.size() - 1; ++i) {
        std::cout << "====== INSTRUCTION " << i << " ======\n";
        for (size_t j = onset_offsets[i]; j < onset_offsets[i + 1]; ++j) {
            const auto& onset = onsets[j];
            std::cout << "  Onset - Qubit: " << onset.qubit_index << ", Probability: " << onset.probability << "\n";
        }
        for (size_t j = check_offsets[i]; j < check_offsets[i + 1]; ++j) {
            const auto& check = checks[j];
            std::cout << "  Check - Qubit: " << check.qubit_index << ", False Negative: " << check.false_negative_prob << ", False Positive: " << check.false_positive_prob << "\n";
        }
        for (size_t j = reset_offsets[i]; j < reset_offsets[i + 1]; ++j) {
            const auto& reset = resets[j];
            std::cout << "  Reset - Qubit: " << reset.qubit_index << ", Reset Failure Prob: " << reset.reset_failure_prob 
                      << ", Reset Channel: (X: " << reset.reset_channel.p_x << ", Y: " << reset.reset_channel.p_y << ", Z: " << reset.reset_channel.p_z << ")\n";
        }
        for (size_t j = spread_offsets[i]; j < spread_offsets[i + 1]; ++j) {
            const auto& spread = spreads[j];
            std::cout << "  Spread - Affected Qubit: " << spread.aff_qubit_index << ", Spread Channel: (X: " << spread.spread_channel.p_x << ", Y: " << spread.spread_channel.p_y << ", Z: " << spread.spread_channel.p_z << ")\n";
        }  
    }
}
