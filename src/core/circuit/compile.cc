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
        } else if (is_hook_op(instr.op)) {
            bool is_reset = is_erasure_reset_op(instr.op);
            for (const auto& target : instr.targets) {
                if (is_reset) {
                    if (possibly_erased_qubits.find(target) == possibly_erased_qubits.end()) {
                        continue; // if qubit couldn't have been erased, skip
                    } 
                    if (possibly_erased_qubits[target] == max_persistence - 1) {
                        possibly_erased_qubits.erase(target); // erasure cannot persist further
                        
                    } else {
                        possibly_erased_qubits[target]++; // qubit survives another reset
                    }
                }
                hooks.push_back({instr.op, target, instr.arg});
            }
        }

        onset_offsets.push_back(static_cast<uint32_t>(onsets.size()));
        hook_offsets.push_back(static_cast<uint32_t>(hooks.size()));
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
        for (size_t j = hook_offsets[i]; j < hook_offsets[i + 1]; ++j) {
            const auto& hook = hooks[j];
            std::cout << "  Hook - Type: " << static_cast<int>(hook.type) << ", Qubit: " << hook.qubit_index << ", Probability: " << hook.probability << "\n";
        }
    }
}
