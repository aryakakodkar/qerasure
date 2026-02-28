#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "instruction.h"
#include "circuit.h"
#include "core/model/pauli_channel.h"
#include "erasure_model.h"

namespace qerasure::circuit {

struct ErasureOnset {
    uint32_t qubit_index;
    double probability;
};

struct ErasureSpread {
    uint32_t aff_qubit_index; // index of qubit that can be affected by spread
    PauliChannel spread_channel;
};

struct ErasureCheck {
    uint32_t qubit_index;
    double false_negative_prob;
    double false_positive_prob;
};

struct ErasureReset {
    uint32_t qubit_index;
    double reset_failure_prob;
    PauliChannel reset_channel;
};

// Group operations by time for cheap sampling at runtime
struct OperationGroup {
    std::optional<Instruction> stim_instruction;
    std::vector<ErasureOnset> onsets;
    std::vector<ErasureSpread> spreads;
    std::vector<ErasureCheck> checks;
    std::vector<ErasureReset> resets;
};

struct CompiledErasureProgram  {
    CompiledErasureProgram(const ErasureCircuit& circuit, const ErasureModel& model);

    std::vector<OperationGroup> operation_groups;

    void print_summary() const;
};

}  // namespace qerasure::circuit
