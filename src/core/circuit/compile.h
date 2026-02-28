# pragma once

#include <cstdint>

#include "instruction.h"
#include "circuit.h"
#include "model/pauli_channel.h"
#include "erasure_model.h"

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

struct CompiledErasureProgram  {
    CompiledErasureProgram(const ErasureCircuit& circuit, const ErasureModel& model);

    std::vector<Instruction> stim_instructions;

    std::vector<ErasureOnset> onsets;
    std::vector<uint32_t> onset_offsets = {0};

    std::vector<ErasureSpread> spreads;
    std::vector<uint32_t> spread_offsets = {0};

    std::vector<ErasureCheck> checks;
    std::vector<uint32_t> check_offsets = {0};

    std::vector<ErasureReset> resets;
    std::vector<uint32_t> reset_offsets = {0};

    void print_summary() const;
};

