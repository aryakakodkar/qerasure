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
    uint32_t qubit_index; // index of qubit that can be affected by spread
    PauliChannel spread_channel;
};

enum class HookType {
    EC,
    ECR,
    COND_ER
};

struct ErasureHook {
    HookType type;
    uint32_t qubit_index;
    double probability = 0.0;
};

struct CompiledErasureProgram  {
    CompiledErasureProgram(ErasureCircuit& circuit);

    std::vector<Instruction> instructions;

    std::vector<ErasureOnset> onsets;
    std::vector<uint32_t> onset_offsets;

    std::vector<ErasureHook> hooks;
    std::vector<uint32_t> hook_offsets;
};

