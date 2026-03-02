#pragma once

#include <cstdint>
#include <vector>

#include "core/circuit/compile.h"

namespace qerasure::simulator {

enum class PauliOperation {
    X,
    Y,
    Z,
    I
};

struct SampledOnset {
    uint32_t qubit_index;
};

struct SampledSpread {
    uint32_t qubit_index;
    PauliOperation operation;
};

enum class CheckOutcome {
    TruePositive,
    FalseNegative,
    FalsePositive, 
    TrueNegative
};

struct SampledCheck {
    uint32_t qubit_index;
    CheckOutcome outcome;
};

struct SampledReset {
    uint32_t qubit_index;
    PauliOperation operation;
};

struct SampledOperationGroup {
    std::vector<SampledOnset> onsets;
    std::vector<SampledSpread> spreads;
    std::vector<SampledCheck> checks;
    std::vector<SampledReset> resets;
};

// Simulation structs
struct SampledShot {
    std::vector<SampledOperationGroup> operation_groups;
};

struct SampledBatch {
    std::vector<SampledShot> shots;
};

struct SamplerParams {
    uint64_t shots;
    uint64_t seed;
};

class ErasureSampler {
    public:
        explicit ErasureSampler(const circuit::CompiledErasureProgram& program) : program_(program) {};
        SampledBatch sample(const SamplerParams& params);

    private:
        const circuit::CompiledErasureProgram& program_;
};

} // namespace qerasure::simulator
