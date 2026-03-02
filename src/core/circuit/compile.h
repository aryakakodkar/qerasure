#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "instruction.h"
#include "circuit.h"
#include "core/model/pauli_channel.h"
#include "erasure_model.h"

namespace qerasure::circuit {

// Using precomputed threshold for probability sampling
struct ErasureOnset {
    uint32_t qubit_index;
    uint64_t prob_threshold;
};

struct ErasureOnsetPair {
    uint32_t qubit_index1;
    uint32_t qubit_index2;
    uint64_t prob_threshold;
};

struct ErasureSpread {
    uint32_t aff_qubit_index; // index of qubit that can be affected by spread
    ThresholdedPauliChannel spread_channel;
};

struct ErasureCheck {
    uint32_t qubit_index;
    uint64_t false_negative_threshold; 
    uint64_t false_positive_threshold;
};

struct ErasureReset {
    uint32_t qubit_index;
    uint64_t reset_failure_threshold; 
    ThresholdedPauliChannel reset_channel;
};

// Group operations by time for cheap sampling at runtime
struct OperationGroup {
    std::optional<Instruction> stim_instruction;
    std::vector<ErasureOnset> onsets;
    std::vector<ErasureOnsetPair> onset_pairs;
    std::vector<ErasureSpread> spreads;
    std::vector<ErasureCheck> checks;
    std::vector<ErasureReset> resets;
    uint32_t op_num = 0; // number of operations in this group, used for vector resizing when sampling
};

struct CompiledErasureProgram  {
    CompiledErasureProgram(const ErasureCircuit& circuit, const ErasureModel& model);

    std::vector<OperationGroup> operation_groups;

    inline uint32_t max_qubit_index() const { return max_qubit_index_; }
    inline const std::vector<uint32_t>& erasable_qubits() const { return erasable_qubits_; }
    inline uint32_t max_persistence() const { return max_persistence_; }
    inline const ThresholdedPauliChannel& thresholded_onset_channel() const { return thresholded_onset_; }

    void print_summary() const;

    private:
        uint32_t max_qubit_index_ = 0;
        uint32_t max_persistence_ = 0;
        std::vector<uint32_t> erasable_qubits_; // check if needed

        // Stored as a reference; the caller must ensure the model outlives this object.
        const ThresholdedPauliChannel thresholded_onset_;
};

}  // namespace qerasure::circuit
