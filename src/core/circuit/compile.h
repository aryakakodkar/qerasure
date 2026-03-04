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
    double probability;
};

struct ErasureOnsetPair {
    uint32_t qubit_index1;
    uint32_t qubit_index2;
    uint64_t prob_threshold;
    double probability;
};

struct ErasureSpread {
    uint32_t source_qubit_index; // index of erased qubit causing spread
    uint32_t aff_qubit_index; // index of qubit that can be affected by spread
    ThresholdedPauliChannel spread_channel;
    PauliChannel spread_probability_channel;
};

struct ErasureCheck {
    uint32_t qubit_index;
    uint64_t false_negative_threshold; 
    uint64_t false_positive_threshold;
    double false_negative_probability;
    double false_positive_probability;
};

struct ErasureReset {
    uint32_t qubit_index;
    uint64_t reset_failure_threshold; 
    ThresholdedPauliChannel reset_channel;
    double reset_failure_probability;
    PauliChannel reset_probability_channel;
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

struct CheckLookbackLink {
    uint32_t qubit_index;
    uint32_t check_op_index;
    int32_t lookback_check_event_index; // -1 if not available
    int32_t reset_op_after_lookback; // -1 if not available
};

struct CompiledErasureProgram  {
    CompiledErasureProgram(const ErasureCircuit& circuit, const ErasureModel& model);

    std::vector<OperationGroup> operation_groups;
    // Per-check precomputed jump links for fast decoder lookback.
    std::vector<CheckLookbackLink> check_lookback_links;
    // Thin per-qubit index layers into `operation_groups` (no duplicated payloads).
    // For qubit q, entries are sorted and unique operation indices.
    std::vector<std::vector<uint32_t>> qubit_operation_indices;
    std::vector<std::vector<uint32_t>> qubit_check_operation_indices;
    std::vector<std::vector<uint32_t>> qubit_reset_operation_indices;

    inline uint32_t max_qubit_index() const { return max_qubit_index_; }
    inline uint32_t num_checks() const { return num_checks_; }
    inline const std::vector<uint32_t>& erasable_qubits() const { return erasable_qubits_; }
    inline uint32_t max_persistence() const { return max_persistence_; }
    inline const ThresholdedPauliChannel& thresholded_onset_channel() const { return thresholded_onset_; }
    inline const ErasureModel& model() const { return model_; }

    void print_summary() const;

    private:
        uint32_t max_qubit_index_ = 0;
        uint32_t num_checks_ = 0;
        uint32_t max_persistence_ = 0;
        std::vector<uint32_t> erasable_qubits_; // check if needed

        const ErasureModel model_;

        // Stored as a reference; the caller must ensure the model outlives this object.
        const ThresholdedPauliChannel thresholded_onset_;
};

}  // namespace qerasure::circuit
