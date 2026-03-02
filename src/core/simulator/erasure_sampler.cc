#include <cstdint>
#include <optional>
#include <algorithm>
#include <unordered_map>

#include "core/model/pauli_channel.h"
#include "core/simulator/fast_rng.h"
#include "core/simulator/erasure_sampler.h"
#include "core/circuit/compile.h"

namespace qerasure::simulator {

namespace {

PauliOperation sample_thresholded_pauli_channel(const ThresholdedPauliChannel& channel,
                                 qerasure::simulator::FastRng* rng) {
    const std::uint64_t draw = rng->next_u64();
    const std::uint64_t x_cut = channel.p_x_threshold;
    const std::uint64_t y_cut = x_cut + channel.p_y_threshold;
    const std::uint64_t z_cut = y_cut + channel.p_z_threshold;
    if (draw < x_cut) {
        return PauliOperation::X;
    }
    if (draw < y_cut) {
        return PauliOperation::Y;
    }
    if (draw < z_cut) {
        return PauliOperation::Z;
    }
    return PauliOperation::I;
}

}  // namespace

SampledBatch ErasureSampler::sample(const SamplerParams& params) {
    SampledBatch batch;
    batch.shots.reserve(params.shots);
    size_t num_ops = program_.operation_groups.size();

    // Shift to uint64_t representation of max persistence for conditional check functionality
    // If a check flags a positive, it will change the current erasure state to max_persistence + 1
    // so that the next reset is certainly applied
    uint64_t max_persistence = static_cast<uint64_t>(program_.max_persistence());

    FastRng rng_(params.seed);

    // Current erasure state holds the check survival count as well
    // 0: unerased, 1: newly erased, n (1 < n): erased and survived n-1 checks
    std::vector<uint64_t> current_erasure_state(program_.max_qubit_index(), 0);
    
    std::vector<uint8_t> last_check_result(program_.max_qubit_index(), 0); // 0: no check or negative, 1: positive

    for (std::uint64_t shot_idx = 0; shot_idx < params.shots; ++shot_idx) {
        SampledShot shot;
        std::fill(current_erasure_state.begin(), current_erasure_state.end(), 0); // reset erasure state for new shot
        shot.operation_groups.resize(num_ops);

        for (size_t op_index = 0; op_index < num_ops; ++op_index) {
            const circuit::OperationGroup& op_group = program_.operation_groups[op_index];
            SampledOperationGroup& group = shot.operation_groups[op_index];

            group.onsets.reserve(op_group.onsets.size());
            group.spreads.reserve(op_group.spreads.size());
            group.checks.reserve(op_group.checks.size());
            group.resets.reserve(op_group.resets.size());

            for (const auto& onset : op_group.onsets) {
                if (rng_.next_u64() <= onset.prob_threshold) {
                    current_erasure_state[onset.qubit_index] = current_erasure_state[onset.qubit_index] == 0 ? 1 : current_erasure_state[onset.qubit_index] + 1; // if not already erased, mark as erased
                    group.onsets.push_back({onset.qubit_index});
                }
            }

            for (const auto& spread : op_group.spreads) {
                if (current_erasure_state[spread.aff_qubit_index] != 0) { // only sampled spread if affected qubit is not erased
                    continue;
                }
                PauliOperation sampled_op = sample_thresholded_pauli_channel(spread.spread_channel, &rng_);
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

            for (const auto& reset : op_group.resets) {
                // TODO: Need to check if reset is conditional on check outcome. For now, all resets are conditional on successful checks.
                // Checks if current state is greater than max persistence + 1 because this means that the qubit has survived more than
                // max_persistence checks, so it must be reset
                if (current_erasure_state[reset.qubit_index] > max_persistence + 1 || (last_check_result[reset.qubit_index] == 1 && rng_.next_u64() <= reset.reset_failure_threshold)) {
                    PauliOperation sampled_op = sample_thresholded_pauli_channel(reset.reset_channel, &rng_);
                    group.resets.push_back({reset.qubit_index, sampled_op});
                    current_erasure_state[reset.qubit_index] = 0; // reset successful, mark qubit as unerased
                    last_check_result[reset.qubit_index] = 0; // reset last check result
                } 
            }
        }
        batch.shots.push_back(std::move(shot));
    }
    return batch;
}

} // namespace qerasure::simulator
