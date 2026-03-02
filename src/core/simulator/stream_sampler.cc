#include <algorithm>
#include <atomic>
#include <mutex>
#include <thread>

#include "core/circuit/compile.h"
#include "core/simulator/stream_sampler.h"
#include "core/simulator/fast_rng.h"
#include "core/circuit/instruction.h"
#include "core/simulator/erasure_sampler.h"
#include "core/simulator/sim_internal_utils.h"

namespace qerasure::simulator {

namespace {

void sample_one_shot(const circuit::CompiledErasureProgram& program,
                     uint64_t max_persistence,
                     FastRng* rng,
                     std::vector<uint64_t>* current_erasure_state,
                     std::vector<uint8_t>* last_check_result,
                     std::vector<uint8_t>* check_results,
                     const std::function<void(const stim::Circuit&, const std::vector<uint8_t>&)>& callback,
                     std::mutex* callback_mutex) {
    stim::Circuit circuit;

    // reset erasure state for new shot
    std::fill(current_erasure_state->begin(), current_erasure_state->end(), 0);
    std::fill(last_check_result->begin(), last_check_result->end(), 0);
    std::fill(check_results->begin(), check_results->end(), 0);

    for (const circuit::OperationGroup& op_group : program.operation_groups) {
        if (op_group.stim_instruction.has_value()) {
            if (circuit::is_measurement_op(op_group.stim_instruction->op)) {
                // TODO: Add probabilistic error
                internal::append_mapped_stim_instruction(*op_group.stim_instruction, &circuit);
            } else if (circuit::is_erasure_skippable_op(op_group.stim_instruction->op)) {
                std::vector<uint32_t> non_erased_targets;
                non_erased_targets.reserve(op_group.stim_instruction->targets.size());

                for (uint32_t target : op_group.stim_instruction->targets) {
                    if ((*current_erasure_state)[target] == 0) {
                        non_erased_targets.push_back(target);
                    }
                }
                if (!non_erased_targets.empty()) {
                    internal::append_mapped_stim_instruction(op_group.stim_instruction->op,
                                                             non_erased_targets,
                                                             op_group.stim_instruction->arg,
                                                             &circuit);
                }
            } else {
                internal::append_mapped_stim_instruction(*op_group.stim_instruction, &circuit);
            }
        }

        // Should I perform an operation upon onset? I don't think so.
        for (const auto& onset : op_group.onsets) {
            if (rng->next_u64() <= onset.prob_threshold) {
                (*current_erasure_state)[onset.qubit_index] =
                    (*current_erasure_state)[onset.qubit_index] == 0
                        ? 1
                        : (*current_erasure_state)[onset.qubit_index] + 1;  // if not already erased, mark as erased
            }
        }

        for (const auto& onset_pair : op_group.onset_pairs) {
            if (rng->next_u64() <= onset_pair.prob_threshold) {
                // If either qubit is already erased, skip operation
                if ((*current_erasure_state)[onset_pair.qubit_index1] != 0 ||
                    (*current_erasure_state)[onset_pair.qubit_index2] != 0) {
                    continue;
                }
                uint32_t unerased;
                if (rng->next_u64() <= (1ULL << 63)) {  // coin-flip for which qubit gets erased
                    (*current_erasure_state)[onset_pair.qubit_index1] = 1;
                    unerased = onset_pair.qubit_index2;
                } else {
                    (*current_erasure_state)[onset_pair.qubit_index2] = 1;
                    unerased = onset_pair.qubit_index1;
                }

                // Sample spread on unerased qubit
                internal::PauliOperation sampled_op =
                    internal::sample_thresholded_pauli_channel(program.thresholded_onset_channel(), rng);
                if (sampled_op != internal::PauliOperation::I) {
                    internal::append_mapped_pauli_operation(unerased, sampled_op, &circuit);
                }
            }
        }

        for (const auto& spread : op_group.spreads) {
            if ((*current_erasure_state)[spread.aff_qubit_index] != 0 ||
                (*current_erasure_state)[spread.source_qubit_index] == 0) {  // only sample spread if affected qubit is erased and source qubit is erased
                continue;
            }
            internal::PauliOperation sampled_op =
                internal::sample_thresholded_pauli_channel(spread.spread_channel, rng);
            if (sampled_op != internal::PauliOperation::I) {
                internal::append_mapped_pauli_operation(spread.aff_qubit_index, sampled_op, &circuit);
            }
        }

        // TODO: Multiple checks of the same bool. Presumably slows down sampling. Probably not a bottleneck.
        uint32_t check_idx = 0;
        for (const auto& check : op_group.checks) {
            if ((*current_erasure_state)[check.qubit_index] == 0) {
                bool false_positive = rng->next_u64() <= check.false_positive_threshold;
                (*check_results)[check_idx++] = false_positive;
                if (false_positive) {
                    (*last_check_result)[check.qubit_index] = 1;
                } else {
                    (*last_check_result)[check.qubit_index] = 0;
                }
            } else {
                bool false_negative = rng->next_u64() <= check.false_negative_threshold;
                (*check_results)[check_idx++] = false_negative;
                if (false_negative) {
                    (*current_erasure_state)[check.qubit_index]++;
                    (*last_check_result)[check.qubit_index] = 0;
                } else {
                    (*last_check_result)[check.qubit_index] = 1;
                }
            }
        }

        for (const auto& reset : op_group.resets) {
            // TODO: Need to check if reset is conditional on check outcome. For now, all resets are conditional on successful checks.
            // Checks if current state is greater than max persistence + 1 because this means that the qubit has survived more than
            // max_persistence checks, so it must be reset
            if ((*current_erasure_state)[reset.qubit_index] > max_persistence + 1 ||
                ((*last_check_result)[reset.qubit_index] == 1 &&
                 rng->next_u64() <= reset.reset_failure_threshold)) {
                internal::PauliOperation sampled_op =
                    internal::sample_thresholded_pauli_channel(reset.reset_channel, rng);
                internal::append_mapped_pauli_operation(reset.qubit_index, sampled_op, &circuit);
                (*current_erasure_state)[reset.qubit_index] = 0;  // reset successful, mark qubit as unerased
                (*last_check_result)[reset.qubit_index] = 0;      // reset last check result
            }
        }
    }

    if (callback_mutex == nullptr) {
        callback(circuit, *check_results);
    } else {
        std::lock_guard<std::mutex> lock(*callback_mutex);
        callback(circuit, *check_results);
    }
}

}  // namespace

void StreamSampler::sample(uint32_t num_shots,
                       uint32_t seed,
                       std::function<void(const stim::Circuit&, const std::vector<uint8_t>&)> callback,
                       uint32_t num_threads) {
    if (num_shots == 0) {
        return;
    }
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) {
            num_threads = 1;
        }
    }
    num_threads = std::min(num_threads, num_shots);

    const uint64_t max_persistence = static_cast<uint64_t>(program_.max_persistence());

    // Preserve old deterministic behavior in single-thread mode.
    if (num_threads == 1) {
        FastRng rng_(seed);
        std::vector<uint64_t> current_erasure_state(program_.max_qubit_index() + 1, 0);
        std::vector<uint8_t> last_check_result(program_.max_qubit_index() + 1, 0);
        std::vector<uint8_t> check_results(program_.num_checks(), 0);
        for (uint32_t shot = 0; shot < num_shots; ++shot) {
            sample_one_shot(program_, max_persistence, &rng_, &current_erasure_state,
                            &last_check_result, &check_results, callback, nullptr);
        }
        return;
    }

    std::atomic<uint32_t> next_shot{0};
    std::mutex callback_mutex;
    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    for (uint32_t tid = 0; tid < num_threads; ++tid) {
        workers.emplace_back([&, tid]() {
            std::vector<uint64_t> current_erasure_state(program_.max_qubit_index() + 1, 0);
            std::vector<uint8_t> last_check_result(program_.max_qubit_index() + 1, 0);
            std::vector<uint8_t> check_results(program_.num_checks(), 0);
            while (true) {
                const uint32_t shot = next_shot.fetch_add(1, std::memory_order_relaxed);
                if (shot >= num_shots) {
                    break;
                }
                FastRng shot_rng(seed, shot);
                sample_one_shot(program_, max_persistence, &shot_rng, &current_erasure_state,
                                &last_check_result, &check_results, callback, &callback_mutex);
            }
        });
    }

    for (std::thread& worker : workers) {
        worker.join();
    }
}

} // namespace qerasure::simulator
