#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <thread>

#include "core/circuit/compile.h"
#include "core/simulator/stream_sampler.h"
#include "core/simulator/fast_rng.h"
#include "core/circuit/instruction.h"
#include "core/simulator/erasure_sampler.h"
#include "core/simulator/sim_internal_utils.h"
#include "stim/simulators/frame_simulator_util.h"

namespace qerasure::simulator {

namespace {

stim::Circuit build_sampled_logical_circuit(const circuit::CompiledErasureProgram& program,
                                            uint64_t max_persistence,
                                            FastRng* rng,
                                            std::vector<uint64_t>* current_erasure_state,
                                            std::vector<uint8_t>* last_check_result,
                                            std::vector<uint8_t>* check_results) {
    stim::Circuit circuit;

    // reset erasure state for new shot
    std::fill(current_erasure_state->begin(), current_erasure_state->end(), 0);
    std::fill(last_check_result->begin(), last_check_result->end(), 0);
    std::fill(check_results->begin(), check_results->end(), 0);
    size_t check_idx = 0;

    for (const circuit::OperationGroup& op_group : program.operation_groups) {
        if (op_group.stim_instruction.has_value()) {
            if (circuit::is_measurement_op(op_group.stim_instruction->op)) {
                // if a qubit is erased, its measurement outcome should be random
                for (uint32_t target : op_group.stim_instruction->targets) {
                    if ((*current_erasure_state)[target] >= 1) {
                        internal::append_mapped_stim_instruction(circuit::OpCode::X_ERROR,
                                                                 {target},
                                                                 0.5,
                                                                 &circuit);
                    }
                }

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

        // Track qubits whose erasure onset fired at this operation.
        std::vector<uint32_t> onset_qubits_this_op;
        onset_qubits_this_op.reserve(op_group.onsets.size() + op_group.onset_pairs.size());
        for (const auto& onset : op_group.onsets) {
            if (rng->next_u64() <= onset.prob_threshold) {
                (*current_erasure_state)[onset.qubit_index] =
                    (*current_erasure_state)[onset.qubit_index] == 0
                        ? 1
                        : (*current_erasure_state)[onset.qubit_index] + 1;
                onset_qubits_this_op.push_back(onset.qubit_index);
            }
        }

        for (const auto& onset_pair : op_group.onset_pairs) {
            if (rng->next_u64() <= onset_pair.prob_threshold) {
                // If either qubit is already erased, skip operation.
                if ((*current_erasure_state)[onset_pair.qubit_index1] != 0 ||
                    (*current_erasure_state)[onset_pair.qubit_index2] != 0) {
                    continue;
                }
                uint32_t unerased;
                if (rng->next_u64() <= (1ULL << 63)) {  // coin-flip for which qubit gets erased
                    (*current_erasure_state)[onset_pair.qubit_index1] = 1;
                    onset_qubits_this_op.push_back(onset_pair.qubit_index1);
                    unerased = onset_pair.qubit_index2;
                } else {
                    (*current_erasure_state)[onset_pair.qubit_index2] = 1;
                    onset_qubits_this_op.push_back(onset_pair.qubit_index2);
                    unerased = onset_pair.qubit_index1;
                }

                // ERASE2_ANY onset-spread on the non-erased partner.
                internal::PauliOperation sampled_op =
                    internal::sample_thresholded_pauli_channel(program.thresholded_onset_channel(), rng);
                if (sampled_op != internal::PauliOperation::I) {
                    internal::append_mapped_pauli_operation(unerased, sampled_op, &circuit);
                }
            }
        }

        for (const auto& spread : op_group.onset_spreads) {
            if ((*current_erasure_state)[spread.aff_qubit_index] != 0 ||
                std::find(onset_qubits_this_op.begin(), onset_qubits_this_op.end(),
                          spread.source_qubit_index) == onset_qubits_this_op.end()) {
                continue;
            }
            internal::PauliOperation sampled_op =
                internal::sample_thresholded_pauli_channel(spread.spread_channel, rng);
            if (sampled_op != internal::PauliOperation::I) {
                internal::append_mapped_pauli_operation(spread.aff_qubit_index, sampled_op, &circuit);
            }
        }

        for (const auto& spread : op_group.persistent_spreads) {
            if ((*current_erasure_state)[spread.aff_qubit_index] != 0 ||
                (*current_erasure_state)[spread.source_qubit_index] == 0) {
                continue;
            }
            internal::PauliOperation sampled_op =
                internal::sample_thresholded_pauli_channel(spread.spread_channel, rng);
            if (sampled_op != internal::PauliOperation::I) {
                internal::append_mapped_pauli_operation(spread.aff_qubit_index, sampled_op, &circuit);
            }
        }

        // TODO: Multiple checks of the same bool. Presumably slows down sampling. Probably not a bottleneck.
        for (const auto& check : op_group.checks) {
            if ((*current_erasure_state)[check.qubit_index] == 0) {
                bool false_positive = rng->next_u64() <= check.false_positive_threshold;
                (*check_results)[check_idx++] = false_positive ? 1 : 0;
                if (false_positive) {
                    (*last_check_result)[check.qubit_index] = 1;
                } else {
                    (*last_check_result)[check.qubit_index] = 0;
                }
            } else {
                const bool force_true_positive =
                    (*current_erasure_state)[check.qubit_index] >= max_persistence;
                if (force_true_positive) {
                    (*check_results)[check_idx++] = 1;
                    (*last_check_result)[check.qubit_index] = 1;
                } else {
                    bool false_negative = rng->next_u64() <= check.false_negative_threshold;
                    (*check_results)[check_idx++] = false_negative ? 0 : 1;
                    if (false_negative) {
                        (*current_erasure_state)[check.qubit_index]++;
                        (*last_check_result)[check.qubit_index] = 0;
                    } else {
                        (*last_check_result)[check.qubit_index] = 1;
                    }
                }
            }
        }

        for (const auto& reset : op_group.resets) {
            // Reset is driven only by the most recent check outcome.
            if ((*last_check_result)[reset.qubit_index] == 1) {
                const bool reset_failed = rng->next_u64() <= reset.reset_failure_threshold;
                if (!reset_failed) {
                    internal::PauliOperation sampled_op =
                        internal::sample_thresholded_pauli_channel(reset.reset_channel, rng);
                    internal::append_mapped_pauli_operation(reset.qubit_index, sampled_op, &circuit);
                    (*current_erasure_state)[reset.qubit_index] = 0;  // successful reset clears erasure
                }
                (*last_check_result)[reset.qubit_index] = 0;      // clear most recent check outcome after reset
            }
        }
    }

    if (check_idx != check_results->size()) {
        throw std::logic_error("StreamSampler check vector size mismatch while sampling shot.");
    }

    return circuit;
}

void fill_stim_syndrome_row(const stim::Circuit& circuit,
                            uint64_t stim_seed,
                            std::vector<uint8_t>* detector_samples,
                            size_t detector_row_offset,
                            std::vector<uint8_t>* observable_flips,
                            size_t observable_row_offset,
                            uint32_t expected_num_detectors,
                            uint32_t expected_num_observables) {
    std::mt19937_64 stim_rng(stim_seed);
    auto sampled = stim::sample_batch_detection_events<stim::MAX_BITWORD_WIDTH>(circuit, 1, stim_rng);
    const auto& det_data = sampled.first;
    const auto& obs_data = sampled.second;

    const uint32_t num_detectors = static_cast<uint32_t>(circuit.count_detectors());
    const uint32_t num_observables = static_cast<uint32_t>(circuit.count_observables());
    if (num_detectors != expected_num_detectors || num_observables != expected_num_observables) {
        throw std::logic_error("StreamSampler sampled circuit changed detector/observable count across shots.");
    }

    for (uint32_t d = 0; d < num_detectors; ++d) {
        (*detector_samples)[detector_row_offset + d] = det_data[d][0] ? 1 : 0;
    }
    for (uint32_t o = 0; o < num_observables; ++o) {
        (*observable_flips)[observable_row_offset + o] = obs_data[o][0] ? 1 : 0;
    }
}

}  // namespace

// DEPRECATED: Use sample_syndromes() instead
void StreamSampler::sample_with_callback(
    uint32_t num_shots,
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
            stim::Circuit circuit = build_sampled_logical_circuit(
                program_, max_persistence, &rng_, &current_erasure_state, &last_check_result,
                &check_results);
            callback(circuit, check_results);
        }
        return;
    }

    std::atomic<uint32_t> next_shot{0};
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
                stim::Circuit circuit = build_sampled_logical_circuit(
                    program_, max_persistence, &shot_rng, &current_erasure_state,
                    &last_check_result, &check_results);
                callback(circuit, check_results);
            }
        });
    }

    for (std::thread& worker : workers) {
        worker.join();
    }
}

SyndromeSampleBatch StreamSampler::sample_syndromes(
    uint32_t num_shots,
    uint32_t seed,
    uint32_t num_threads) {
    SyndromeSampleBatch out;
    out.num_shots = num_shots;
    out.num_checks = program_.num_checks();
    if (num_shots == 0) {
        return out;
    }

    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) {
            num_threads = 1;
        }
    }
    num_threads = std::min(num_threads, num_shots);

    const uint64_t max_persistence = static_cast<uint64_t>(program_.max_persistence());

    // Initialize output dimensions by sampling shot 0 first.
    {
        std::vector<uint64_t> current_erasure_state(program_.max_qubit_index() + 1, 0);
        std::vector<uint8_t> last_check_result(program_.max_qubit_index() + 1, 0);
        std::vector<uint8_t> check_results(program_.num_checks(), 0);
        FastRng rng0(seed, 0);
        stim::Circuit circuit0 = build_sampled_logical_circuit(
            program_, max_persistence, &rng0, &current_erasure_state, &last_check_result,
            &check_results);
        out.num_detectors = static_cast<uint32_t>(circuit0.count_detectors());
        out.num_observables = static_cast<uint32_t>(circuit0.count_observables());

        out.detector_samples.resize(
            static_cast<size_t>(out.num_shots) * static_cast<size_t>(out.num_detectors), 0);
        out.observable_flips.resize(
            static_cast<size_t>(out.num_shots) * static_cast<size_t>(out.num_observables), 0);
        out.check_flags.resize(
            static_cast<size_t>(out.num_shots) * static_cast<size_t>(out.num_checks), 0);

        fill_stim_syndrome_row(
            circuit0,
            (static_cast<uint64_t>(seed) << 32) ^ static_cast<uint64_t>(0),
            &out.detector_samples,
            /*detector_row_offset=*/0,
            &out.observable_flips,
            /*observable_row_offset=*/0,
            out.num_detectors,
            out.num_observables);

        if (!check_results.empty()) {
            std::copy(
                check_results.begin(),
                check_results.end(),
                out.check_flags.begin());
        }
    }

    if (num_shots == 1) {
        return out;
    }

    std::atomic<uint32_t> next_shot{1};
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
                stim::Circuit circuit = build_sampled_logical_circuit(
                    program_, max_persistence, &shot_rng, &current_erasure_state,
                    &last_check_result, &check_results);

                const size_t detector_offset =
                    static_cast<size_t>(shot) * static_cast<size_t>(out.num_detectors);
                const size_t observable_offset =
                    static_cast<size_t>(shot) * static_cast<size_t>(out.num_observables);
                fill_stim_syndrome_row(
                    circuit,
                    (static_cast<uint64_t>(seed) << 32) ^ static_cast<uint64_t>(shot),
                    &out.detector_samples,
                    detector_offset,
                    &out.observable_flips,
                    observable_offset,
                    out.num_detectors,
                    out.num_observables);

                if (!check_results.empty()) {
                    const size_t check_offset =
                        static_cast<size_t>(shot) * static_cast<size_t>(out.num_checks);
                    std::copy(
                        check_results.begin(),
                        check_results.end(),
                        out.check_flags.begin() + static_cast<ptrdiff_t>(check_offset));
                }
            }
        });
    }

    for (std::thread& worker : workers) {
        worker.join();
    }
    return out;
}

std::pair<stim::Circuit, std::vector<uint8_t>> StreamSampler::sample_exact_shot(
    uint32_t seed,
    uint32_t shot) const {
    const uint64_t max_persistence = static_cast<uint64_t>(program_.max_persistence());
    std::vector<uint64_t> current_erasure_state(program_.max_qubit_index() + 1, 0);
    std::vector<uint8_t> last_check_result(program_.max_qubit_index() + 1, 0);
    std::vector<uint8_t> check_results(program_.num_checks(), 0);
    FastRng shot_rng(seed, shot);
    stim::Circuit circuit = build_sampled_logical_circuit(
        program_,
        max_persistence,
        &shot_rng,
        &current_erasure_state,
        &last_check_result,
        &check_results);
    return {std::move(circuit), std::move(check_results)};
}

} // namespace qerasure::simulator
