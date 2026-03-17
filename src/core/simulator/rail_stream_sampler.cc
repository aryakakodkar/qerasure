#include "core/simulator/rail_stream_sampler.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <thread>

#include "core/circuit/instruction.h"
#include "core/simulator/fast_rng.h"
#include "core/simulator/sim_internal_utils.h"
#include "stim/simulators/frame_simulator_util.h"

namespace qerasure::simulator {

namespace {

int32_t choose_z_rail(
    const circuit::RailSurfaceCompiledProgram& rail_program,
    uint32_t data_qubit,
    FastRng* rng) {
  const auto slots = rail_program.data_z_ancilla_slots(data_qubit);
  const bool has_slot0 = slots.first >= 0;
  const bool has_slot1 = slots.second >= 0;
  if (has_slot0 && has_slot1) {
    return (rng->next_u64() <= (1ULL << 63)) ? slots.first : slots.second;
  }
  if (has_slot0 || has_slot1) {
    if (rng->next_u64() <= (1ULL << 63)) {
      return has_slot0 ? slots.first : slots.second;
    }
    return -1;
  }
  return -1;
}

stim::Circuit build_sampled_logical_circuit(
    const circuit::RailSurfaceCompiledProgram& rail_program,
    FastRng* rng,
    std::vector<uint64_t>* current_erasure_state,
    std::vector<uint8_t>* last_check_result,
    std::vector<uint8_t>* check_results,
    std::vector<int32_t>* chosen_z_rail) {
  const circuit::CompiledErasureProgram& program = rail_program.base_program();
  const uint64_t max_persistence = static_cast<uint64_t>(program.max_persistence());
  stim::Circuit circuit;

  std::fill(current_erasure_state->begin(), current_erasure_state->end(), 0);
  std::fill(last_check_result->begin(), last_check_result->end(), 0);
  std::fill(check_results->begin(), check_results->end(), 0);
  std::fill(chosen_z_rail->begin(), chosen_z_rail->end(), -1);

  size_t check_idx = 0;
  for (uint32_t op_index = 0; op_index < program.operation_groups.size(); ++op_index) {
    const circuit::OperationGroup& op_group = program.operation_groups[op_index];
    if (op_group.stim_instruction.has_value()) {
      const circuit::Instruction& instr = *op_group.stim_instruction;
      if (instr.op == circuit::OpCode::CX) {
        // Rail-resolved spread: if a data qubit is already erased before this interaction,
        // it deterministically injects X on its selected Z-ancilla rail.
        for (size_t k = 0; k + 1 < instr.targets.size(); k += 2) {
          const uint32_t control = instr.targets[k];
          const uint32_t target = instr.targets[k + 1];
          if (!rail_program.is_data_qubit(control)) {
            continue;
          }
          if ((*current_erasure_state)[control] == 0) {
            continue;
          }
          if ((*chosen_z_rail)[control] != static_cast<int32_t>(target)) {
            continue;
          }
          internal::append_mapped_stim_instruction(
              circuit::OpCode::X_ERROR, {target}, 1.0, &circuit);
        }
      }

      if (circuit::is_measurement_op(instr.op)) {
        for (const uint32_t target : instr.targets) {
          if ((*current_erasure_state)[target] >= 1) {
            internal::append_mapped_stim_instruction(
                circuit::OpCode::X_ERROR, {target}, 0.5, &circuit);
          }
        }
        internal::append_mapped_stim_instruction(instr, &circuit);
      } else if (circuit::is_erasure_skippable_op(instr.op)) {
        std::vector<uint32_t> non_erased_targets;
        non_erased_targets.reserve(instr.targets.size());
        for (const uint32_t target : instr.targets) {
          if ((*current_erasure_state)[target] == 0) {
            non_erased_targets.push_back(target);
          }
        }
        if (!non_erased_targets.empty()) {
          internal::append_mapped_stim_instruction(
              instr.op, non_erased_targets, instr.arg, &circuit);
        }
      } else {
        internal::append_mapped_stim_instruction(instr, &circuit);
      }
    }

    std::vector<uint32_t> onset_qubits_this_op;
    onset_qubits_this_op.reserve(op_group.onsets.size() + op_group.onset_pairs.size());

    for (const auto& onset : op_group.onsets) {
      if (rng->next_u64() <= onset.prob_threshold) {
        const bool was_erased = (*current_erasure_state)[onset.qubit_index] != 0;
        (*current_erasure_state)[onset.qubit_index] =
            was_erased ? (*current_erasure_state)[onset.qubit_index] + 1 : 1;
        onset_qubits_this_op.push_back(onset.qubit_index);
        if (!was_erased && rail_program.is_data_qubit(onset.qubit_index)) {
          (*chosen_z_rail)[onset.qubit_index] = choose_z_rail(
              rail_program, onset.qubit_index, rng);
        }
      }
    }

    for (const auto& onset_pair : op_group.onset_pairs) {
      if (rng->next_u64() > onset_pair.prob_threshold) {
        continue;
      }
      if ((*current_erasure_state)[onset_pair.qubit_index1] != 0 ||
          (*current_erasure_state)[onset_pair.qubit_index2] != 0) {
        continue;
      }
      uint32_t unerased = 0;
      if (rng->next_u64() <= (1ULL << 63)) {
        (*current_erasure_state)[onset_pair.qubit_index1] = 1;
        onset_qubits_this_op.push_back(onset_pair.qubit_index1);
        unerased = onset_pair.qubit_index2;
        if (rail_program.is_data_qubit(onset_pair.qubit_index1)) {
          (*chosen_z_rail)[onset_pair.qubit_index1] = choose_z_rail(
              rail_program, onset_pair.qubit_index1, rng);
        }
      } else {
        (*current_erasure_state)[onset_pair.qubit_index2] = 1;
        onset_qubits_this_op.push_back(onset_pair.qubit_index2);
        unerased = onset_pair.qubit_index1;
        if (rail_program.is_data_qubit(onset_pair.qubit_index2)) {
          (*chosen_z_rail)[onset_pair.qubit_index2] = choose_z_rail(
              rail_program, onset_pair.qubit_index2, rng);
        }
      }

      internal::PauliOperation sampled_op =
          internal::sample_thresholded_pauli_channel(program.thresholded_onset_channel(), rng);
      if (sampled_op != internal::PauliOperation::I) {
        internal::append_mapped_pauli_operation(unerased, sampled_op, &circuit);
      }
    }

    for (const auto& spread : op_group.onset_spreads) {
      if ((*current_erasure_state)[spread.aff_qubit_index] != 0 ||
          std::find(
              onset_qubits_this_op.begin(),
              onset_qubits_this_op.end(),
              spread.source_qubit_index) == onset_qubits_this_op.end()) {
        continue;
      }
      const internal::PauliOperation sampled_op =
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
      const internal::PauliOperation sampled_op =
          internal::sample_thresholded_pauli_channel(spread.spread_channel, rng);
      if (sampled_op != internal::PauliOperation::I) {
        internal::append_mapped_pauli_operation(spread.aff_qubit_index, sampled_op, &circuit);
      }
    }

    for (const auto& check : op_group.checks) {
      if ((*current_erasure_state)[check.qubit_index] == 0) {
        const bool false_positive = rng->next_u64() <= check.false_positive_threshold;
        (*check_results)[check_idx++] = false_positive ? 1 : 0;
        (*last_check_result)[check.qubit_index] = false_positive ? 1 : 0;
      } else {
        const bool force_true_positive = (*current_erasure_state)[check.qubit_index] >= max_persistence;
        if (force_true_positive) {
          (*check_results)[check_idx++] = 1;
          (*last_check_result)[check.qubit_index] = 1;
        } else {
          const bool false_negative = rng->next_u64() <= check.false_negative_threshold;
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
      if ((*last_check_result)[reset.qubit_index] != 1) {
        continue;
      }
      const bool reset_failed = rng->next_u64() <= reset.reset_failure_threshold;
      if (!reset_failed) {
        const internal::PauliOperation sampled_op =
            internal::sample_thresholded_pauli_channel(reset.reset_channel, rng);
        internal::append_mapped_pauli_operation(reset.qubit_index, sampled_op, &circuit);
        (*current_erasure_state)[reset.qubit_index] = 0;
        if (rail_program.is_data_qubit(reset.qubit_index)) {
          (*chosen_z_rail)[reset.qubit_index] = -1;
        }
      }
      (*last_check_result)[reset.qubit_index] = 0;
    }
  }

  if (check_idx != check_results->size()) {
    throw std::logic_error("RailStreamSampler check vector size mismatch");
  }
  return circuit;
}

void fill_stim_syndrome_row(
    const stim::Circuit& circuit,
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
    throw std::logic_error("RailStreamSampler detector/observable width changed across shots");
  }

  for (uint32_t d = 0; d < num_detectors; ++d) {
    (*detector_samples)[detector_row_offset + d] = det_data[d][0] ? 1 : 0;
  }
  for (uint32_t o = 0; o < num_observables; ++o) {
    (*observable_flips)[observable_row_offset + o] = obs_data[o][0] ? 1 : 0;
  }
}

}  // namespace

SyndromeSampleBatch RailStreamSampler::sample_syndromes(
    uint32_t num_shots,
    uint32_t seed,
    uint32_t num_threads) {
  const circuit::CompiledErasureProgram& base = program_.base_program();
  SyndromeSampleBatch out;
  out.num_shots = num_shots;
  out.num_checks = base.num_checks();
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

  {
    std::vector<uint64_t> current_erasure_state(base.max_qubit_index() + 1, 0);
    std::vector<uint8_t> last_check_result(base.max_qubit_index() + 1, 0);
    std::vector<uint8_t> check_results(base.num_checks(), 0);
    std::vector<int32_t> chosen_z_rail(base.max_qubit_index() + 1, -1);
    FastRng rng0(seed, 0);
    stim::Circuit circuit0 = build_sampled_logical_circuit(
        program_, &rng0, &current_erasure_state, &last_check_result, &check_results, &chosen_z_rail);

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
        0,
        &out.observable_flips,
        0,
        out.num_detectors,
        out.num_observables);
    if (!check_results.empty()) {
      std::copy(check_results.begin(), check_results.end(), out.check_flags.begin());
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
      std::vector<uint64_t> current_erasure_state(base.max_qubit_index() + 1, 0);
      std::vector<uint8_t> last_check_result(base.max_qubit_index() + 1, 0);
      std::vector<uint8_t> check_results(base.num_checks(), 0);
      std::vector<int32_t> chosen_z_rail(base.max_qubit_index() + 1, -1);
      while (true) {
        const uint32_t shot = next_shot.fetch_add(1, std::memory_order_relaxed);
        if (shot >= num_shots) {
          break;
        }
        FastRng shot_rng(seed, shot);
        stim::Circuit circuit = build_sampled_logical_circuit(
            program_,
            &shot_rng,
            &current_erasure_state,
            &last_check_result,
            &check_results,
            &chosen_z_rail);

        const size_t detector_offset = static_cast<size_t>(shot) * out.num_detectors;
        const size_t observable_offset = static_cast<size_t>(shot) * out.num_observables;
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
          const size_t check_offset = static_cast<size_t>(shot) * out.num_checks;
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

std::pair<stim::Circuit, std::vector<uint8_t>> RailStreamSampler::sample_exact_shot(
    uint32_t seed,
    uint32_t shot) const {
  const circuit::CompiledErasureProgram& base = program_.base_program();
  std::vector<uint64_t> current_erasure_state(base.max_qubit_index() + 1, 0);
  std::vector<uint8_t> last_check_result(base.max_qubit_index() + 1, 0);
  std::vector<uint8_t> check_results(base.num_checks(), 0);
  std::vector<int32_t> chosen_z_rail(base.max_qubit_index() + 1, -1);
  FastRng shot_rng(seed, shot);
  stim::Circuit circuit = build_sampled_logical_circuit(
      program_,
      &shot_rng,
      &current_erasure_state,
      &last_check_result,
      &check_results,
      &chosen_z_rail);
  return {std::move(circuit), std::move(check_results)};
}

}  // namespace qerasure::simulator
