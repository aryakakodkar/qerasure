#include "core/decode/hmm_decoder.h"

#include <algorithm>
#include <stdexcept>

namespace qerasure::decode {

HmmDecoder::HmmDecoder(const circuit::CompiledErasureProgram& program) : program_(program) {
  check_event_to_qubit_.reserve(program_.num_checks());
  check_event_to_op_index_.reserve(program_.num_checks());

  for (uint32_t op_index = 0; op_index < program_.operation_groups.size(); ++op_index) {
    const circuit::OperationGroup& op_group = program_.operation_groups[op_index];
    for (const circuit::ErasureCheck& check : op_group.checks) {
      check_event_to_qubit_.push_back(check.qubit_index);
      check_event_to_op_index_.push_back(op_index);
    }
  }

  if (check_event_to_qubit_.size() != program_.num_checks()) {
    throw std::logic_error("HmmDecoder check-event map size mismatch with CompiledErasureProgram");
  }
}

void HmmDecoder::process_shot(const stim::Circuit& circuit, const std::vector<uint8_t>* check_results) {
  (void)circuit;
  flagged_check_mappings_.clear();

  if (check_results == nullptr) {
    throw std::invalid_argument("HmmDecoder::process_shot requires non-null check_results pointer");
  }
  if (check_results->size() != check_event_to_qubit_.size()) {
    throw std::invalid_argument("HmmDecoder::process_shot check_results size mismatch");
  }

  for (uint32_t check_event_index = 0; check_event_index < check_results->size(); ++check_event_index) {
    const uint8_t bit = (*check_results)[check_event_index];
    if (bit == 0) {
      continue;
    }

    const uint32_t qubit = check_event_to_qubit_[check_event_index];
    const uint32_t op_index = check_event_to_op_index_[check_event_index];
    const std::vector<uint32_t>& op_indices_for_qubit = program_.qubit_operation_indices.at(qubit);

    const auto it = std::lower_bound(op_indices_for_qubit.begin(), op_indices_for_qubit.end(), op_index);
    if (it == op_indices_for_qubit.end() || *it != op_index) {
      throw std::logic_error(
          "HmmDecoder expected check op-index to be present in qubit_operation_indices");
    }

    flagged_check_mappings_.push_back(
        {check_event_index, qubit, op_index, static_cast<uint32_t>(it - op_indices_for_qubit.begin())});
  }
}

}  // namespace qerasure::decode
