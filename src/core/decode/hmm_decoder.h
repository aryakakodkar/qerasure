#pragma once

#include <cstdint>
#include <vector>

#include "core/circuit/compile.h"
#include "stim/circuit/circuit.h"

namespace qerasure::decode {

struct FlaggedCheckMapping {
  // Global check-event index in sampled check-results order.
  uint32_t check_event_index;
  // Checked qubit for this event.
  uint32_t qubit_index;
  // Operation-group index where the check occurred.
  uint32_t op_index;
  // Position of `op_index` inside program.qubit_operation_indices[qubit_index].
  uint32_t qubit_operation_offset;
};

struct HmmDecoder {
 public:
  explicit HmmDecoder(const circuit::CompiledErasureProgram& program);

  // Callback-compatible shot hook.
  // `check_results` is expected to be in compiled check-event order.
  void process_shot(const stim::Circuit& circuit, const std::vector<uint8_t>* check_results);

  const std::vector<FlaggedCheckMapping>& flagged_check_mappings() const {
    return flagged_check_mappings_;
  }

 private:
  const circuit::CompiledErasureProgram& program_;

  // Event-order lookup tables built once from CompiledErasureProgram.
  std::vector<uint32_t> check_event_to_qubit_;
  std::vector<uint32_t> check_event_to_op_index_;

  std::vector<FlaggedCheckMapping> flagged_check_mappings_;
};

}  // namespace qerasure::decode
