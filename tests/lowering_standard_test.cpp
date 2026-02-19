#include "qerasure/core/code/rotated_surface_code.h"
#include "qerasure/core/lowering/lowering.h"
#include "qerasure/core/sim/erasure_simulator.h"

#include <stdexcept>

int main() {
  using namespace qerasure;

  RotatedSurfaceCode code(3);
  const std::size_t data_idx = 4;  // Center data qubit has all ancilla slots on d=3.
  if (data_idx >= code.x_anc_offset()) {
    throw std::runtime_error("Test data index is out of data-qubit range");
  }
  if (code.data_to_x_ancilla_slots()[data_idx].first == kNoPartner ||
      code.data_to_x_ancilla_slots()[data_idx].second == kNoPartner ||
      code.data_to_z_ancilla_slots()[data_idx].first == kNoPartner ||
      code.data_to_z_ancilla_slots()[data_idx].second == kNoPartner) {
    throw std::runtime_error("Expected all partner slots for center data qubit");
  }

  ErasureSimResult sim_result;
  sim_result.sparse_erasures.resize(1);
  sim_result.erasure_timestep_offsets.resize(1);
  sim_result.sparse_erasures[0] = {{data_idx, EventType::ERASURE}};
  // One gate timestep then terminal offset.
  sim_result.erasure_timestep_offsets[0] = {0, 1, 1};

  SpreadProgram program;
  program.add_correlated_error(0.0, {{PauliError::X_ERROR, PartnerSlot::X_1}});
  program.add_else_correlated_error(
      1.0,
      {{PauliError::Z_ERROR, PartnerSlot::X_1},
       {PauliError::Z_ERROR, PartnerSlot::X_2},
       {PauliError::Z_ERROR, PartnerSlot::Z_1},
       {PauliError::Z_ERROR, PartnerSlot::Z_2}});
  LoweredErrorParams reset_none{PauliError::NO_ERROR, 0.0};
  SpreadProgram default_program;
  LoweringParams params(default_program, reset_none);
  params.set_data_qubit_program(data_idx, program);

  Lowerer lowerer(code, params);
  LoweringResult result = lowerer.lower(sim_result);

  if (result.sparse_cliffords.size() != 1 || result.sparse_cliffords[0].empty()) {
    throw std::runtime_error("Expected at least one lowering event from correlated chain");
  }
  for (const LoweredErrorEvent& event : result.sparse_cliffords[0]) {
    if (event.error_type != PauliError::Z_ERROR) {
      throw std::runtime_error("ELSE_CORRELATED_ERROR path produced unexpected error type");
    }
  }

  SpreadProgram program_fire;
  program_fire.add_correlated_error(
      1.0,
      {{PauliError::X_ERROR, PartnerSlot::X_1},
       {PauliError::X_ERROR, PartnerSlot::X_2},
       {PauliError::X_ERROR, PartnerSlot::Z_1},
       {PauliError::X_ERROR, PartnerSlot::Z_2}});
  program_fire.add_else_correlated_error(
      1.0,
      {{PauliError::Z_ERROR, PartnerSlot::X_1},
       {PauliError::Z_ERROR, PartnerSlot::X_2},
       {PauliError::Z_ERROR, PartnerSlot::Z_1},
       {PauliError::Z_ERROR, PartnerSlot::Z_2}});
  params.set_data_qubit_program(data_idx, program_fire);

  Lowerer lowerer_fire(code, params);
  result = lowerer_fire.lower(sim_result);
  if (result.sparse_cliffords[0].empty()) {
    throw std::runtime_error("Expected correlated events when first branch fires");
  }
  for (const LoweredErrorEvent& event : result.sparse_cliffords[0]) {
    if (event.error_type != PauliError::X_ERROR) {
      throw std::runtime_error("CORRELATED_ERROR path produced unexpected error type");
    }
  }

  return 0;
}
