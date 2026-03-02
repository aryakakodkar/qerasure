

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "core/simulator/injector.h"
#include "core/simulator/erasure_sampler.h"
#include "stim/circuit/gate_target.h"

namespace qerasure::simulator {

namespace {

// Appends one circuit-model Stim instruction into a Stim circuit.
// For DETECTOR and OBSERVABLE_INCLUDE, targets are stored as positive rec lookbacks.
void append_mapped_stim_instruction(const circuit::Instruction& instr, stim::Circuit* circuit) {
  switch (instr.op) {
    case circuit::OpCode::H:
      circuit->safe_append_u("H", instr.targets);
      return;
    case circuit::OpCode::CX:
      circuit->safe_append_u("CX", instr.targets);
      return;
    case circuit::OpCode::M:
      circuit->safe_append_u("M", instr.targets);
      return;
    case circuit::OpCode::R:
      circuit->safe_append_u("R", instr.targets);
      return;
    case circuit::OpCode::MR:
      circuit->safe_append_u("MR", instr.targets);
      return;
    case circuit::OpCode::X_ERROR:
      circuit->safe_append_ua("X_ERROR", instr.targets, instr.arg);
      return;
    case circuit::OpCode::Z_ERROR:
      circuit->safe_append_ua("Z_ERROR", instr.targets, instr.arg);
      return;
    case circuit::OpCode::DEPOLARIZE1:
      circuit->safe_append_ua("DEPOLARIZE1", instr.targets, instr.arg);
      return;
    case circuit::OpCode::DETECTOR: {
      std::vector<uint32_t> rec_targets;
      rec_targets.reserve(instr.targets.size());
      for (const uint32_t lookback : instr.targets) {
        rec_targets.push_back(lookback | stim::TARGET_RECORD_BIT);
      }
      circuit->safe_append_u("DETECTOR", rec_targets);
      return;
    }
    case circuit::OpCode::OBSERVABLE_INCLUDE: {
      std::vector<uint32_t> rec_targets;
      rec_targets.reserve(instr.targets.size());
      for (const uint32_t lookback : instr.targets) {
        rec_targets.push_back(lookback | stim::TARGET_RECORD_BIT);
      }
      circuit->safe_append_ua("OBSERVABLE_INCLUDE", rec_targets, 0.0);
      return;
    }
    default:
      throw std::invalid_argument(
          "append_mapped_stim_instruction only supports Stim-compatible opcodes.");
  }
}

void append_mapped_pauli_operation(const uint32_t qubit_index, const simulator::PauliOperation operation, stim::Circuit* circuit) {
  	switch (operation) {
		case simulator::PauliOperation::I:
	  		return; 
		case simulator::PauliOperation::X:
	  		circuit->safe_append_ua("X_ERROR", {qubit_index}, 1.0);
	  		return;
		case simulator::PauliOperation::Y:
	  		circuit->safe_append_ua("Y_ERROR", {qubit_index}, 1.0);
	  		return;
		case simulator::PauliOperation::Z:
	  		circuit->safe_append_ua("Z_ERROR", {qubit_index}, 1.0);
	  	return;
  }

}  
} // namespace

Injector::Injector() = default;

stim::Circuit Injector::inject(const SampledBatch& batch, uint32_t shot_index = 0) {
	stim::Circuit circuit;
    uint32_t num_shots_sampled = 0; // for debug purposes only
	const SampledShot& shot = batch.shots.at(shot_index);

	for (const SampledOperationGroup& group : shot.operation_groups) {
		if (group.stim_instruction.has_value()) {
			append_mapped_stim_instruction(group.stim_instruction.value(), &circuit);
		}

		// unclear what to do on onset
		for (const SampledOnset& onset : group.onsets) {
			(void)onset;
		}
		
		// TODO: Worth grouping by Pauli operation?
		for (const SampledSpread& spread : group.spreads) {
			append_mapped_pauli_operation(spread.qubit_index, spread.operation, &circuit);
		}

		// will need to output check info later
		for (const SampledCheck& check : group.checks) {
			(void)check;
		}

		for (const SampledReset& reset : group.resets) {
			append_mapped_pauli_operation(reset.qubit_index, reset.operation, &circuit);
		}

		num_shots_sampled++;
	}

	return circuit;
}

}  // namespace qerasure::simulator
