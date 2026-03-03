
#include <cstdint>

#include "core/simulator/injector.h"
#include "core/simulator/erasure_sampler.h"
#include "core/simulator/sim_internal_utils.h"

namespace qerasure::simulator {

namespace {

internal::PauliOperation to_internal_pauli_operation(PauliOperation op) {
  switch (op) {
    case PauliOperation::X:
      return internal::PauliOperation::X;
    case PauliOperation::Y:
      return internal::PauliOperation::Y;
    case PauliOperation::Z:
      return internal::PauliOperation::Z;
    case PauliOperation::I:
      return internal::PauliOperation::I;
  }
  return internal::PauliOperation::I;
}

}  // namespace

Injector::Injector() = default;

stim::Circuit Injector::inject(const SampledBatch& batch, uint32_t shot_index = 0) {
	stim::Circuit circuit;
    uint32_t num_shots_sampled = 0; // for debug purposes only
	const SampledShot& shot = batch.shots.at(shot_index);

	for (const SampledOperationGroup& group : shot.operation_groups) {
		if (group.stim_instruction.has_value()) {
			internal::append_mapped_stim_instruction(group.stim_instruction.value(), &circuit);
		}

		// unclear what to do on onset
		for (const SampledOnset& onset : group.onsets) {
			(void)onset;
		}
		
		// TODO: Worth grouping by Pauli operation?
		for (const SampledSpread& spread : group.spreads) {
			internal::append_mapped_pauli_operation(
          spread.qubit_index, to_internal_pauli_operation(spread.operation), &circuit);
		}

		// will need to output check info later
		for (const SampledCheck& check : group.checks) {
			(void)check;
		}

		for (const SampledReset& reset : group.resets) {
			internal::append_mapped_pauli_operation(
          reset.qubit_index, to_internal_pauli_operation(reset.operation), &circuit);
		}

		num_shots_sampled++;
	}

	return circuit;
}

}  // namespace qerasure::simulator
