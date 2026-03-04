#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "core/circuit/compile.h"
#include "stim/circuit/circuit.h"

namespace qerasure::decode {

struct SpreadInjectionEvent {
	uint32_t op_index;
	uint32_t target_qubit;
	double p_x;
	double p_y;
	double p_z;
};

using SpreadInjectionBuckets = std::vector<std::vector<SpreadInjectionEvent>>;

class SurfHMMDecoder {
	public:
	explicit SurfHMMDecoder(const circuit::CompiledErasureProgram& program);

	// Computes posterior-weighted spread injections for a single shot.
	SpreadInjectionBuckets compute_spread_injections(
		const std::vector<uint8_t>* check_results,
		bool print_posteriors = false) const;

	// Builds and returns a Stim circuit with spread injections added in time order.
	stim::Circuit decode(
		const stim::Circuit& base_circuit,
		const std::vector<uint8_t>* check_results,
		bool print_posteriors = false) const;

	// Iterates operation-group indices in qubit-local offset range [start, end].
	void for_each_operation_in_qubit_range(
		uint32_t qubit_index, uint32_t start_qubit_op_offset, uint32_t end_qubit_op_offset,
		const std::function<void(uint32_t op_index, const circuit::OperationGroup&)>& fn) const;

	private:
	const circuit::CompiledErasureProgram& program_;

	// Global check-event order lookups.
	std::vector<uint32_t> check_event_to_qubit_;
	std::vector<uint32_t> check_event_to_op_index_;
	// Maps operation-group index to the visible Stim emission slot.
	std::vector<uint32_t> op_to_emit_op_index_;
};

}  // namespace qerasure::decode
