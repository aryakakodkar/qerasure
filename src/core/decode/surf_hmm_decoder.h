#pragma once

#include <cstdint>
#include <string>
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
		bool verbose = false) const;

	// Builds and returns a Stim circuit with spread injections added in time order.
	stim::Circuit decode(
		const std::vector<uint8_t>* check_results,
		bool verbose = false) const;

	// Builds a text debug representation of the decoded circuit.
	// This does not enforce Stim disjointness constraints and is useful for
	// diagnosing invalid PAULI_CHANNEL_1 probability tuples.
	std::string debug_decoded_circuit_text(
		const std::vector<uint8_t>* check_results,
		bool verbose = false) const;

	private:
	const circuit::CompiledErasureProgram& program_;

	// Global check-event order lookups.
	std::vector<uint32_t> check_event_to_qubit_;
	std::vector<uint32_t> check_event_to_op_index_;
	// Maps operation-group index to the visible Stim emission slot.
	std::vector<uint32_t> op_to_emit_op_index_;
};

}  // namespace qerasure::decode
