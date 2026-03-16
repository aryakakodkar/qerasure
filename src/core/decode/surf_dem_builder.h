#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
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
using SkippableReweightMap = std::unordered_map<uint64_t, double>;

class SurfDemBuilder {
	public:
	explicit SurfDemBuilder(const circuit::CompiledErasureProgram& program);

	// Computes posterior-weighted spread injections for a single shot.
	SpreadInjectionBuckets compute_spread_injections(
		const std::vector<uint8_t>* check_results,
		bool verbose = false,
		SkippableReweightMap* skippable_reweights = nullptr) const;

	// Builds and returns a Stim circuit with spread injections added in time order.
	stim::Circuit build_decoded_circuit(
		const std::vector<uint8_t>* check_results,
		bool verbose = false) const;

	// Builds a text debug representation of the decoded circuit.
	// This does not enforce Stim disjointness constraints and is useful for
	// diagnosing invalid PAULI_CHANNEL_1 probability tuples.
	std::string build_decoded_circuit_text(
		const std::vector<uint8_t>* check_results,
		bool verbose = false) const;

	private:
	const circuit::CompiledErasureProgram& program_;

	// Propagates hidden-erasure mass over the end-of-circuit tail window and
	// emits any resulting spread, reweight, and measurement-randomization terms.
	void add_tail_hidden_injections(
		const std::vector<uint8_t>* check_results,
		uint32_t qubit,
		uint32_t start_op,
		uint32_t final_meas_op,
		SpreadInjectionBuckets* buckets,
		SkippableReweightMap* skippable_reweights) const;

	// Global check-event order lookups.
	std::vector<uint32_t> check_event_to_qubit_;
	std::vector<uint32_t> check_event_to_op_index_;
	// Per-qubit check events in the same local order as qubit_check_operation_indices.
	std::vector<std::vector<uint32_t>> qubit_check_events_;
	// Maps operation-group index to the visible Stim emission slot.
	std::vector<uint32_t> op_to_emit_op_index_;
};

}  // namespace qerasure::decode
