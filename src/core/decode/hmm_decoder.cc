#include "core/decode/hmm_decoder.h"
#include <core/circuit/compile.h>

#include <algorithm>
#include <optional>
#include <stdexcept>

namespace qerasure::decode {

HMMDecoder::HMMDecoder(const circuit::CompiledErasureProgram& program) : program_(program) {
	if (program_.operation_groups.empty()) {
		throw std::invalid_argument("Compiled erasure program must contain at least one operation group.");
	}

	// Store indices of check operations to map erasure checks back to corresponding operations during decoding.
	check_to_op_index.reserve(program_.num_checks());
	check_to_qubit_index.reserve(program_.num_checks());

	for (size_t op_index = 0; op_index < program_.operation_groups.size(); ++op_index) {
		const circuit::OperationGroup& group = program_.operation_groups[op_index];
		for (const circuit::ErasureCheck& check : group.checks) {
			check_to_op_index.push_back(op_index);
			check_to_qubit_index.push_back(check.qubit_index);
		}
	}
}

void HMMDecoder::decode(const stim::Circuit& circuit, const std::vector<uint8_t>& check_results) {
	for (size_t check_index = 0; check_index < check_results.size(); ++check_index) {
		uint8_t result = check_results[check_index];

		if (result != 0 && result != 1) {
			throw std::invalid_argument("Check results must be binary (0 or 1).");
		}
		if (result == 0) {
			continue; // Skip non-flagged checks
		}

		// Map check result back to corresponding operation
		size_t op_index = check_to_op_index[check_index];
		size_t qubit_index = check_to_qubit_index[check_index];

		// Find the last guaranteed successful reset operation
                const circuit::CheckLookbackLink& link = program_.check_lookback_links[check_index];
		if (link.lookback_check_event_index == -1) {
			// Check within the first max_persistence checks, no previous guaranteed correct checks
		} else {
			// Check the corresponding reset
			int32_t reset_op_index = link.reset_op_after_lookback;
			if (reset_op_index == -1) {
				throw std::runtime_error("No reset within max_persistence checks for qubit " + std::to_string(qubit_index));
			}
		}
	}
}

}	// namespace qerasure::decode
