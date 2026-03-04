#pragma once

#include <cstdint>
#include <vector>

#include "core/circuit/compile.h"
#include "stim/circuit/circuit.h"

namespace qerasure::decode {

struct HMMDecoder {
	public:
    	explicit HMMDecoder(const circuit::CompiledErasureProgram& program);

		void decode(const stim::Circuit& circuit, const std::vector<uint8_t>& check_results);

	private:
		const circuit::CompiledErasureProgram& program_;

		std::vector<uint32_t> check_to_op_index;
		std::vector<uint32_t> check_to_qubit_index;
};

}  // namespace qerasure::decode
