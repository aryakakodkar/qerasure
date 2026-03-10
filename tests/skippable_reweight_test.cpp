#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/circuit/circuit.h"
#include "core/circuit/compile.h"
#include "core/circuit/erasure_model.h"
#include "core/decode/surf_dem_builder.h"
#include "core/model/pauli_channel.h"

namespace {

std::vector<double> extract_depolarize_probs(const std::string& stim_text) {
	std::vector<double> probs;
	std::istringstream in(stim_text);
	std::string line;
	while (std::getline(in, line)) {
		if (line.rfind("DEPOLARIZE1(", 0) != 0) {
			continue;
		}
		const std::size_t open = line.find('(');
		const std::size_t close = line.find(')');
		if (open == std::string::npos || close == std::string::npos || close <= open + 1) {
			throw std::runtime_error("Malformed DEPOLARIZE1 line in decoded circuit.");
		}
		probs.push_back(std::stod(line.substr(open + 1, close - open - 1)));
	}
	return probs;
}

}  // namespace

int main() {
	using namespace qerasure::circuit;  // NOLINT
	using namespace qerasure::decode;   // NOLINT

	ErasureCircuit circuit;
	circuit.append(OpCode::H, {0});
	circuit.append(OpCode::ERASE, {0}, 0.5);
	circuit.append(OpCode::DEPOLARIZE1, {0}, 0.2);
	circuit.append(OpCode::ECR, {0}, 0.0);
	circuit.append(OpCode::M, {0});

	ErasureModel model(
		/*max_persistence=*/2,
		/*onset=*/PauliChannel(0.0, 0.0, 0.0),
		/*reset=*/PauliChannel(0.0, 0.0, 0.0),
		/*spread=*/TQGSpreadModel(PauliChannel(0.0, 0.0, 0.0), PauliChannel(0.0, 0.0, 0.0)));
	model.check_false_negative_prob = 0.0;
	model.check_false_positive_prob = 0.0;

	const CompiledErasureProgram compiled(circuit, model);
	if (compiled.num_checks() != 1) {
		throw std::runtime_error("Expected exactly one check event in skippable_reweight_test.");
	}
	if (compiled.qubit_skippable_operation_indices.size() <= 0 ||
		compiled.qubit_skippable_operation_indices[0].size() != 1) {
		throw std::runtime_error("Compiler did not record expected skippable op indices.");
	}

	SurfDemBuilder decoder(compiled);

	const std::vector<uint8_t> unflagged_checks = {0};
	const stim::Circuit unflagged_decoded =
		decoder.build_decoded_circuit(&unflagged_checks, /*verbose=*/false);
	const std::vector<double> unflagged_probs = extract_depolarize_probs(unflagged_decoded.str());
	if (unflagged_probs.size() != 1) {
		throw std::runtime_error("Expected one DEPOLARIZE1 in unflagged decoded circuit.");
	}
	if (std::fabs(unflagged_probs[0] - 0.2) > 1e-12) {
		throw std::runtime_error("Unflagged DEPOLARIZE1 probability changed unexpectedly.");
	}

	const std::vector<uint8_t> flagged_checks = {1};
	const stim::Circuit flagged_decoded =
		decoder.build_decoded_circuit(&flagged_checks, /*verbose=*/false);
	const std::vector<double> flagged_probs = extract_depolarize_probs(flagged_decoded.str());
	if (!flagged_probs.empty()) {
		throw std::runtime_error("Flagged decoded circuit still contains DEPOLARIZE1 after reweight.");
	}

	return 0;
}
