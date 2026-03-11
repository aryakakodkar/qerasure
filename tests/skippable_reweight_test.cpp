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

std::vector<double> extract_single_arg_probs(const std::string& stim_text, const std::string& op_name) {
	std::vector<double> probs;
	std::istringstream in(stim_text);
	std::string line;
	const std::string prefix = op_name + "(";
	while (std::getline(in, line)) {
		if (line.rfind(prefix, 0) != 0) {
			continue;
		}
		const std::size_t open = line.find('(');
		const std::size_t close = line.find(')');
		if (open == std::string::npos || close == std::string::npos || close <= open + 1) {
			throw std::runtime_error("Malformed probabilistic line in decoded circuit.");
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

	// Ensure skippable ops between earlier checks and a later flagged check are reweighted.
	ErasureCircuit lookback_circuit;
	lookback_circuit.append(OpCode::ERASE, {0}, 1.0);         // establish erasable tracking
	lookback_circuit.append(OpCode::ECR, {0}, 0.0);           // check #0
	lookback_circuit.append(OpCode::ERASE, {0}, 1.0);         // onset after check #0
	lookback_circuit.append(OpCode::DEPOLARIZE1, {0}, 0.2);   // before check #1
	lookback_circuit.append(OpCode::X_ERROR, {0}, 0.2);       // before check #1
	lookback_circuit.append(OpCode::ECR, {0}, 0.0);           // check #1
	lookback_circuit.append(OpCode::DEPOLARIZE1, {0}, 0.2);   // after check #1 (n-1 interval)
	lookback_circuit.append(OpCode::X_ERROR, {0}, 0.2);       // after check #1 (n-1 interval)
	lookback_circuit.append(OpCode::ECR, {0}, 0.0);           // check #2 (flagged)
	lookback_circuit.append(OpCode::M, {0});

	ErasureModel lookback_model(
		/*max_persistence=*/2,
		/*onset=*/PauliChannel(0.0, 0.0, 0.0),
		/*reset=*/PauliChannel(0.0, 0.0, 0.0),
		/*spread=*/TQGSpreadModel(PauliChannel(0.0, 0.0, 0.0), PauliChannel(0.0, 0.0, 0.0)));
	// Force missed intermediate checks and forced detection at persistence boundary.
	lookback_model.check_false_negative_prob = 1.0;
	lookback_model.check_false_positive_prob = 0.0;

	const CompiledErasureProgram lookback_compiled(lookback_circuit, lookback_model);
	if (lookback_compiled.num_checks() != 3) {
		throw std::runtime_error("Expected exactly three check events in lookback reweight test.");
	}
	SurfDemBuilder lookback_decoder(lookback_compiled);

	const std::vector<uint8_t> lookback_unflagged = {0, 0, 0};
	const std::string lookback_unflagged_text =
		lookback_decoder.build_decoded_circuit_text(&lookback_unflagged, /*verbose=*/false);
	const std::vector<double> lookback_unflagged_depolarize =
		extract_single_arg_probs(lookback_unflagged_text, "DEPOLARIZE1");
	const std::vector<double> lookback_unflagged_x =
		extract_single_arg_probs(lookback_unflagged_text, "X_ERROR");
	if (lookback_unflagged_depolarize.size() != 2 || lookback_unflagged_x.size() != 2) {
		throw std::runtime_error(
			"Expected two DEPOLARIZE1 and two X_ERROR ops in unflagged lookback decoded circuit.");
	}

	const std::vector<uint8_t> lookback_flagged = {0, 0, 1};
	const std::string lookback_flagged_text =
		lookback_decoder.build_decoded_circuit_text(&lookback_flagged, /*verbose=*/false);
	const std::vector<double> lookback_flagged_depolarize =
		extract_single_arg_probs(lookback_flagged_text, "DEPOLARIZE1");
	const std::vector<double> lookback_flagged_x =
		extract_single_arg_probs(lookback_flagged_text, "X_ERROR");
	if (!lookback_flagged_depolarize.empty() || !lookback_flagged_x.empty()) {
		throw std::runtime_error(
			"Lookback-window skippable DEPOLARIZE1/X_ERROR ops were not fully reweighted away.");
	}

	return 0;
}
