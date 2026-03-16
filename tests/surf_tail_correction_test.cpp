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

std::vector<double> extract_pauli_channel_x_probs(const std::string& stim_text) {
	std::vector<double> probs;
	std::istringstream in(stim_text);
	std::string line;
	while (std::getline(in, line)) {
		if (line.rfind("PAULI_CHANNEL_1(", 0) != 0) {
			continue;
		}
		const std::size_t open = line.find('(');
		const std::size_t comma = line.find(',', open);
		if (open == std::string::npos || comma == std::string::npos || comma <= open + 1) {
			throw std::runtime_error("Malformed PAULI_CHANNEL_1 line in decoded circuit text.");
		}
		probs.push_back(std::stod(line.substr(open + 1, comma - open - 1)));
	}
	return probs;
}

int count_exact_prob_target(
	const std::string& stim_text,
	double expected_prob,
	uint32_t target_qubit) {
	int count = 0;
	std::istringstream in(stim_text);
	std::string line;
	while (std::getline(in, line)) {
		if (line.rfind("PAULI_CHANNEL_1(", 0) != 0) {
			continue;
		}
		const std::size_t open = line.find('(');
		const std::size_t comma = line.find(',', open);
		const std::size_t close = line.find(')');
		const std::size_t last_space = line.find_last_of(' ');
		if (open == std::string::npos || comma == std::string::npos || close == std::string::npos ||
			last_space == std::string::npos) {
			throw std::runtime_error("Malformed PAULI_CHANNEL_1 line in decoded circuit text.");
		}
		const double p_x = std::stod(line.substr(open + 1, comma - open - 1));
		const uint32_t target = static_cast<uint32_t>(std::stoul(line.substr(last_space + 1)));
		if (std::fabs(p_x - expected_prob) < 1e-12 && target == target_qubit) {
			count++;
		}
	}
	return count;
}

bool contains_prob(const std::vector<double>& probs, double expected) {
	for (double p : probs) {
		if (std::fabs(p - expected) < 1e-12) {
			return true;
		}
	}
	return false;
}

}  // namespace

int main() {
	using namespace qerasure::circuit;  // NOLINT
	using namespace qerasure::decode;   // NOLINT

	ErasureCircuit circuit;
	circuit.append(OpCode::ERASE, {0}, 0.0);  // establish erasure tracking before the first check
	circuit.append(OpCode::ECR, {0}, 0.0);    // check #0
	circuit.append(OpCode::ERASE, {0}, 1.0);  // onset after check #0
	circuit.append(OpCode::ECR, {0}, 0.0);    // check #1
	circuit.append(OpCode::ECR, {0}, 0.0);    // check #2
	circuit.append(OpCode::M, {0});

	ErasureModel model(
		/*max_persistence=*/3,
		/*onset=*/PauliChannel(0.0, 0.0, 0.0),
		/*reset=*/PauliChannel(0.0, 0.0, 0.0),
		/*spread=*/TQGSpreadModel(PauliChannel(0.0, 0.0, 0.0), PauliChannel(0.0, 0.0, 0.0)));
	model.check_false_negative_prob = 1.0;
	model.check_false_positive_prob = 0.0;

	const CompiledErasureProgram compiled(circuit, model);
	if (compiled.num_checks() != 3) {
		throw std::runtime_error("Expected exactly three checks in surf_tail_correction_test.");
	}

	SurfDemBuilder decoder(compiled);
	const std::vector<uint8_t> check_results = {0, 0, 0};
	const std::string decoded_text =
		decoder.build_decoded_circuit_text(&check_results, /*verbose=*/false);
	const std::vector<double> x_probs = extract_pauli_channel_x_probs(decoded_text);

	if (!contains_prob(x_probs, 0.5)) {
		throw std::runtime_error(
			"Expected a PAULI_CHANNEL_1 with X probability 0.5 for the missed final-measurement erasure tail.");
	}

	ErasureCircuit spread_circuit;
	spread_circuit.append(OpCode::ERASE, {0}, 0.0);  // establish tracking before check #0
	spread_circuit.append(OpCode::ECR, {0}, 0.0);    // check #0
	spread_circuit.append(OpCode::ERASE, {0}, 1.0);  // hidden onset after check #0
	spread_circuit.append(OpCode::CX, {0, 1}, 0.0);  // last two rounds can now spread from qubit 0
	spread_circuit.append(OpCode::ECR, {0}, 0.0);    // check #1 missed
	spread_circuit.append(OpCode::CX, {0, 1}, 0.0);
	spread_circuit.append(OpCode::ECR, {0}, 0.0);    // final check #2 missed
	spread_circuit.append(OpCode::M, {0, 1});

	ErasureModel spread_model(
		/*max_persistence=*/3,
		/*onset=*/PauliChannel(0.0, 0.0, 0.0),
		/*reset=*/PauliChannel(0.0, 0.0, 0.0),
		/*spread=*/TQGSpreadModel(PauliChannel(1.0, 0.0, 0.0), PauliChannel(0.0, 0.0, 0.0)));
	spread_model.check_false_negative_prob = 1.0;
	spread_model.check_false_positive_prob = 0.0;

	const CompiledErasureProgram spread_compiled(spread_circuit, spread_model);
	SurfDemBuilder spread_decoder(spread_compiled);
	const std::vector<uint8_t> spread_checks = {0, 0, 0};
	const std::string spread_text =
		spread_decoder.build_decoded_circuit_text(&spread_checks, /*verbose=*/false);
	const int hidden_tail_spreads = count_exact_prob_target(spread_text, 1.0, 1);
	if (hidden_tail_spreads < 2) {
		throw std::runtime_error(
			"Expected hidden-tail persistent spread injections on the target qubit across both final rounds.");
	}

	return 0;
}
