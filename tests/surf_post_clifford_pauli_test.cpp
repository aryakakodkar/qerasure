#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "core/circuit/instruction.h"
#include "core/gen/surf.h"

namespace {

bool almost_equal(double a, double b, double tol = 1e-12) {
	return std::fabs(a - b) <= tol;
}

void verify_no_depolarize(const qerasure::circuit::ErasureCircuit& circuit) {
	for (const auto& instr : circuit.instructions()) {
		if (instr.op == qerasure::circuit::OpCode::DEPOLARIZE1 ||
			instr.op == qerasure::circuit::OpCode::X_ERROR) {
			throw std::runtime_error(
				"Found injected post-clifford/pre-measurement error even though post_clifford_pauli_prob=0.");
		}
	}
}

void verify_post_clifford_layout(const qerasure::circuit::ErasureCircuit& circuit,
								 bool single_qubit_errors,
								 double expected_prob) {
	using namespace qerasure::circuit;  // NOLINT

	const std::vector<Instruction>& instructions = circuit.instructions();
	uint64_t h_count = 0;
	uint64_t cx_count = 0;
	uint64_t ecr_count = 0;
	uint64_t mr_count = 0;
	uint64_t m_count = 0;
	uint64_t depolarize_count = 0;
	uint64_t x_error_count = 0;

	for (size_t i = 0; i < instructions.size(); ++i) {
		const Instruction& instr = instructions[i];
		if (instr.op == OpCode::DEPOLARIZE1) {
			++depolarize_count;
			if (!almost_equal(instr.arg, expected_prob)) {
				throw std::runtime_error("DEPOLARIZE1 argument does not match requested probability.");
			}
		}
		if (instr.op == OpCode::X_ERROR) {
			++x_error_count;
			if (!almost_equal(instr.arg, expected_prob)) {
				throw std::runtime_error("X_ERROR argument does not match requested probability.");
			}
		}
		if (instr.op == OpCode::H) {
			++h_count;
			size_t depolarize_index = i + 1;
			if (single_qubit_errors) {
				if (depolarize_index >= instructions.size() ||
					instructions[depolarize_index].op != OpCode::ERASE) {
					throw std::runtime_error(
						"Expected ERASE immediately after H when single_qubit_errors=true.");
				}
				++depolarize_index;
			}
			if (depolarize_index >= instructions.size() ||
				instructions[depolarize_index].op != OpCode::DEPOLARIZE1) {
				throw std::runtime_error("Expected DEPOLARIZE1 after H-eraasure sequence.");
			}
			if (instructions[depolarize_index].targets != instr.targets) {
				throw std::runtime_error("DEPOLARIZE1 targets after H do not match H targets.");
			}
		}
		if (instr.op == OpCode::CX) {
			++cx_count;
			if (i + 2 >= instructions.size()) {
				throw std::runtime_error("CX should be followed by ERASE2*/DEPOLARIZE1.");
			}
			const Instruction& erase = instructions[i + 1];
			if (erase.op != OpCode::ERASE2 && erase.op != OpCode::ERASE2_ANY) {
				throw std::runtime_error("Expected ERASE2/ERASE2_ANY immediately after CX.");
			}
			const Instruction& depolarize = instructions[i + 2];
			if (depolarize.op != OpCode::DEPOLARIZE1) {
				throw std::runtime_error("Expected DEPOLARIZE1 after CX-eraasure sequence.");
			}
			if (depolarize.targets != instr.targets) {
				throw std::runtime_error("DEPOLARIZE1 targets after CX do not match CX targets.");
			}
		}
		if (instr.op == OpCode::ECR) {
			++ecr_count;
			if (i + 1 >= instructions.size() || instructions[i + 1].op != OpCode::DEPOLARIZE1) {
				throw std::runtime_error("Expected DEPOLARIZE1 immediately after ECR.");
			}
			if (instructions[i + 1].targets != instr.targets) {
				throw std::runtime_error("DEPOLARIZE1 targets after ECR do not match ECR targets.");
			}
		}
		if (instr.op == OpCode::MR) {
			++mr_count;
			if (i == 0 || instructions[i - 1].op != OpCode::X_ERROR) {
				throw std::runtime_error("Expected X_ERROR immediately before MR.");
			}
			if (instructions[i - 1].targets != instr.targets) {
				throw std::runtime_error("Pre-MR X_ERROR targets do not match MR targets.");
			}
		}
		if (instr.op == OpCode::M) {
			++m_count;
			if (i == 0 || instructions[i - 1].op != OpCode::X_ERROR) {
				throw std::runtime_error("Expected X_ERROR immediately before final M.");
			}
			if (instructions[i - 1].targets != instr.targets) {
				throw std::runtime_error("Pre-M X_ERROR targets do not match M targets.");
			}
		}
	}

	if (depolarize_count != h_count + cx_count + ecr_count) {
		throw std::runtime_error(
			"Number of DEPOLARIZE1 instructions does not equal H + CX + ECR count.");
	}
	if (x_error_count != mr_count + m_count) {
		throw std::runtime_error(
			"Number of X_ERROR instructions does not equal MR + M count.");
	}
}

}  // namespace

int main() {
	using namespace qerasure::gen;  // NOLINT

	constexpr uint32_t kDistance = 3;
	constexpr uint32_t kRounds = 2;
	constexpr double kErasureProb = 0.01;
	constexpr double kPostCliffordProb = 0.037;

	SurfaceCodeRotated generator(kDistance);

	const auto no_pauli_circuit = generator.build_circuit(
		kRounds,
		kErasureProb,
		/*erasable_qubits=*/"ALL",
		/*reset_failure_prob=*/0.0,
		/*ecr_after_each_step=*/false,
		/*single_qubit_errors=*/true,
		/*post_clifford_pauli_prob=*/0.0);
	verify_no_depolarize(no_pauli_circuit);

	const auto with_pauli_single_qubit = generator.build_circuit(
		kRounds,
		kErasureProb,
		/*erasable_qubits=*/"ALL",
		/*reset_failure_prob=*/0.0,
		/*ecr_after_each_step=*/false,
		/*single_qubit_errors=*/true,
		/*post_clifford_pauli_prob=*/kPostCliffordProb);
	verify_post_clifford_layout(
		with_pauli_single_qubit, /*single_qubit_errors=*/true, kPostCliffordProb);

	const auto with_pauli_two_qubit_only = generator.build_circuit(
		kRounds,
		kErasureProb,
		/*erasable_qubits=*/"ALL",
		/*reset_failure_prob=*/0.0,
		/*ecr_after_each_step=*/false,
		/*single_qubit_errors=*/false,
		/*post_clifford_pauli_prob=*/kPostCliffordProb);
	verify_post_clifford_layout(
		with_pauli_two_qubit_only, /*single_qubit_errors=*/false, kPostCliffordProb);

	std::cout << "surf_post_clifford_pauli_test\n";
	std::cout << "status: post-Clifford DEPOLARIZE1 and pre-measurement X_ERROR placement verified\n";
	return 0;
}
