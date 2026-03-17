#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "core/circuit/instruction.h"
#include "core/gen/surf.h"

namespace {

using qerasure::circuit::ErasureCircuit;
using qerasure::circuit::Instruction;
using qerasure::circuit::OpCode;
using qerasure::gen::SurfaceCodeRotated;

std::size_t count_ops(const ErasureCircuit& circuit, OpCode op) {
	std::size_t count = 0;
	for (const Instruction& instruction : circuit.instructions()) {
		if (instruction.op == op) {
			++count;
		}
	}
	return count;
}

void expect(bool condition, const char* message) {
	if (!condition) {
		throw std::runtime_error(message);
	}
}

void check_first_syndrome_detectors_have_single_lookback(const ErasureCircuit& circuit) {
	bool found_first_mr = false;
	bool found_detector = false;
	for (const Instruction& instruction : circuit.instructions()) {
		if (!found_first_mr) {
			if (instruction.op == OpCode::MR) {
				found_first_mr = true;
			}
			continue;
		}
		if (instruction.op != OpCode::DETECTOR) {
			if (found_detector) {
				return;
			}
			continue;
		}
		found_detector = true;
		expect(instruction.targets.size() == 1,
			   "first checked round should not reference a nonexistent previous syndrome");
	}
	throw std::runtime_error("failed to locate first syndrome-detector block");
}

}  // namespace

int main() {
	constexpr uint32_t kDistance = 3;
	constexpr uint32_t kRounds = 5;
	constexpr double kErasureProb = 0.01;

	SurfaceCodeRotated generator(kDistance);

	const ErasureCircuit every_round =
		generator.build_circuit(kRounds, kErasureProb, "ALL", 0.0, false, false, 0.0, 1);
	expect(count_ops(every_round, OpCode::ECR) == 5,
		   "rounds_per_check=1 should emit an end-of-round erasure check every round");
	expect(count_ops(every_round, OpCode::MR) == 5,
		   "rounds_per_check=1 should measure ancillas every round");

	const ErasureCircuit every_other_round =
		generator.build_circuit(kRounds, kErasureProb, "ALL", 0.0, false, false, 0.0, 2);
	expect(count_ops(every_other_round, OpCode::ECR) == 3,
		   "rounds_per_check=2 should check rounds 2, 4, and the final round");
	expect(count_ops(every_other_round, OpCode::MR) == 5,
		   "rounds_per_check should not change the syndrome-measurement cadence");

	const ErasureCircuit every_third_round =
		generator.build_circuit(kRounds, kErasureProb, "ALL", 0.0, false, false, 0.0, 3);
	expect(count_ops(every_third_round, OpCode::ECR) == 2,
		   "rounds_per_check=3 should check round 3 and the final round");
	expect(count_ops(every_third_round, OpCode::MR) == 5,
		   "syndrome measurements should still happen every round");
	check_first_syndrome_detectors_have_single_lookback(every_third_round);

	bool threw = false;
	try {
		(void)generator.build_circuit(kRounds, kErasureProb, "ALL", 0.0, false, false, 0.0, 0);
	} catch (const std::invalid_argument&) {
		threw = true;
	}
	expect(threw, "rounds_per_check=0 should be rejected");

	return 0;
}
