#include <cstddef>
#include <iostream>
#include <stdexcept>

#include "core/gen/surf.h"

int main() {
  using qerasure::circuit::OpCode;

  qerasure::gen::SurfaceCodeRotated generator(3);
  const qerasure::circuit::ErasureCircuit circuit = generator.build_circuit(
      /*rounds=*/3, /*erasure_prob=*/0.05, /*erasable_qubits=*/"ALL");

  // Print circuit text in Stim-like format for inspection.
  std::cout << "Distance-3, 3-round SurfaceCodeRotated circuit:\n";
  std::cout << circuit << "\n";

  std::size_t h_count = 0;
  std::size_t cx_count = 0;
  std::size_t m_count = 0;
  std::size_t mr_count = 0;
  std::size_t erase2_any_count = 0;
  std::size_t ecr_count = 0;
  std::size_t detector_count = 0;
  std::size_t observable_include_count = 0;
  std::size_t ecr_before_mr_rounds = 0;

  for (const auto& instr : circuit.instructions()) {
    switch (instr.op) {
      case OpCode::H:
        ++h_count;
        break;
      case OpCode::CX:
        ++cx_count;
        break;
      case OpCode::M:
        ++m_count;
        break;
      case OpCode::MR:
        ++mr_count;
        break;
      case OpCode::ERASE2_ANY:
        ++erase2_any_count;
        break;
      case OpCode::ECR:
        ++ecr_count;
        break;
      case OpCode::DETECTOR:
        ++detector_count;
        break;
      case OpCode::OBSERVABLE_INCLUDE:
        ++observable_include_count;
        break;
      default:
        break;
    }
  }

  for (std::size_t i = 0; i + 1 < circuit.instructions().size(); ++i) {
    if (circuit.instructions()[i].op == OpCode::ECR &&
        circuit.instructions()[i + 1].op == OpCode::MR) {
      ++ecr_before_mr_rounds;
    }
  }

  if (h_count != 6) {
    throw std::runtime_error("Expected 6 H layers (2 per round for 3 rounds)");
  }
  if (cx_count != 12) {
    throw std::runtime_error("Expected 12 CX steps (4 per round for 3 rounds)");
  }
  if (erase2_any_count != 12) {
    throw std::runtime_error("Expected 12 ERASE2_ANY steps (4 per round for 3 rounds)");
  }
  if (m_count != 1) {
    throw std::runtime_error("Expected one final data M layer");
  }
  if (mr_count != 3) {
    throw std::runtime_error("Expected 3 MR layers (1 per round)");
  }
  if (ecr_count != 3) {
    throw std::runtime_error("Expected 3 ECR layers (1 per round)");
  }
  if (ecr_before_mr_rounds != 3) {
    throw std::runtime_error("Expected ECR to be immediately before MR in each round");
  }
  if (detector_count != 16) {
    throw std::runtime_error(
        "Expected 16 DETECTOR ops (4 per round for 3 rounds + 4 final readout)");
  }
  if (observable_include_count != 1) {
    throw std::runtime_error("Expected one OBSERVABLE_INCLUDE op");
  }

  return 0;
}
