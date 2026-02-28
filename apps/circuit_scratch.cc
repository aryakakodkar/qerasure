#include "circuit.h"
#include "compile.h"
#include "pauli_channel.h"
#include "erasure_model.h"

#include <cstdint>
#include <vector>
#include <iostream>

int main() {

  ErasureCircuit circuit;
  const std::vector<uint32_t> targets = {0, 1};

  circuit.append(OpCode::CX, targets);
  circuit.safe_append("CX", {1, 0}, 0.0);
  circuit.safe_append("ECR", {1, 0}, 0.05);
  circuit.safe_append("ERASE", {0}, 0.1);

  CompiledErasureProgram compiled_program = CompiledErasureProgram(circuit);
  compiled_program.print_summary();
}
