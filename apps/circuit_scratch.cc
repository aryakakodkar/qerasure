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
  circuit.safe_append("CX", {0, 1});
  circuit.safe_append("EC", {0}, 0.05);
  circuit.safe_append("COND_ER", {1, 0}, 0.05);

  // Simple erasure model for testing - will need to be nicer for real use cases.
  ErasureModel model(2, 
                     PauliChannel(0.1, 0.05, 0.0), 
                     PauliChannel(0.01, 0.01, 0.01), 
                     PauliChannel(0.05, 0.02, 0.01), 
                     PauliChannel(0.02, 0.01, 0.01));

  model.check_false_negative_prob = 0.05;
  model.check_false_positive_prob = 0.02;

  CompiledErasureProgram compiled_program = CompiledErasureProgram(circuit, model);
  compiled_program.print_summary();
}
