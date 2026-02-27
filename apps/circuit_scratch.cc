#include "circuit.h"

#include <cstdint>
#include <vector>
#include <iostream>

int main() {
  ErasureCircuit circuit;
  const std::vector<uint32_t> targets = {0, 1};
  circuit.append(OpCode::CX, targets);
  circuit.safe_append("CX", {1}, 0.01);
  return 0;
}
