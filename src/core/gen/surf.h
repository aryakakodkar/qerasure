#pragma once

#include <cstdint>
#include <string>

#include "core/circuit/circuit.h"
#include "qerasure/core/code/rotated_surface_code.h"

namespace qerasure::gen {

class SurfaceCodeRotated {
 public:
  explicit SurfaceCodeRotated(uint32_t distance);

  // TODO: Add support for custom erasure check and reset frequencies.
  circuit::ErasureCircuit build_circuit(uint32_t rounds, double erasure_prob,
                                        std::string erasable_qubits = "ALL");

 private:
  RotatedSurfaceCode code_;
};

}  // namespace qerasure::gen
