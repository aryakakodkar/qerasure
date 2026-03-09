#pragma once

#include <cstdint>
#include <string>

#include "core/circuit/circuit.h"
#include "core/code/rotated_surface_code.h"

namespace qerasure::gen {

class SurfaceCodeRotated {
 public:
  explicit SurfaceCodeRotated(uint32_t distance);

  // TODO: Add support for custom erasure check and reset frequencies.
  circuit::ErasureCircuit build_circuit(uint32_t rounds, double erasure_prob,
                                        std::string erasable_qubits = "ALL",
                                        double reset_failure_prob = 0.0,
                                        bool ecr_after_each_step = false,
                                        bool single_qubit_errors = false);

 private:
  RotatedSurfaceCode code_;
};

}  // namespace qerasure::gen
