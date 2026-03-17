#pragma once

#include <cstdint>
#include <string>

#include "core/circuit/circuit.h"
#include "core/code/rotated_surface_code.h"

namespace qerasure::gen {

class SurfaceCodeRotated {
 public:
  explicit SurfaceCodeRotated(uint32_t distance);

  // Controls how many syndrome rounds elapse between end-of-round erasure checks.
  // A value of 1 preserves the current behavior of checking every round, while
  // larger values check every Nth round and always on the final round.
  circuit::ErasureCircuit build_circuit(uint32_t rounds, double erasure_prob,
                                        std::string erasable_qubits = "ALL",
                                        double reset_failure_prob = 0.0,
                                        bool ecr_after_each_step = false,
                                        bool single_qubit_errors = false,
                                        double post_clifford_pauli_prob = 0.0,
                                        uint32_t rounds_per_check = 1);

 private:
  RotatedSurfaceCode code_;
};

}  // namespace qerasure::gen
