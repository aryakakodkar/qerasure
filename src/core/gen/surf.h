#include <cstdint>

#include "core/circuit/circuit.h"
#include "core/circuit/erasure_model.h"

namespace qerasure::gen {

class SurfaceCodeRotated {
    public: 
        SurfaceCodeRotated(uint32_t distance);

        circuit::ErasureCircuit build_circuit(
            uint32_t distance, uint32_t rounds, 
            double erasure_prob,
            circuit::ErasureModel erasure_model,
            std::string erasable_qubits = "ALL");

    private:
        uint32_t num_qubits_;
        
        uint32_t x_anc_offset_;
        uint32_t z_anc_offset_;
};

} // namespace qerasure::gen
