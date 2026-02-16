#pragma once

#include <cstdint>
#include <vector>

#include "qerasure/core/code/rotated_surface_code.h"
#include "qerasure/core/sim/erasure_simulator.h"

namespace qerasure {

enum class PauliError : uint8_t {
    NO_ERROR = 0,
    X_ERROR = 1,
    Z_ERROR = 2,
    Y_ERROR = 3,
    DEPOLARIZE = 4
};

struct LoweredErrorParams {
    PauliError error_type;
    double probability = 0.0;
};

struct LoweredErrorEvent {
    std::size_t qubit_idx;
    PauliError error_type;
};

struct LoweringResult {
    std::vector<std::vector<LoweredErrorEvent>> sparse_cliffords;
    std::vector<std::vector<std::size_t>> clifford_timestep_offsets;
};

struct LoweringParams {
    LoweredErrorParams reset_params_;
    std::pair<LoweredErrorParams, LoweredErrorParams> x_ancilla_params_;
    std::pair<LoweredErrorParams, LoweredErrorParams> z_ancilla_params_;

    LoweringParams(const LoweredErrorParams& reset, const LoweredErrorParams& ancillas) {} // all ancillas share same lowering protocol
    LoweringParams(const LoweredErrorParams& reset, const LoweredErrorParams& x_ancillas,
                    const LoweredErrorParams& z_ancillas) {} // separate lowering protocols for X and Z ancillas
    LoweringParams(const LoweredErrorParams& reset, const std::pair<LoweredErrorParams, LoweredErrorParams>& x_ancillas, 
                    const std::pair<LoweredErrorParams, LoweredErrorParams>& z_ancillas) {} // separate lowering protocol for each ancilla
};

class Lowerer {
    public:
        explicit Lowerer(const RotatedSurfaceCode& code, const LoweringParams& params);

        RotatedSurfaceCode code_;
        LoweringParams params_;

        LoweringResult lower(const ErasureSimResult& sim_result);
};

} // namespace qerasure
