#include "qerasure/core/lowering/lowering.h"

namespace qerasure {

LoweringParams::LoweringParams(const LoweredErrorParams& reset, const LoweredErrorParams& ancillas) {
    reset_params_ = reset;
    x_ancilla_params_ = {ancillas, ancillas};
    z_ancilla_params_ = {ancillas, ancillas};
}

LoweringParams::LoweringParams(const LoweredErrorParams& reset, const LoweredErrorParams& x_ancillas,
                    const LoweredErrorParams& z_ancillas) {
    reset_params_ = reset;
    x_ancilla_params_ = {x_ancillas, x_ancillas};
    z_ancilla_params_ = {z_ancillas, z_ancillas};
}

LoweringParams::LoweringParams(const LoweredErrorParams& reset, const std::pair<LoweredErrorParams, LoweredErrorParams>& x_ancillas,
                    const std::pair<LoweredErrorParams, LoweredErrorParams>& z_ancillas) {
    reset_params_ = reset;
    x_ancilla_params_ = x_ancillas;
    z_ancilla_params_ = z_ancillas;
}

LoweringResult Lowerer::lower(const ErasureSimResult& sim_result) {
    LoweringResult result;

    result.sparse_cliffords.resize(sim_result.sparse_erasures.size()); // clear inefficiency as these are both the same size
    result.clifford_timestep_offsets.resize(sim_result.erasure_timestep_offsets.size());

    std::vector<std::size_t> states(code_.num_qubits(), 0);
    
    for (std::size_t shot = 0; shot < sim_result.sparse_erasures.size(); ++shot) {
        std::size_t index = 0;
        std::size_t num_lowering_events = 0;
    
        const auto& events = sim_result.sparse_erasures[shot];
        const auto& offsets = sim_result.erasure_timestep_offsets[shot];

        for (std::size_t t = 0; t < offsets.size() - 1; ++t) {
            std::size_t offset_t = offsets[t + 1];
            
            for (; index < offset_t; ++index) {
                EventType event_type = events[index].event_type;
                std::size_t qubit_idx = events[index].qubit_idx;
                
                // For now, deal only with reset events
                if (event_type == EventType::RESET) {
                    result.sparse_cliffords[shot].push_back({qubit_idx, params_.reset_params_.error_type});
                    states[qubit_idx] = 0;
                    ++num_lowering_events;
                }
            }
        }
        result.clifford_timestep_offsets[shot].push_back(num_lowering_events);
    }
    return result;
}

} // namespace qerasure
