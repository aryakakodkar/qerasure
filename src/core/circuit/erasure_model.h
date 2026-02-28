# pragma once

#include <cstdint>
#include <stdexcept>

#include "core/model/pauli_channel.h"

namespace qerasure::circuit {

// Two qubit gate spread model
struct TQGSpreadModel {
    PauliChannel control_spread; // how errors spread if erased qubit is control
    PauliChannel target_spread; // how errors spread if erased qubit is target

    TQGSpreadModel(const PauliChannel& control_spread = {}, const PauliChannel& target_spread = {})
        : control_spread(control_spread), target_spread(target_spread) {}
};

inline void validate_max_persistence(uint32_t max_persistence) {
    if (max_persistence == 0) {
        throw std::invalid_argument("Max persistence must be greater than 0.");
    }
}

struct ErasureModel {
    uint32_t max_persistence;

    PauliChannel onset;
    PauliChannel reset;
    TQGSpreadModel spread;

    double check_false_negative_prob = 0.0;
    double check_false_positive_prob = 0.0;

    ErasureModel(uint32_t max_persistence = 0, 
                 const PauliChannel& onset = {}, 
                 const PauliChannel& reset = {}, 
                 const TQGSpreadModel& spread = {})
        : max_persistence(max_persistence), onset(onset), reset(reset), spread(spread) {
            validate_max_persistence(max_persistence);
        }

    ErasureModel(uint32_t max_persistence = 0, 
                 const PauliChannel& onset = {}, 
                 const PauliChannel& reset = {},
                 const PauliChannel& control_spread = {}, 
                 const PauliChannel& target_spread = {})
        : max_persistence(max_persistence), onset(onset), reset(reset), spread(TQGSpreadModel{control_spread, target_spread}) {
            validate_max_persistence(max_persistence);
        }

    ErasureModel(uint32_t max_persistence = UINT32_MAX,
                 const PauliChannel& onset = {}, 
                 const PauliChannel& reset = {}, 
                 const PauliChannel& cx_spread = {})
        : max_persistence(max_persistence), onset(onset), reset(reset), spread(TQGSpreadModel{cx_spread, cx_spread}) {
            validate_max_persistence(max_persistence);
        }

};

}  // namespace qerasure::circuit
