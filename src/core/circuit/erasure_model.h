# pragma once

#include <cstdint>

#include "core/model/pauli_channel.h"

// Two qubit gate spread model
struct TQGSpreadModel {
    PauliChannel control_spread; // how errors spread if erased qubit is control
    PauliChannel target_spread; // how errors spread if erased qubit is target

    TQGSpreadModel(const PauliChannel& control_spread = {}, const PauliChannel& target_spread = {})
        : control_spread(control_spread), target_spread(target_spread) {}
};

struct ErasureModel {
    uint32_t max_persistence;

    PauliChannel onset;
    PauliChannel reset;
    TQGSpreadModel spread;

    ErasureModel(uint32_t max_persistence = 0, 
                 const PauliChannel& onset = {}, 
                 const PauliChannel& reset = {}, 
                 const TQGSpreadModel& spread = {})
        : max_persistence(max_persistence), onset(onset), reset(reset), spread(spread) {}

    ErasureModel(uint32_t max_persistence = 0, 
                 const PauliChannel& onset = {}, 
                 const PauliChannel& reset = {}, 
                 const PauliChannel& control_spread = {}, 
                 const PauliChannel& target_spread = {})
        : max_persistence(max_persistence), onset(onset), reset(reset), spread(TQGSpreadModel{control_spread, target_spread}) {}

    ErasureModel(uint32_t max_persistence = 0,
                 const PauliChannel& onset = {}, 
                 const PauliChannel& reset = {}, 
                 const PauliChannel& cx_spread = {})
        : max_persistence(max_persistence), onset(onset), reset(reset), spread(TQGSpreadModel{cx_spread, cx_spread}) {}

};
