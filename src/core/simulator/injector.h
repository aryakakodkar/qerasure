#pragma once

#include <cstdint>

#include "core/simulator/erasure_sampler.h"
#include "stim/circuit/circuit.h"

namespace qerasure::simulator {

class Injector {
    public:
        Injector();

        stim::Circuit inject(const SampledBatch& batch, uint32_t shot_index);
};

} // namespace qerasure::simulator
