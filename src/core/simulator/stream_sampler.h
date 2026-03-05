#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "stim/circuit/circuit.h"
#include "core/circuit/compile.h"

namespace qerasure::simulator {

struct SyndromeSampleBatch {
    uint32_t num_shots = 0;
    uint32_t num_detectors = 0;
    uint32_t num_observables = 0;
    uint32_t num_checks = 0;

    // Row-major flattened buffers:
    // detector_samples[shot * num_detectors + detector_index]
    // observable_flips[shot * num_observables + observable_index]
    // check_flags[shot * num_checks + check_index]
    std::vector<uint8_t> detector_samples;
    std::vector<uint8_t> observable_flips;
    std::vector<uint8_t> check_flags;
};

class StreamSampler {
    public:
        StreamSampler(const circuit::CompiledErasureProgram& program) : program_(program) {};

        // Callback-based shot processing (e.g. sample + decode).
        // In multi-thread mode, callback may run concurrently on multiple threads.
        void sample_with_callback(
            uint32_t num_shots,
            uint32_t seed,
            std::function<void(const stim::Circuit&, const std::vector<uint8_t>&)> callback,
            uint32_t num_threads = 1);

        // Native syndrome sampling path. Builds sampled logical circuits, runs Stim detector
        // sampling internally, and returns detector bits, observable flips, and erasure-check
        // flags as uint8 row-major buffers.
        SyndromeSampleBatch sample_syndromes(
            uint32_t num_shots,
            uint32_t seed,
            uint32_t num_threads = 1);

    private:
        const circuit::CompiledErasureProgram& program_;
};

} // namespace qerasure::simulator
