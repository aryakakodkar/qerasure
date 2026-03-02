#include <cstdint>
#include <functional>
#include <vector>

#include "stim/circuit/circuit.h"
#include "core/circuit/compile.h"

namespace qerasure::simulator {

class StreamSampler {
    public:
        StreamSampler(const circuit::CompiledErasureProgram& program) : program_(program) {};

        // Callback for shot processing (e.g. sample + decode)
        void sample(uint32_t num_shots,
                    uint32_t seed,
                    std::function<void(const stim::Circuit&, const std::vector<uint8_t>&)> callback);

    private:
        const circuit::CompiledErasureProgram& program_;
};

} // namespace qerasure::simulator
