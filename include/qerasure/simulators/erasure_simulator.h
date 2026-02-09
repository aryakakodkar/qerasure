#include <qerasure/code/code.h>
#include <qerasure/noise/noise.h>
#include <vector>
#include <random>

#ifndef ERASURE_SIMULATOR
#define ERASURE_SIMULATOR

// Struct to hold parameters for the erasure simulator
struct ErasureSimParams {
    RotatedSurfaceCode code;
    NoiseParams noise;
    std::size_t qec_rounds; // number of rounds of QEC to simulate
    std::size_t shots; // number of shots to simulate
};

// Possible event types for sparse storage of spacetime erasure events
enum class EventType {
    ERASURE = 0,
    RESET = 1,
    CHECK_ERROR = 2
};

// Struct to hold a spacetime erasure event
struct SimEvent {
    std::size_t qubit_idx;
    EventType event_type;
};

// Struct to hold the result of an erasure simulation
struct ErasureSimResult {
    std::vector<std::vector<SimEvent>> sparse_erasures;
    std::vector<std::vector<std::size_t>> erasure_timestep_offsets;
};

// Class to simulate erasures in a QEC code.
// I intend to create another class to simulate erasure + circuit, so use that if that's what you're looking for.
// Currently only supports rotated surface codes.
class ErasureSimulator {
    public:
        ErasureSimulator(const ErasureSimParams& params);
        ErasureSimResult simulate();
        ErasureSimResult simulate_single_shot();

    private:
        ErasureSimParams params_;
        mutable std::mt19937 gen_;
        mutable std::uniform_real_distribution<double> dist_;
};

#endif // ERASURE_SIMULATOR