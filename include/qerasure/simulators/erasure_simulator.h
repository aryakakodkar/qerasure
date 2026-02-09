#include <qerasure/code/code.h>
#include <qerasure/noise/noise.h>
#include <vector>
#include <random>
#include <stdexcept>

#ifndef ERASURE_SIMULATOR
#define ERASURE_SIMULATOR

// Struct to hold parameters for the erasure simulator
struct ErasureSimParams {
    RotatedSurfaceCode code;
    NoiseParams noise;
    std::size_t qec_rounds; // number of rounds of QEC to simulate
    std::size_t shots; // number of shots to simulate

    // Constructor with validation
    ErasureSimParams(RotatedSurfaceCode code_, NoiseParams noise_, std::size_t qec_rounds_, std::size_t shots_)
        : code(std::move(code_)), noise(std::move(noise_)), qec_rounds(qec_rounds_), shots(shots_) {
        if (shots <= 0) {
            throw std::invalid_argument("Number of shots must be greater than 0");
        }
        if (qec_rounds <= 0) {
            throw std::invalid_argument("Number of QEC rounds must be greater than 0");
        }
    }
};

// Possible event types for sparse storage of spacetime erasure events
enum class EventType : std::uint8_t {
    ERASURE = 0,
    RESET = 1,
    CHECK_ERROR = 2
};

// Struct to hold a spacetime erasure event
struct ErasureSimEvent {
    std::size_t qubit_idx;
    EventType event_type;
};

// Struct to hold the result of an erasure simulation
// The elements of sparse_erasures between erasure_timestep_offsets[shot][t] and
// erasure_timestep_offsets[shot][t+1] correspond to the erasure events that occurred at timestep t.
struct ErasureSimResult {
    std::vector<std::vector<ErasureSimEvent>> sparse_erasures;
    std::vector<std::vector<std::size_t>> erasure_timestep_offsets;

    ErasureSimResult() noexcept = default;
    ErasureSimResult(ErasureSimResult&&) noexcept = default;
    ErasureSimResult& operator=(ErasureSimResult&&) noexcept = default;
    ErasureSimResult(const ErasureSimResult&) = default;
    ErasureSimResult& operator=(const ErasureSimResult&) = default;
};

// Class to simulate erasures in a QEC code.
// Currently only supports rotated surface codes.
class ErasureSimulator {
    public:
        explicit ErasureSimulator(const ErasureSimParams& params);
        ErasureSimResult simulate();

    private:
        ErasureSimParams params_;
        std::mt19937 gen_;
        std::uniform_real_distribution<double> dist_;
};

#endif // ERASURE_SIMULATOR
