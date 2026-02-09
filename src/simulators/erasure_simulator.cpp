#include "qerasure/simulators/erasure_simulator.h"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

// Constructor for the erasure simulator
ErasureSimulator::ErasureSimulator(const ErasureSimParams& params) 
    : params_(params), gen_(std::random_device{}()), dist_(0.0, 1.0) {
}

// Simulate the erasure events for the given number of shots and return the result
// Currently only supports RotatedSurfaceCode and two-qubit gate erasure noise
ErasureSimResult ErasureSimulator::simulate() {
    // Check if the number of shots is valid
    std::size_t expected_events_per_shot = (params_.noise.get("p_two_qubit_erasure") * params_.code.num_qubits() * params_.qec_rounds * 8); // params_.qec_rounds * 4 = number of time steps, and then *2 as each erasure requires two events
    if (params_.shots <= 0) {
        throw std::invalid_argument("Number of shots must be greater than 0");
    } else if (expected_events_per_shot * params_.shots > 1e7) { 
        // Arbitrary threshold to prevent excessive memory usage (can be adjusted as needed)
        throw std::invalid_argument("Number of shots is likely to occupy a large proportion of memory");
    }

    // Check if the noise parameters are valid
    if (params_.noise.get("p_single_qubit_erasure") == 0.0 && params_.noise.get("p_two_qubit_erasure") == 0.0) {
        std::cerr << "Warning: No erasure noise parameters provided. Returning all 0s." << std::endl;
    }

    // Get the code object
    const RotatedSurfaceCode& code = params_.code;
    std::size_t num_qubits = code.num_qubits();

    // Noise parameters 
    double p_two = params_.noise.get("p_two_qubit_erasure");
    double p_erasure_check_error = params_.noise.get("p_erasure_check_error");

    // Sparse erasures vector containing the erasure events for each shot and the corresponding timestep offsets for each shot.
    // The elements of sparse_erasures between erasure_timestep_offsets[shot][t] and erasure_timestep_offsets[shot][t+1]
    // correspond to the erasure events that occurred at timestep t of the shot.
    // Returned to user at the end of the simulation
    std::vector<std::vector<ErasureSimEvent>> sparse_erasures;
    std::vector<std::vector<std::size_t>> erasure_timestep_offsets;
    std::size_t num_erasure_events = 0;

    // Optional internal timing (set QERASURE_REPORT_TIMING=1 to enable)
    const char* report_timing_env = std::getenv("QERASURE_REPORT_TIMING");
    const bool report_timing = (report_timing_env != nullptr && report_timing_env[0] == '1');

    using Clock = std::chrono::high_resolution_clock;
    using Nanoseconds = std::chrono::nanoseconds;
    std::chrono::nanoseconds t_setup{0};
    std::chrono::nanoseconds t_erasure_check_loop{0};
    std::chrono::nanoseconds t_evolution_loop{0};
    std::chrono::nanoseconds t_timestep_offsets{0};

    // Cache partner_map pointer outside shot loop
    const std::size_t* partner_map_ptr = code.partner_map().data();
    
    // Conservative estimate: expect ~15% of max possible events per shot
    std::size_t estimated_events_per_shot = (num_qubits * params_.qec_rounds * 15) / 100;
    std::size_t num_timesteps = params_.qec_rounds * 4 + 1;

    // Pre-allocate outer vectors to avoid repeated push_back + copy
    sparse_erasures.resize(params_.shots);
    erasure_timestep_offsets.resize(params_.shots);

    // Reusable current_state buffer (uint8_t for cache-friendliness)
    std::vector<uint8_t> current_state(num_qubits, 0);

    for (std::size_t shot = 0; shot < params_.shots; shot++) {
        auto t0 = Clock::now();
        num_erasure_events = 0; // Reset number of erasure events for the current shot

        sparse_erasures[shot].clear();
        sparse_erasures[shot].reserve(estimated_events_per_shot);
        erasure_timestep_offsets[shot].assign(num_timesteps + 1, 0);

        // Reset current state to all zeros
        std::fill(current_state.begin(), current_state.end(), 0);
        
        double random_val = 0.0;
        if (report_timing) t_setup += std::chrono::duration_cast<Nanoseconds>(Clock::now() - t0);

        for (std::size_t round = 0; round < params_.qec_rounds; round++) {
            for (std::size_t step = 0; step < 4; ++step) {
                std::size_t offset = step * num_qubits;
                
                // Perform evolution and apply two-qubit erasure noise
                auto t2 = Clock::now();
                for (std::size_t qubit = 0; qubit < num_qubits; qubit++) {
                    if (current_state[qubit] == 0 && partner_map_ptr[offset + qubit] != NO_PARTNER) {
                        random_val = dist_(gen_);
                        if (random_val < p_two) {
                            current_state[qubit] = 1;
                            sparse_erasures[shot].push_back({qubit, EventType::ERASURE});
                            num_erasure_events++;
                        }
                    }
                }
                auto t_after_evolution = Clock::now();
                
                if (report_timing) t_evolution_loop += std::chrono::duration_cast<Nanoseconds>(t_after_evolution - t2);
                erasure_timestep_offsets[shot][round * 4 + step + 1] = num_erasure_events;
                if (report_timing) t_timestep_offsets += std::chrono::duration_cast<Nanoseconds>(Clock::now() - t_after_evolution);

                // Perform erasure check and reset
                if (step == 3) {
                    auto t1 = Clock::now();
                    for (std::size_t qubit = 0; qubit < num_qubits; qubit++) {
                        random_val = dist_(gen_);
                        if (random_val < p_erasure_check_error) {
                            sparse_erasures[shot].push_back({qubit, EventType::CHECK_ERROR});
                            num_erasure_events++;
                        } else {
                            if (current_state[qubit] == 1) {
                                sparse_erasures[shot].push_back({qubit, EventType::RESET});
                                current_state[qubit] = 0;
                                num_erasure_events++;
                            }
                        }
                    }
                    if (report_timing) t_erasure_check_loop += std::chrono::duration_cast<Nanoseconds>(Clock::now() - t1);
                }
            }
        }
        erasure_timestep_offsets[shot][num_timesteps] = num_erasure_events; // Final offset for total events in the shot
    }

    if (report_timing) {
        auto total_ns = t_setup.count() + t_erasure_check_loop.count() + t_evolution_loop.count() + t_timestep_offsets.count();
        auto pct = [total_ns](std::chrono::nanoseconds t) {
            return total_ns > 0 ? (100.0 * t.count() / total_ns) : 0.0;
        };
        std::cerr << "=== simulate() timing breakdown ===\n"
                  << "  per-shot setup (allocations, push_back): " << (t_setup.count() / 1000) << " us  (" << pct(t_setup) << "%)\n"
                  << "  erasure check loop (step==3, RNG + reset): " << (t_erasure_check_loop.count() / 1000) << " us  (" << pct(t_erasure_check_loop) << "%)\n"
                  << "  evolution loop (partner_map + two-qubit): " << (t_evolution_loop.count() / 1000) << " us  (" << pct(t_evolution_loop) << "%)\n"
                  << "  timestep_offsets write:                  " << (t_timestep_offsets.count() / 1000) << " us  (" << pct(t_timestep_offsets) << "%)\n"
                  << "  total (inside simulate):                 " << (total_ns / 1000) << " us\n";
    }

    return {sparse_erasures, erasure_timestep_offsets};
}
