
/*
 * benchmark_erasure_sim.cpp
 *
 * Usage:
 *   ./benchmark_erasure_sim <distance> <shots> [qec_rounds]
 *
 *   <distance>   : Code distance (e.g., 3, 5, 7)
 *   <shots>      : Number of simulation shots
 *   [qec_rounds] : Number of QEC rounds (default: 10)
 *
 * Example:
 *   ./benchmark_erasure_sim 5 1000 10
 *
 * This program runs the erasure simulator with the given parameters and prints timing and event statistics.
 */

#include "qerasure/code/code.h"
#include "qerasure/noise/noise.h"
#include "qerasure/simulators/erasure_simulator.h"
#include <iostream>
#include <chrono>
#include <cstdlib>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <distance> <shots> [qec_rounds]\n";
        return 1;
    }

    std::size_t distance = std::stoull(argv[1]);
    std::size_t shots = std::stoull(argv[2]);
    std::size_t qec_rounds = (argc >= 4) ? std::stoull(argv[3]) : 10;

    RotatedSurfaceCode code(distance);
    NoiseParams noise;
    noise.set("p_two_qubit_erasure", 0.01);
    noise.set("p_erasure_check_error", 0.05);

    ErasureSimParams params = {code, noise, qec_rounds, shots};
    ErasureSimulator simulator(params);

    auto start_time = std::chrono::high_resolution_clock::now();
    ErasureSimResult result = simulator.simulate();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "\n=== Results ===\n"
              << "Total time: " << duration.count() << " ms\n"
              << "Time per shot: " << (duration.count() / static_cast<double>(shots)) << " ms\n"
              << "Shots per second: " << (shots * 1000.0 / duration.count()) << "\n";

    std::size_t mem_usage = 0;
    for (const auto& shot : result.sparse_erasures) {
        mem_usage += shot.size() * sizeof(ErasureSimEvent);
    }
    mem_usage += result.erasure_timestep_offsets.size() * sizeof(std::vector<std::size_t>);
    for (const auto& timestep_offsets : result.erasure_timestep_offsets) {
        mem_usage += timestep_offsets.size() * sizeof(std::size_t);
    }
    std::cout << "Memory usage: " << (mem_usage / 1024.0 / 1024.0) << " MB\n";
    
    return 0;
}
