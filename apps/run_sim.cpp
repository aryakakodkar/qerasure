#include "qerasure/code/code.h"
#include "qerasure/noise/noise.h"
#include "qerasure/simulators/erasure_simulator.h"
#include <iostream>

int main() {
    RotatedSurfaceCode code(3); // Create a distance-3 rotated surface code
    NoiseParams noise;

    noise.set("p_two_qubit_erasure", 0.01);
    noise.set("p_erasure_check_error", 0.05);

    ErasureSimParams params = {code, noise, 10, 1000};

    ErasureSimulator simulator(params);
    ErasureSimResult result = simulator.simulate();

    // Find the total memory usage of the result
    std::size_t mem_usage = 0;
    for (const auto& shot : result.sparse_erasures) {
        mem_usage += shot.size() * sizeof(SimEvent);
    }
    mem_usage += result.erasure_timestep_offsets.size() * sizeof(std::vector<std::size_t>);
    for (const auto& timestep_offsets : result.erasure_timestep_offsets) {
        mem_usage += timestep_offsets.size() * sizeof(std::size_t);
    }
    std::cout << "Total memory usage: " << mem_usage << " bytes\n" << std::endl;

    return 0;
}
