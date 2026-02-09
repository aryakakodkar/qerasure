#include "qerasure/simulators/erasure_simulator.h"
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
    if (params_.shots <= 0) {
        throw std::invalid_argument("Number of shots must be greater than 0");
    } else if (params_.shots > 1000) {
        throw std::invalid_argument("Number of shots requires too much memory. Use a number less than 1000.");
    }

    // Check if the noise parameters are valid
    if (params_.noise.get("p_single_qubit_erasure") == 0.0 && params_.noise.get("p_two_qubit_erasure") == 0.0) {
        std::cout << "Warning: No erasure noise parameters provided. Returning all 0s." << std::endl;
    }

    // Get the code object
    const RotatedSurfaceCode& code = params_.code;
    std::size_t num_qubits = code.num_qubits();

    // Noise parameters 
    double p_two = params_.noise.get("p_two_qubit_erasure");
    double p_erasure_check_error = params_.noise.get("p_erasure_check_error");

    // Sparse erasures vector containing the erasure events for each shot
    // Returned to user at the end of the simulation
    std::vector<std::vector<SimEvent>> sparse_erasures;
    std::vector<std::vector<std::size_t>> erasure_timestep_offsets;
    std::size_t num_erasure_events = 0;

    std::vector<std::size_t> DEFAULT_STATE(num_qubits, 0); // Default initial state for all shots

    for (std::size_t shot = 0; shot < params_.shots; shot++) {
        std::vector<std::vector<std::size_t>> dense_erasures(
            num_qubits, 
            std::vector<std::size_t>(num_qubits * params_.qec_rounds * 4 + 1, 0)
        ); // dense erasure vector containing the erasure events for each round of QEC

        num_erasure_events = 0; // Reset number of erasure events for the current shot

        sparse_erasures.push_back(std::vector<SimEvent>());
        erasure_timestep_offsets.push_back(std::vector<std::size_t>(params_.qec_rounds * 4 + 1, 0));

        std::vector<std::size_t> current_state = DEFAULT_STATE; // Current state of the qubits for the current shot
        
        double random_val = 0.0;

        for (std::size_t round = 0; round < params_.qec_rounds; round++) {
            for (std::size_t step = 0; step < 4; step++) {
                // Check for erasures at the start of each round
                if (step == 0) {
                    for (std::size_t qubit = 0; qubit < num_qubits; qubit++) {
                        random_val = dist_(gen_);
                        if (random_val < p_erasure_check_error) {
                            // Erasure check error occurred
                            sparse_erasures[shot].push_back({qubit, EventType::CHECK_ERROR}); // Add check error event to sparse erasures vector
                            num_erasure_events++;
                        } else {
                            // Erasure check error did not occur
                            if (current_state[qubit] == 1) {
                                sparse_erasures[shot].push_back({qubit, EventType::RESET}); // Add reset event to sparse erasures vector
                                current_state[qubit] = 0;
                                num_erasure_events++;
                            }
                        }
                }
                
                // Simulate evolution of erasure state for all qubits in this step
                for (std::size_t qubit = 0; qubit < num_qubits; qubit++) {
                    if (current_state[qubit] == 1 || code.partner_map()[step * num_qubits + qubit] == NO_PARTNER) {
                        // If qubit is erased or has no partner, leave as is
                        dense_erasures[qubit][round * 4 + step] = dense_erasures[qubit][round * 4 + step - 1];
                    } else if (current_state[qubit] == 0) {
                        // If qubit is not currently erased and is involved in a gate, run Markovian sampling for two-qubit erasure
                        random_val = dist_(gen_);
                        if (random_val < p_two) {
                            // Erasure occurred after two-qubit gate
                            current_state[qubit] = 1;
                            dense_erasures[qubit][round * 4 + step] = 1;
                            sparse_erasures[shot].push_back({qubit, EventType::ERASURE}); // Add erasure event to sparse erasures vector
                            num_erasure_events++;
                        }
                    }
                }
                erasure_timestep_offsets[shot][round * 4 + step + 1] = num_erasure_events;
            }
        }
    }

    return {sparse_erasures, erasure_timestep_offsets};
}