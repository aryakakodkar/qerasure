#include "qerasure/code/code.h"
#include <vector>
#include <iostream>

// RotatedSurfaceCode constructor initializes the code with the specified distance and builds the lattice and stabilizers
RotatedSurfaceCode::RotatedSurfaceCode(std::size_t distance) : distance_(distance) {
    if (distance < 3 || distance % 2 == 0) {
        throw std::invalid_argument("Distance must be an odd integer greater than or equal to 3");
    }
    build();
}

// Builds the entire structure of the rotated surface code by first constructing the lattice of qubits
// and then defining the stabilizers and corresponding gates for syndrome extraction.
void RotatedSurfaceCode::build() {
    build_lattice();
    build_stabilizers();
}

// Builds the lattice structure for the rotated surface code.
// Initializes the positions and indices of all data and ancilla qubits,
// populates the coordinate-to-index and index-to-coordinate mappings,
// and sets up the offsets for X and Z ancilla qubit indices based on the code distance.
void RotatedSurfaceCode::build_lattice() {
    // Build data qubit lattice: data qubits are located at (x, y) where x, y are odd integers from 1 to 2*distance-1
    QubitIndex index = 0;
    for (QubitIndex x = 1; x < 2 * distance_ + 1; x += 2) {
        for (QubitIndex y = 1; y < 2 * distance_ + 1; y += 2) {
            index_to_coord_[index].first = x;
            index_to_coord_[index].second = y;
            coord_to_index_[{x, y}] = index++;
        }
    }

    // Build x-ancilla lattice: x-ancilla qubits are located at (x, y) where x, y are even and x + y is congruent to 2 mod 4
    x_anc_offset_ = index;
    for (QubitIndex x = 2; x < 2 * distance_; x += 4) {
        for (QubitIndex y = 0; y < 2 * distance_ + 2; y += 2) {
            index_to_coord_[index].first = x + 2 - (y % 4);
            index_to_coord_[index].second = y;
            coord_to_index_[{x + 2 - (y % 4), y}] = index++;
        }
    }

    // Build z-ancilla lattice: z-ancilla qubits are located at (x, y) where x, y are even and x + y is congruent to 0 mod 4
    z_anc_offset_ = index;
    for (QubitIndex x = 0; x < 2 * distance_ + 2; x += 2) {
        for (QubitIndex y = 2; y < 2 * distance_; y += 4) {
            index_to_coord_[index].first = x;
            index_to_coord_[index].second = y + (x % 4);
            coord_to_index_[{x, y + (x % 4)}] = index++;
        }
    }

    num_qubits_ = index;
}

// Builds the stabilizers for the rotated surface code
// Populates the list of gates which contains pairs of qubits involved in CNOT operations in the form (control, target)
// Also populates the step pointers vector which delimits the different gate steps in the syndrome extraction schedule
void RotatedSurfaceCode::build_stabilizers() {
    // Number of gates: 2 gates per corner qubit, 3 per boundary qubit, 4 per bulk qubit
    QubitIndex gates_per_step = (8 + 12 * (distance_ - 2) + 4 * (distance_ - 2) * (distance_ - 2))/4;
    gates_.resize(gates_per_step * 4); // 4 steps in the schedule

    partner_map_.resize(4 * num_qubits_, NO_PARTNER); // 4 steps, each with num_qubits_ entries

    // TODO: Is the step_iters_ vector necessary? Could I just pass gates_per_step and do the calculations from there?
    for (std::size_t step = 0; step < 4; step++) {
        step_iters_[step] = gates_.begin() + step * gates_per_step; // Find the starting point for each step in the gates vector
    }

    for (size_t step = 0; step < 4; step++) {
        std::pair<QubitIndex, QubitIndex> x_gate_direction;
        std::pair<QubitIndex, QubitIndex> z_gate_direction;

        switch (step) {
            case 0:
                x_gate_direction = {-1, 1}; // NW
                z_gate_direction = {-1, 1}; // NW
                break;
            case 1:
                x_gate_direction = {1, 1}; // NE
                z_gate_direction = {-1, -1}; // SW
                break;
            case 2:
                x_gate_direction = {-1, -1}; // SW
                z_gate_direction = {1, 1}; // NE
                break;
            case 3:
                x_gate_direction = {1, -1}; // SE
                z_gate_direction = {1, -1}; // SE
                break;
        }

        QubitIndex gate_idx = 0;
        std::unordered_map<std::pair<QubitIndex, QubitIndex>, QubitIndex, PairHash>::iterator current_ptr;
        for (QubitIndex idx = x_anc_offset_; idx < z_anc_offset_; idx++) {
            current_ptr = coord_to_index_.find({index_to_coord_[idx].first + x_gate_direction.first, index_to_coord_[idx].second + x_gate_direction.second});
            if (current_ptr != coord_to_index_.end()) {
                *(step_iters_[step] + gate_idx++) = {idx, current_ptr->second}; // CNOT from x-ancilla to data qubit
                partner_map_[step * num_qubits_ + idx] = current_ptr->second;
                partner_map_[step * num_qubits_ + current_ptr->second] = idx;
            }
        }

        for (QubitIndex idx = z_anc_offset_; idx < num_qubits_; idx++) { 
            current_ptr = coord_to_index_.find({index_to_coord_[idx].first + z_gate_direction.first, index_to_coord_[idx].second + z_gate_direction.second});
            if (current_ptr != coord_to_index_.end()) {
                *(step_iters_[step] + gate_idx++) = {current_ptr->second, idx}; // CNOT from data qubit to z-ancilla
                partner_map_[step * num_qubits_ + current_ptr->second] = idx;
                partner_map_[step * num_qubits_ + idx] = current_ptr->second;
            }
        }
    }
}
