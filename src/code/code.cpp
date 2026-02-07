#include "qerasure/code/code.h"
#include <vector>
#include <iostream>

RotatedSurfaceCode::RotatedSurfaceCode(std::size_t distance) : distance_(distance) {
    build();
}

void RotatedSurfaceCode::build() {
    build_lattice();
    build_stabilizers();
}

void RotatedSurfaceCode::build_lattice() {
    // Build data qubit lattice
    QubitIndex index = 0;
    for (QubitIndex x = 1; x < 2 * distance_ + 1; x += 2) {
        for (QubitIndex y = 1; y < 2 * distance_ + 1; y += 2) {
            index_to_coord_[index] = {x, y};
            coord_to_index_[{x, y}] = index++;
        }
    }

    // Build x-ancilla lattice
    x_anc_offset_ = index;
    for (QubitIndex x = 2; x < 2 * distance_; x += 4) {
        for (QubitIndex y = 0; y < 2 * distance_ + 2; y += 2) {
            index_to_coord_[index] = {x + 2 - (y % 4), y};
            coord_to_index_[{x + 2 - (y % 4), y}] = index++;
        }
    }

    // Build z-ancilla lattice
    z_anc_offset_ = index;
    for (QubitIndex x = 0; x < 2 * distance_ + 2; x += 2) {
        for (QubitIndex y = 2; y < 2 * distance_; y += 4) {
            index_to_coord_[index] = {x, y + (x % 4)};
            coord_to_index_[{x, y + (x % 4)}] = index++;
        }
    }

    num_qubits_ = index;
}

/*
Gate Schedule:
Step 0: X (NW), Z (NW)
Step 1: X (NE), Z (SW)
Step 2: X (SW), Z (NE)
Step 3: X (SE), Z (SE)
*/

void RotatedSurfaceCode::build_stabilizers() {
    QubitIndex gates_per_step = (8 + 12 * (distance_ - 2) + 4 * (distance_ - 2) * (distance_ - 2))/4; // 8 for corners, 12 for edges, 4 for interior
    gates_.resize(gates_per_step * 4); // 4 steps in the schedule

    step_pointers_ = {gates_.data() + gates_per_step * 0, gates_.data() + gates_per_step * 1, 
        gates_.data() + gates_per_step * 2, gates_.data() + gates_per_step * 3};

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
        for (QubitIndex idx = x_anc_offset_; idx < z_anc_offset_; idx++) {
            if (coord_to_index_.find({index_to_coord_[idx][0] + x_gate_direction.first, index_to_coord_[idx][1] + x_gate_direction.second}) == coord_to_index_.end()) {
                continue; // Skip if the target data qubit is out of bounds
            }
            *(step_pointers_[step] + gate_idx++) = {idx, coord_to_index_[{index_to_coord_[idx][0] + x_gate_direction.first, index_to_coord_[idx][1] + x_gate_direction.second}]};
        }

        for (QubitIndex idx = z_anc_offset_; idx < num_qubits_; idx++) { 
            if (coord_to_index_.find({index_to_coord_[idx][0] + z_gate_direction.first, index_to_coord_[idx][1] + z_gate_direction.second}) == coord_to_index_.end()) {
                continue; // Skip if the target data qubit is out of bounds
            }
            *(step_pointers_[step] + gate_idx++) = {coord_to_index_[{index_to_coord_[idx][0] + z_gate_direction.first, index_to_coord_[idx][1] + z_gate_direction.second}], idx};
        }
    }
}
