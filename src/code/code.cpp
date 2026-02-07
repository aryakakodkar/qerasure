#include "qerasure/code/code.h"
#include <vector>
#include <iostream>

RotatedSurfaceCode::RotatedSurfaceCode(std::size_t distance) : distance_(distance) {
    build();
}

void RotatedSurfaceCode::build() {
    build_lattice();
    // build_stabilizers();
}

void RotatedSurfaceCode::build_lattice() {
    // Build data qubit lattice
    for (std::size_t i = 0; i < 2*distance_; i += 2) {
        std::cout << "Adding a qubit at position " << i << '\n';
    }
}
