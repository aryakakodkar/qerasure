#include "qerasure/code/code.h"
#include "qerasure/noise/noise.h"
#include <iostream>

int main() {
    std::cout << "Creating a RotatedSurfaceCode...\n\n";
    
    std::size_t distance = 3;
    
    RotatedSurfaceCode code(distance);

    std::cout << "Creating a new noise model...\n\n";

    NoiseParams noise = build_noise_model({.p_two_qubit_erasure = 0.01, .p_erasure_check_error = 0.05});

    return 0;
}
