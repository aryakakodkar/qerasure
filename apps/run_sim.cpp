#include "qerasure/code/code.h"
#include "qerasure/noise/noise.h"
#include <iostream>

int main() {
    std::cout << "Creating a RotatedSurfaceCode...\n\n";
    
    std::size_t distance = 3;
    
    RotatedSurfaceCode code(distance);

    std::cout << "Creating a new noise model...\n\n";

    NoiseParams noise;
    noise.set("p_erasure_check_error", 0.05);
    noise.set("p_two_qubit_erasure", 0.01);

    std::cout << "Noise model parameters:\n";
    std::cout << "p_erasure_check_error: " << noise.get("p_erasure_check_error") << "\n";
    std::cout << "p_two_qubit_erasure: " << noise.get("p_two_qubit_erasure") << "\n";
    std::cout << "p_single_qubit_depolarize: " << noise.get("p_single_qubit_depolarize") << "\n";

    return 0;
}
