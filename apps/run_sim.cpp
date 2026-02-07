#include "qerasure/code/code.h"
#include <iostream>

int main() {
    std::cout << "Creating a RotatedSurfaceCode...\n\n";
    
    std::size_t distance = 3;
    
    RotatedSurfaceCode code(distance);

    const std::vector<std::vector<std::pair<QubitIndex, QubitIndex>>>& gates = code.gates();

    for (std::size_t i = 0; i < 4; i++) {
        std::cout << "Step " << i << ":\n";
        for (const auto& gate : gates[i]) {
            std::cout << "  CNOT between qubits " << gate.first << " and " << gate.second << "\n";
        }
        std::cout << "\n";
    }
    return 0;
}
