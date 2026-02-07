#include "qerasure/code/code.h"
#include <iostream>

int main() {
    std::cout << "Creating a RotatedSurfaceCode...\n\n";
    
    // Create a distance-3 surface code
    std::size_t distance = 3;
    RotatedSurfaceCode code(distance);
    
    // Print some basic information
    std::cout << "\n=== Code Properties ===" << '\n';
    std::cout << "Distance: " << code.distance() << '\n';
    std::cout << "Number of qubits: " << code.num_qubits() << '\n';
    std::cout << "Number of stabilizers: " << code.stabilizers().size() << '\n';
    
    // Try different distances
    std::cout << "\n=== Trying different distances ===" << '\n';
    for (std::size_t d : {3, 5, 7}) {
        RotatedSurfaceCode test_code(d);
        std::cout << "d=" << d << ": " << test_code.num_qubits() << " qubits" << '\n';
    }
    
    return 0;
}
