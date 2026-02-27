# pragma once

#include <stdexcept>

struct PauliChannel {
    double p_x;
    double p_y;
    double p_z;

    PauliChannel(double p_x = 0.0, double p_y = 0.0, double p_z = 0.0)
        : p_x(p_x), p_y(p_y), p_z(p_z) {
            if (p_x < 0.0 || p_y < 0.0 || p_z < 0.0 || p_x + p_y + p_z > 1.0) {
                throw std::invalid_argument("Invalid Pauli channel probabilities.");
            }
        }
};
