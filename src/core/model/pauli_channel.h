# pragma once

#include <cstdint>
#include <limits>
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

struct ThresholdedPauliChannel {
    uint64_t p_x_threshold;
    uint64_t p_y_threshold;
    uint64_t p_z_threshold;

    ThresholdedPauliChannel(const PauliChannel& channel)
        : p_x_threshold(probability_to_threshold(channel.p_x)),
          p_y_threshold(probability_to_threshold(channel.p_y)),
          p_z_threshold(probability_to_threshold(channel.p_z)) {}
    
    static uint64_t probability_to_threshold(double p) {
        if (p < 0.0 || p > 1.0) {
            throw std::invalid_argument("Probability must be between 0 and 1.");
        }
        if (p == 0.0) {
            return 0;
        }
        if (p == 1.0) {
            return std::numeric_limits<std::uint64_t>::max();
        }
        return static_cast<std::uint64_t>(p * static_cast<long double>(std::numeric_limits<std::uint64_t>::max()));
    }
};
