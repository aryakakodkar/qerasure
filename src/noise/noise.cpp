#include "qerasure/noise/noise.h"
#include <initializer_list>
#include <stdexcept>
#include <string>

NoiseParams::NoiseParams() {
    probabilities["p_single_qubit_depolarize"] = 0.0;
    probabilities["p_two_qubit_depolarize"] = 0.0;
    probabilities["p_measurement_error"] = 0.0;
    probabilities["p_single_qubit_erasure"] = 0.0;
    probabilities["p_two_qubit_erasure"] = 0.0;
    probabilities["p_erasure_check_error"] = 0.0;
}

// Set the probability for a given noise mechanism, validating the input
void NoiseParams::set(const std::string& err_mech, double prob) {
    if (probabilities.find(err_mech) == probabilities.end()) {
        throw std::invalid_argument("Invalid noise parameter err_mech: " + err_mech);
    }
    if (prob < 0.0 || prob > 1.0) {
        throw std::invalid_argument("Probability probs must be between 0 and 1");
    }
    probabilities[err_mech] = prob;
}

// Get the probability for a given noise mechanism, validating the input
const double& NoiseParams::get(const std::string& err_mech) const {
    auto it = probabilities.find(err_mech);
    if (it == probabilities.end()) {
        throw std::invalid_argument("Invalid noise parameter err_mech: " + err_mech);
    }
    return it->second;
}

const std::unordered_map<std::string, double>& NoiseParams::get_all() const {
    return probabilities;
}
