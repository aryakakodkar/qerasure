#include "qerasure/noise/noise.h"
#include <initializer_list>
#include <stdexcept>

// Throws std::invalid_argument if any probability is invalid
void NoiseParams::validate() const {
	std::initializer_list<double> params = {
		p_single_qubit_depolarize, p_two_qubit_depolarize, p_measurement_error,
		p_single_qubit_erasure, p_two_qubit_erasure, p_erasure_check_error
	};
	for (double p : params) {
		if (p < 0.0 || p > 1.0) {
			throw std::invalid_argument("Noise probabilities must be between 0 and 1");
		}
	}
}

// Factory function to construct and validate NoiseParams
NoiseParams build_noise_model(NoiseParams params) {
    params.validate();
	return params;
}
