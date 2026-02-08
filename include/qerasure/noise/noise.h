/**
 * Factory function to construct and validate NoiseParams.
 * Throws std::invalid_argument if any probability is invalid.
 */
#ifndef NOISE_PARAMS
#define NOISE_PARAMS

// Struct to hold noise parameters for the error model. All probabilities must be between 0 and 1.
// Do not initialize the struct directly; use the build_noise_model factory function to ensure validation.
struct NoiseParams {
        double p_single_qubit_depolarize = 0.0;
        double p_two_qubit_depolarize = 0.0;
        double p_measurement_error = 0.0;
        double p_single_qubit_erasure = 0.0;
        double p_two_qubit_erasure = 0.0;
        double p_erasure_check_error = 0.0;

        void validate() const;
};

// Factory function to construct and validate NoiseParams
// Unspecified probabilities default to 0.0. Throws std::invalid_argument if any probability is invalid.
// Example usage: NoiseParams noise = build_noise_model({.p_two_qubit_erasure = 0.01, .p_erasure_check_error = 0.05});
NoiseParams build_noise_model(NoiseParams params = {});

#endif // NOISE_PARAMS
