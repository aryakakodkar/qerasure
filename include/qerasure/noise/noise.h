#include<unordered_map>
#include<string>

/**
 * Factory function to construct and validate NoiseParams.
 * Throws std::invalid_argument if any probability is invalid.
 */
#ifndef NOISE_PARAMS
#define NOISE_PARAMS

// Struct to hold noise parameters for the error model. All probabilities must be between 0 and 1.
struct NoiseParams {
    private:
        std::unordered_map<std::string, double> probabilities; // Map of noise mechanism names to their probabilities

    public:
        explicit NoiseParams();

        void set(const std::string& err_mech, double prob);

        const double& get(const std::string& err_mech) const;
        const std::unordered_map<std::string, double>& get_all() const;

        auto begin() const { return probabilities.begin(); }
        auto end() const { return probabilities.end(); }
};

#endif // NOISE_PARAMS
