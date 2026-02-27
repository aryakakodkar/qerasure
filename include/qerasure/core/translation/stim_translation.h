#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "qerasure/core/code/rotated_surface_code.h"

namespace stim {
struct Circuit;
}

namespace qerasure {
struct LoweringParams;
struct LoweringResult;
struct SpreadProgram;

// Build a Stim-format circuit for rotated-surface-code syndrome extraction and readout.
//
// Semantics:
// - Repeats stabilizer extraction for (qec_rounds - 1) rounds.
// - Uses H on X ancillas, then 4 schedule-step CNOT layers, then H on X ancillas.
// - Measures all ancillas each extraction round and emits DETECTOR parity checks.
// - Performs final data measurement, Z-plaquette boundary detectors, and logical-Z observable.
stim::Circuit build_surf_stabilizer_circuit_object(const RotatedSurfaceCode& code, std::size_t qec_rounds);
std::string build_surf_stabilizer_circuit(const RotatedSurfaceCode& code, std::size_t qec_rounds);

// Backward-compatible aliases.
stim::Circuit build_surface_code_stim_circuit_object(const RotatedSurfaceCode& code, std::size_t qec_rounds);
std::string build_surface_code_stim_circuit(const RotatedSurfaceCode& code, std::size_t qec_rounds);

// Build the logically equivalent circuit from lowered erasure events for one shot.
// Lowering errors are injected immediately after the gate-step where they occur.
stim::Circuit build_logical_stabilizer_circuit_object(
    const RotatedSurfaceCode& code, const LoweringResult& lowering_result, std::size_t shot_index = 0);
std::string build_logical_stabilizer_circuit(
    const RotatedSurfaceCode& code, const LoweringResult& lowering_result, std::size_t shot_index = 0);

// Backward-compatible aliases.
stim::Circuit build_logically_equivalent_erasure_stim_circuit_object(
    const RotatedSurfaceCode& code, const LoweringResult& lowering_result, std::size_t shot_index = 0);
std::string build_logically_equivalent_erasure_stim_circuit(
    const RotatedSurfaceCode& code, const LoweringResult& lowering_result, std::size_t shot_index = 0);

// Build a virtual decoder circuit by injecting spread-program probabilities directly into
// syndrome-extraction timesteps, weighted by an erasure-arrival model.
//
// `condition_on_erasure_in_round` controls whether first-erasure timestep probabilities are
// normalized conditioned on "an erasure occurred within the round".
stim::Circuit build_virtual_decoder_stim_circuit_object(
    const RotatedSurfaceCode& code, std::size_t qec_rounds, const LoweringParams& lowering_params,
    const LoweringResult& lowering_result, std::size_t shot_index,
    double two_qubit_erasure_probability, bool condition_on_erasure_in_round = true);
std::string build_virtual_decoder_stim_circuit(
    const RotatedSurfaceCode& code, std::size_t qec_rounds, const LoweringParams& lowering_params,
    const LoweringResult& lowering_result, std::size_t shot_index,
    double two_qubit_erasure_probability, bool condition_on_erasure_in_round = true);

// Build a syndrome-conditioned virtual decoder circuit.
//
// For non-boundary data qubits, first-erasure timestep probabilities are selected from the
// provided conditional distributions based on same-round parity consistency of the two associated
// Z-check detectors in `z_detector_syndrome_bits`.
//
// For boundary data qubits, the default Bernoulli-by-step model is used.
stim::Circuit build_virtual_decoder_stim_circuit_conditioned_object(
    const RotatedSurfaceCode& code, std::size_t qec_rounds, const LoweringParams& lowering_params,
    const LoweringResult& lowering_result, std::size_t shot_index,
    double two_qubit_erasure_probability, const std::vector<std::uint8_t>& z_detector_syndrome_bits,
    const std::vector<double>& p_step_given_consistent_xzzx,
    const std::vector<double>& p_step_given_inconsistent_xzzx,
    const std::vector<double>& p_step_given_consistent_zxxz,
    const std::vector<double>& p_step_given_inconsistent_zxxz,
    bool condition_on_erasure_in_round = true);
std::string build_virtual_decoder_stim_circuit_conditioned(
    const RotatedSurfaceCode& code, std::size_t qec_rounds, const LoweringParams& lowering_params,
    const LoweringResult& lowering_result, std::size_t shot_index,
    double two_qubit_erasure_probability, const std::vector<std::uint8_t>& z_detector_syndrome_bits,
    const std::vector<double>& p_step_given_consistent_xzzx,
    const std::vector<double>& p_step_given_inconsistent_xzzx,
    const std::vector<double>& p_step_given_consistent_zxxz,
    const std::vector<double>& p_step_given_inconsistent_zxxz,
    bool condition_on_erasure_in_round = true);

}  // namespace qerasure
