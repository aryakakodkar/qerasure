#include "core/simulator/sim_internal_utils.h"
#include "core/circuit/instruction.h"
#include "stim/circuit/gate_target.h"

namespace qerasure::simulator::internal {

void append_mapped_stim_instruction(circuit::OpCode op, const std::vector<uint32_t>& targets, double arg, stim::Circuit* circuit) {
    switch (op) {
        case circuit::OpCode::H:
            circuit->safe_append_u("H", targets);
            return;
        case circuit::OpCode::CX:
            circuit->safe_append_u("CX", targets);
            return;
        case circuit::OpCode::M:
            circuit->safe_append_u("M", targets);
            return;
        case circuit::OpCode::R:
            circuit->safe_append_u("R", targets);
            return;
        case circuit::OpCode::MR:
            circuit->safe_append_u("MR", targets);
            return;
        case circuit::OpCode::DETECTOR: {
            std::vector<uint32_t> rec_targets;
            rec_targets.reserve(targets.size());
            for (const uint32_t lookback : targets) {
                rec_targets.push_back(lookback | stim::TARGET_RECORD_BIT);
            }
            circuit->safe_append_u("DETECTOR", rec_targets);
            return;
        }
        case circuit::OpCode::OBSERVABLE_INCLUDE: {
            std::vector<uint32_t> rec_targets;
            rec_targets.reserve(targets.size());
            for (const uint32_t lookback : targets) {
                rec_targets.push_back(lookback | stim::TARGET_RECORD_BIT);
            }
            circuit->safe_append_ua("OBSERVABLE_INCLUDE", rec_targets, 0.0);
            return;
        }
        case circuit::OpCode::X_ERROR:
            circuit->safe_append_ua("X_ERROR", targets, arg);
            return;
        case circuit::OpCode::Z_ERROR:
            circuit->safe_append_ua("Z_ERROR", targets, arg);
            return;
        case circuit::OpCode::DEPOLARIZE1:
            circuit->safe_append_ua("DEPOLARIZE1", targets, arg);
            return;
        default:
            throw std::invalid_argument("Unsupported instruction for mapping to Stim: " + std::to_string(static_cast<int>(op)));
    }
};

void append_mapped_stim_instruction(const circuit::Instruction& instr, stim::Circuit* circuit) {
    append_mapped_stim_instruction(instr.op, instr.targets, instr.arg, circuit);
}

void append_mapped_pauli_operation(const uint32_t qubit_index,
                                   const PauliOperation operation,
                                   stim::Circuit* circuit) {
    switch (operation) {
        case PauliOperation::I:
            return;
        case PauliOperation::X:
            circuit->safe_append_ua("X_ERROR", {qubit_index}, 1.0);
            return;
        case PauliOperation::Y:
            circuit->safe_append_ua("Y_ERROR", {qubit_index}, 1.0);
            return;
        case PauliOperation::Z:
            circuit->safe_append_ua("Z_ERROR", {qubit_index}, 1.0);
            return;
    }
}

PauliOperation sample_thresholded_pauli_channel(const ThresholdedPauliChannel& channel,
                                 qerasure::simulator::FastRng* rng) {
    const std::uint64_t draw = rng->next_u64();
    const std::uint64_t x_cut = channel.p_x_threshold;
    const std::uint64_t y_cut = x_cut + channel.p_y_threshold;
    const std::uint64_t z_cut = y_cut + channel.p_z_threshold;
    if (draw < x_cut) {
        return PauliOperation::X;
    }
    if (draw < y_cut) {
        return PauliOperation::Y;
    }
    if (draw < z_cut) {
        return PauliOperation::Z;
    }
    return PauliOperation::I;
}

} // namespace qerasure::simulator::internal
