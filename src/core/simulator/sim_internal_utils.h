#pragma once

#include "stim/circuit/circuit.h"
#include "core/circuit/instruction.h"
#include "core/model/pauli_channel.h"
#include "core/simulator/fast_rng.h"

namespace qerasure::simulator::internal {

enum class PauliOperation {
    X,
    Y,
    Z,
    I
};

void append_mapped_stim_instruction(const circuit::Instruction& instr, stim::Circuit* circuit);

void append_mapped_stim_instruction(const circuit::OpCode op, 
                                    const std::vector<uint32_t>& targets, 
                                    double arg, 
                                    stim::Circuit* circuit);

void append_mapped_pauli_operation(const uint32_t qubit_index, 
                                    const PauliOperation operation, 
                                    stim::Circuit* circuit);

PauliOperation sample_thresholded_pauli_channel(const ThresholdedPauliChannel& channel,
                                                qerasure::simulator::FastRng* rng);

};
