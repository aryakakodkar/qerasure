# pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <stdexcept>

namespace qerasure::circuit {

enum class OpCode {
    // Stim operations
    STIM_OPS,
    H,
    CX,
    M,
    R,
    MR,
    DETECTOR,
    OBSERVABLE_INCLUDE,
    STIM_PROB_OPS, // probabilistic operations start here
    X_ERROR,
    Z_ERROR,
    DEPOLARIZE1,
    
    // Erasure instructions
    ERASURE_OPS,
    ERASE,
    ERASE2,
    ERASE2_ANY,
    EC,
    ECR,
    COND_ER
};

inline bool is_stim_op(OpCode op) {
    return op > OpCode::STIM_OPS && op != OpCode::STIM_PROB_OPS && op < OpCode::ERASURE_OPS;
}

inline bool is_measurement_op(OpCode op) {
    return op == OpCode::M || op == OpCode::MR;
}

inline bool is_entangling_op(OpCode op) {
    return op == OpCode::CX;
}

inline bool is_erasure_op(OpCode op) {
    return op > OpCode::ERASURE_OPS;
}

inline bool is_probabilistic_op(OpCode op) {
    return op > OpCode::STIM_PROB_OPS && op != OpCode::ERASURE_OPS;
}

inline bool uses_measurement_record_targets(OpCode op) {
    return op == OpCode::DETECTOR || op == OpCode::OBSERVABLE_INCLUDE;
}

inline bool is_two_qubit_op(OpCode op) {
    return op == OpCode::CX || op == OpCode::ERASE2 || op == OpCode::ERASE2_ANY; // limits support for now
}

inline bool is_single_onset_op(OpCode op) {
    return op == OpCode::ERASE;
}

inline bool is_multi_onset_op(OpCode op) {
    return op == OpCode::ERASE2 || op == OpCode::ERASE2_ANY;
}

inline bool is_hook_op(OpCode op) {
    return op == OpCode::EC || op == OpCode::ECR || op == OpCode::COND_ER;
}

inline bool is_erasure_check_op(OpCode op) {
    return op == OpCode::EC || op == OpCode::ECR;
}

inline bool is_erasure_reset_op(OpCode op) {
    return op == OpCode::ECR || op == OpCode::COND_ER;
}

// These operations should be skippped if any target is erased
inline bool is_erasure_skippable_op(OpCode op) {
    return op == OpCode::X_ERROR || op == OpCode::Z_ERROR || op == OpCode::DEPOLARIZE1;
}

inline const std::unordered_map<std::string, OpCode>& opcode_map() {
    static const std::unordered_map<std::string, OpCode> kMap = {
        {"H",           OpCode::H},
        {"CX",          OpCode::CX},
        {"M",           OpCode::M},
        {"R",           OpCode::R},
        {"MR",          OpCode::MR},
        {"DETECTOR",    OpCode::DETECTOR},
        {"OBSERVABLE_INCLUDE", OpCode::OBSERVABLE_INCLUDE},
        {"X_ERROR",     OpCode::X_ERROR},
        {"Z_ERROR",     OpCode::Z_ERROR},
        {"DEPOLARIZE1", OpCode::DEPOLARIZE1},
        {"ERASE",       OpCode::ERASE},
        {"ERASE2",      OpCode::ERASE2},
        {"ERASE2_ANY",  OpCode::ERASE2_ANY},
        {"EC",          OpCode::EC},
        {"ECR",         OpCode::ECR},
        {"COND_ER",     OpCode::COND_ER},
    };
    return kMap;
}

inline const char* opcode_name(OpCode op) {
    switch (op) {
        case OpCode::H:
            return "H";
        case OpCode::CX:
            return "CX";
        case OpCode::M:
            return "M";
        case OpCode::R:
            return "R";
        case OpCode::MR:
            return "MR";
        case OpCode::DETECTOR:
            return "DETECTOR";
        case OpCode::OBSERVABLE_INCLUDE:
            return "OBSERVABLE_INCLUDE";
        case OpCode::X_ERROR:
            return "X_ERROR";
        case OpCode::Z_ERROR:
            return "Z_ERROR";
        case OpCode::DEPOLARIZE1:
            return "DEPOLARIZE1";
        case OpCode::ERASE:
            return "ERASE";
        case OpCode::ERASE2:
            return "ERASE2";
        case OpCode::ERASE2_ANY:
            return "ERASE2_ANY";
        case OpCode::EC:
            return "EC";
        case OpCode::ECR:
            return "ECR";
        case OpCode::COND_ER:
            return "COND_ER";
        case OpCode::STIM_OPS:
        case OpCode::STIM_PROB_OPS:
        case OpCode::ERASURE_OPS:
            break;
    }
    throw std::invalid_argument("opcode_name does not support sentinel opcode");
}

struct Instruction {
    OpCode op;

    std::vector<uint32_t> targets;

    double arg = 0.0; // probability if needed
};

}  // namespace qerasure::circuit
