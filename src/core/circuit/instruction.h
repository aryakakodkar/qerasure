# pragma once

#include<stdexcept>
#include <vector>
#include <string>
#include <unordered_map>

enum class OpCode {
    // Stim operations
    STIM_OPS,
    H,
    CX,
    M,
    R,
    MR,
    STIM_PROB_OPS, // probabilistic operations start here
    X_ERROR,
    Z_ERROR,
    DEPOLARIZE1,
    
    // Erasure instructions
    ERASURE_OPS,
    ERASE,
    ERASE2,
    EC,
    ECR,
    COND_ER
};

inline bool is_stim_op(OpCode op) {
    return op > OpCode::STIM_OPS && op != OpCode::STIM_PROB_OPS && op < OpCode::ERASURE_OPS;
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

inline bool is_two_qubit_op(OpCode op) {
    return op == OpCode::CX || op == OpCode::ERASE2; // limits support for now
}

inline bool is_single_onset_op(OpCode op) {
    return op == OpCode::ERASE;
}

inline bool is_multi_onset_op(OpCode op) {
    return op == OpCode::ERASE2;
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

inline const std::unordered_map<std::string, OpCode>& opcode_map() {
    static const std::unordered_map<std::string, OpCode> kMap = {
        {"H",           OpCode::H},
        {"CX",          OpCode::CX},
        {"M",           OpCode::M},
        {"R",           OpCode::R},
        {"MR",          OpCode::MR},
        {"X_ERROR",     OpCode::X_ERROR},
        {"Z_ERROR",     OpCode::Z_ERROR},
        {"DEPOLARIZE1", OpCode::DEPOLARIZE1},
        {"ERASE",       OpCode::ERASE},
        {"ERASE2",      OpCode::ERASE2},
        {"EC",          OpCode::EC},
        {"ECR",         OpCode::ECR},
        {"COND_ER",     OpCode::COND_ER},
    };
    return kMap;
}

struct Instruction {
    OpCode op;

    std::vector<uint32_t> targets;

    double arg = 0.0; // probability if needed
};
