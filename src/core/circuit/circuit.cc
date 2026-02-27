#include <sstream>
#include <fstream>

#include "instruction.h"
#include "circuit.h"

ErasureCircuit::ErasureCircuit() {}

void ErasureCircuit::validate_instruction_(OpCode op, const std::vector<uint32_t>& targets, double arg) {
    if (is_probabilistic_op(op) && (arg < 0.0 || arg > 1.0)) {
        throw std::invalid_argument("Probability argument must be between 0 and 1.");
    } else if (!is_probabilistic_op(op) && arg != 0.0) {
        throw std::invalid_argument("Non-probabilistic operations should not have a non-zero argument.");
    }
    if (is_two_qubit_op(op) && targets.size()%2 != 0)  {
        throw std::invalid_argument("Two-qubit operations require an even number of targets.");
    }
}

void ErasureCircuit::append(OpCode op, const std::vector<uint32_t>& targets, double arg) {
    validate_instruction_(op, targets, arg);
    
    instructions_.push_back({op, targets, arg});
}

void ErasureCircuit::safe_append(std::string op, const std::vector<uint32_t>& targets, double arg) {
    auto it = opcode_map().find(op);
    if (it == opcode_map().end()) {
        throw std::invalid_argument("Invalid operation: " + op);
    }
    validate_instruction_(it->second, targets, arg);

    append(it->second, targets, arg);
}

void ErasureCircuit::from_stream_(std::istream& stream) {
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty()) continue;

        std::istringstream line_stream(line);
        std::string op_str;
        line_stream >> op_str;

        double arg = 0.0;
        auto paren_open  = op_str.find('(');
        auto paren_close = op_str.find(')');
        if (paren_open != std::string::npos && paren_close != std::string::npos) {
            arg = std::stod(op_str.substr(paren_open + 1, paren_close - paren_open - 1));
            op_str = op_str.substr(0, paren_open);
        }

        std::vector<uint32_t> targets;
        std::string target;
        while (line_stream >> target) {
            targets.push_back(std::stoul(target));
        }

        safe_append(op_str, targets, arg);
    }
}

void ErasureCircuit::from_string(const std::string& circuit_str) {
    std::istringstream stream(circuit_str);
    from_stream_(stream);
}

void ErasureCircuit::from_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::invalid_argument("Could not open file: " + filepath);
    }
    from_stream_(file);
}
    