#include <fstream>
#include <limits>
#include <sstream>
#include <string>

#include "instruction.h"
#include "circuit.h"

#include <cctype>
#include <stdexcept>

namespace qerasure::circuit {

namespace {

std::string trim_copy(const std::string& value) {
    std::size_t begin = 0;
    while (begin < value.size() &&
           std::isspace(static_cast<unsigned char>(value[begin])) != 0) {
        ++begin;
    }
    std::size_t end = value.size();
    while (end > begin &&
           std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
        --end;
    }
    return value.substr(begin, end - begin);
}

std::string strip_comments(const std::string& line) {
    std::size_t cut = line.find('#');
    const std::size_t slash_comment = line.find("//");
    if (slash_comment != std::string::npos && (cut == std::string::npos || slash_comment < cut)) {
        cut = slash_comment;
    }
    if (cut == std::string::npos) {
        return line;
    }
    return line.substr(0, cut);
}

double parse_probability_token(const std::string& token, std::size_t line_number,
                               std::string* op_name) {
    const std::size_t open_paren = token.find('(');
    const std::size_t close_paren = token.find(')');
    if (open_paren == std::string::npos && close_paren == std::string::npos) {
        *op_name = token;
        return 0.0;
    }
    if (open_paren == std::string::npos || close_paren == std::string::npos ||
        close_paren < open_paren || close_paren + 1 != token.size()) {
        throw std::invalid_argument("Malformed op token at line " + std::to_string(line_number) +
                                    ": " + token);
    }

    *op_name = token.substr(0, open_paren);
    if (op_name->empty()) {
        throw std::invalid_argument("Missing operation name at line " + std::to_string(line_number));
    }
    const std::string arg_text = token.substr(open_paren + 1, close_paren - open_paren - 1);
    if (arg_text.empty()) {
        throw std::invalid_argument("Missing probability value at line " +
                                    std::to_string(line_number));
    }

    std::size_t parsed = 0;
    double value = 0.0;
    try {
        value = std::stod(arg_text, &parsed);
    } catch (const std::exception&) {
        throw std::invalid_argument("Invalid probability '" + arg_text + "' at line " +
                                    std::to_string(line_number));
    }
    if (parsed != arg_text.size()) {
        throw std::invalid_argument("Invalid probability '" + arg_text + "' at line " +
                                    std::to_string(line_number));
    }
    return value;
}

uint32_t parse_target_token(const std::string& token, std::size_t line_number) {
    if (!token.empty() && token[0] == '-') {
        throw std::invalid_argument("Negative target index '" + token + "' at line " +
                                    std::to_string(line_number));
    }
    std::size_t parsed = 0;
    unsigned long long raw = 0;
    try {
        raw = std::stoull(token, &parsed, 10);
    } catch (const std::exception&) {
        throw std::invalid_argument("Invalid target token '" + token + "' at line " +
                                    std::to_string(line_number));
    }
    if (parsed != token.size() || raw > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("Invalid target token '" + token + "' at line " +
                                    std::to_string(line_number));
    }
    return static_cast<uint32_t>(raw);
}

}  // namespace

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

void ErasureCircuit::safe_append(const std::string& op, const std::vector<uint32_t>& targets, double arg) {
    auto it = opcode_map().find(op);
    if (it == opcode_map().end()) {
        throw std::invalid_argument("Invalid operation: " + op);
    }
    validate_instruction_(it->second, targets, arg);

    append(it->second, targets, arg);
}

void ErasureCircuit::from_stream_(std::istream& stream) {
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(stream, line)) {
        ++line_number;
        const std::string stripped = trim_copy(strip_comments(line));
        if (stripped.empty()) {
            continue;
        }

        std::istringstream line_stream(stripped);
        std::string op_token;
        line_stream >> op_token;
        if (op_token.empty()) {
            continue;
        }

        std::string op_name;
        const double arg = parse_probability_token(op_token, line_number, &op_name);

        std::vector<uint32_t> targets;
        std::string target_token;
        while (line_stream >> target_token) {
            targets.push_back(parse_target_token(target_token, line_number));
        }
        try {
            safe_append(op_name, targets, arg);
        } catch (const std::exception& ex) {
            throw std::invalid_argument("Line " + std::to_string(line_number) + ": " + ex.what());
        }
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

}  // namespace qerasure::circuit
