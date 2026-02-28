# pragma once

#include <iosfwd>
#include <string>

#include "instruction.h"

namespace qerasure::circuit {

class ErasureCircuit {
    public:
        ErasureCircuit();

        // Fast append
        void append(OpCode op, const std::vector<uint32_t>& targets, double arg = 0.0);

        // Safe append with string
        void safe_append(const std::string& op, const std::vector<uint32_t>& targets,
                         double arg = 0.0);

        void from_string(const std::string& circuit_str);
        void from_file(const std::string& filepath);
        
        const std::vector<Instruction>& instructions() const {
            return instructions_;
        }   

    private:
        std::vector<Instruction> instructions_;

        static void validate_instruction_(OpCode op, const std::vector<uint32_t>& targets, double arg);

        // Shared parsing logic for from_string and from_file.
        void from_stream_(std::istream& stream);
};

}  // namespace qerasure::circuit
