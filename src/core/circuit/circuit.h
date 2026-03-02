# pragma once

#include <iosfwd>
#include <string>

#include "instruction.h"
#include "core/circuit/erasure_model.h"

namespace qerasure::circuit {

class ErasureCircuit {
    public:
        ErasureCircuit();

        // Fast append
        void append(OpCode op, const std::vector<uint32_t>& targets, double arg = 0.0);

        // Safe append with string
        void safe_append(const std::string& op, const std::vector<uint32_t>& targets,
                         double arg = 0.0);
        
        // Append measurement-record based detector using positive rec lookbacks.
        void append_detector(const std::vector<uint32_t>& rec_lookbacks);

        // Append measurement-record based logical observable include using positive rec lookbacks.
        void append_observable_include(const std::vector<uint32_t>& rec_lookbacks);

        void from_string(const std::string& circuit_str);
        void from_file(const std::string& filepath);
        std::string to_string() const;
        
        const std::vector<Instruction>& instructions() const {
            return instructions_;
        }   

    private:
        std::vector<Instruction> instructions_;

        static void validate_instruction_(OpCode op, const std::vector<uint32_t>& targets, double arg);

        // Shared parsing logic for from_string and from_file.
        void from_stream_(std::istream& stream);
};

std::ostream& operator<<(std::ostream& out, const ErasureCircuit& circuit);

}  // namespace qerasure::circuit
