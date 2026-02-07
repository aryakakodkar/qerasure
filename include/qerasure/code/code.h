#include <vector>
#include <map>
#include <unordered_map>

// Hash function for std::vector<T> so it can be used as a key in std::unordered_map
namespace std {
    template <typename T>
    struct hash<std::vector<T>> {
        std::size_t operator()(const std::vector<T>& v) const {
            std::size_t seed = v.size();
            for (const auto& elem : v) {
                seed ^= std::hash<T>{}(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };
}

using QubitIndex = std::size_t;
using Stabilizer = std::vector<QubitIndex>;

#ifndef QERASURE_CODE_H
#define QERASURE_CODE_H

class RotatedSurfaceCode {
    public:
        explicit RotatedSurfaceCode(std::size_t distance);

        const std::size_t& distance() const noexcept { return distance_; }
        const std::size_t& num_qubits() const noexcept { return num_qubits_; }

        const std::vector<Stabilizer>& stabilizers() const noexcept { return stabilizers_; }

        const std::vector<std::vector<std::pair<QubitIndex, QubitIndex>>>& gates() const noexcept { return gates_; }

        const std::unordered_map<Stabilizer, QubitIndex>& coord_to_index() const noexcept { return coord_to_index_; }
        const std::unordered_map<QubitIndex, Stabilizer>& index_to_coord() const noexcept { return index_to_coord_; }

        const std::size_t& x_anc_offset() const noexcept { return x_anc_offset_; }
        const std::size_t& z_anc_offset() const noexcept { return z_anc_offset_; }

    private:
        std::size_t distance_;
        std::size_t num_qubits_;
        std::vector<Stabilizer> stabilizers_;
        
        std::unordered_map<Stabilizer, QubitIndex> coord_to_index_; // maps (x, y) coordinates to qubit indices
        std::unordered_map<QubitIndex, Stabilizer> index_to_coord_; // maps qubit indices to (x, y) coordinates

        std::vector<std::vector<std::pair<QubitIndex, QubitIndex>>> gates_;

        std::size_t x_anc_offset_; // offset for x-ancilla qubit indices
        std::size_t z_anc_offset_; // offset for z-ancilla qubit indices

        void build();

        void build_lattice();
        void build_stabilizers();
};

#endif // QERASURE_CODE_H
