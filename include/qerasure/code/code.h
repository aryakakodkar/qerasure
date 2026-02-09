#include <vector>
#include <map>
#include <unordered_map>

#ifndef QERASURE_NO_PARTNER_DEFINED
#define QERASURE_NO_PARTNER_DEFINED

const std::size_t NO_PARTNER = std::numeric_limits<std::size_t>::max(); // Value to represent no partner for a stabilizer

#endif

using QubitIndex = std::size_t;
using Stabilizer = std::vector<QubitIndex>;

#ifndef ROTATED_SURFACE_CODE
#define ROTATED_SURFACE_CODE

// Hash function for std::pair<QubitIndex, QubitIndex> to enable use as unordered_map key
struct PairHash {
    std::size_t operator()(const std::pair<QubitIndex, QubitIndex>& p) const {
        std::size_t h1 = std::hash<QubitIndex>{}(p.first);
        std::size_t h2 = std::hash<QubitIndex>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

class RotatedSurfaceCode {
    public:
        explicit RotatedSurfaceCode(std::size_t distance);

        const std::size_t distance() const noexcept { return distance_; }
        const std::size_t num_qubits() const noexcept { return num_qubits_; }

        const std::vector<std::pair<QubitIndex, QubitIndex>>& gates() const noexcept { return gates_; }

        const std::unordered_map<std::pair<QubitIndex, QubitIndex>, QubitIndex, PairHash>& coord_to_index() const noexcept { return coord_to_index_; }
        const std::unordered_map<QubitIndex, std::pair<QubitIndex, QubitIndex>>& index_to_coord() const noexcept { return index_to_coord_; }
        const std::vector<std::size_t>& partner_map() const noexcept { return partner_map_; }

        const std::size_t& x_anc_offset() const noexcept { return x_anc_offset_; }
        const std::size_t& z_anc_offset() const noexcept { return z_anc_offset_; }

    private:
        std::size_t distance_; // code distance
        std::size_t num_qubits_; // total number of qubits in the code (2 * d**2 - 1)
        
        std::unordered_map<std::pair<QubitIndex, QubitIndex>, QubitIndex, PairHash> coord_to_index_; // maps (x, y) coordinates to qubit indices
        std::unordered_map<QubitIndex, std::pair<QubitIndex, QubitIndex>> index_to_coord_; // maps qubit indices to (x, y) coordinates

        std::vector<std::pair<QubitIndex, QubitIndex>> gates_; // A flat vector of CNOT gates in the syndrome extraction circuit
        std::array<std::vector<std::pair<QubitIndex, QubitIndex>>::iterator, 4> step_iters_; // Iterators delimiting the 4 steps in the syndrome extraction schedule

        std::vector<std::size_t> partner_map_; // Maps each qubit to its partner in the gate

        std::size_t x_anc_offset_; // offset for x-ancilla qubit indices
        std::size_t z_anc_offset_; // offset for z-ancilla qubit indices

        void build();

        void build_lattice();
        void build_stabilizers();
};

#endif // ROTATED_SURFACE_CODE
