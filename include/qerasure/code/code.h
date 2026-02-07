#include<vector>

using QubitIndex = std::size_t;
using Stabilizer = std::vector<QubitIndex>;

#ifndef QERASURE_CODE_H
#define QERASURE_CODE_H

class RotatedSurfaceCode {
    public:
        explicit RotatedSurfaceCode(std::size_t distance);

        const std::size_t& distance() const noexcept { return distance_; }
        const std::size_t& num_qubits() const noexcept { return 2 * distance_ * distance_ - 1; }

        const std::vector<Stabilizer> &stabilizers() const noexcept { return stabilizers_; }

    private:
        std::size_t distance_;
        std::vector<Stabilizer> stabilizers_;

        void build();

        void build_lattice();
        void build_stabilizers();
};

#endif // QERASURE_CODE_H
