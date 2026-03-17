#include <pybind11/pybind11.h>

#include "core/gen/surf.h"

namespace py = pybind11;

namespace qerasure::python_bindings {

void bind_surf_gen(py::module_& m) {
	py::class_<gen::SurfaceCodeRotated>(m, "SurfaceCodeRotated")
		.def(py::init<uint32_t>(), py::arg("distance"))
		.def("build_circuit", &gen::SurfaceCodeRotated::build_circuit, py::arg("rounds"),
			 py::arg("erasure_prob"), py::arg("erasable_qubits") = "ALL",
			 py::arg("reset_failure_prob") = 0.0, py::arg("ecr_after_each_step") = false,
			 py::arg("single_qubit_errors") = false,
			 py::arg("post_clifford_pauli_prob") = 0.0,
			 py::arg("rounds_per_check") = 1);
}

}  // namespace qerasure::python_bindings
