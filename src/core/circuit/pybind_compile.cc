#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/circuit/compile.h"

namespace py = pybind11;

namespace qerasure::python_bindings {

void bind_compile(py::module_& m) {
	py::class_<circuit::CheckLookbackLink>(m, "CheckLookbackLink")
		.def_readonly("qubit_index", &circuit::CheckLookbackLink::qubit_index)
		.def_readonly("check_op_index", &circuit::CheckLookbackLink::check_op_index)
		.def_readonly("lookback_check_event_index",
					  &circuit::CheckLookbackLink::lookback_check_event_index)
		.def_readonly("reset_op_after_lookback", &circuit::CheckLookbackLink::reset_op_after_lookback);

	py::class_<circuit::CompiledErasureProgram>(m, "CompiledErasureProgram")
		.def(py::init<const circuit::ErasureCircuit&, const circuit::ErasureModel&>(),
			 py::arg("circuit"), py::arg("model"))
		.def_property_readonly("max_qubit_index", &circuit::CompiledErasureProgram::max_qubit_index)
		.def_property_readonly("num_checks", &circuit::CompiledErasureProgram::num_checks)
		.def_property_readonly("max_persistence", &circuit::CompiledErasureProgram::max_persistence)
		.def_property_readonly("erasable_qubits", &circuit::CompiledErasureProgram::erasable_qubits)
		.def_readonly("check_lookback_links", &circuit::CompiledErasureProgram::check_lookback_links)
		.def_readonly("qubit_operation_indices",
					  &circuit::CompiledErasureProgram::qubit_operation_indices)
		.def_readonly("qubit_check_operation_indices",
					  &circuit::CompiledErasureProgram::qubit_check_operation_indices)
		.def_readonly("qubit_reset_operation_indices",
					  &circuit::CompiledErasureProgram::qubit_reset_operation_indices)
		.def("print_summary", &circuit::CompiledErasureProgram::print_summary);
}

}  // namespace qerasure::python_bindings
