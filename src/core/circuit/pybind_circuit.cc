#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/circuit/circuit.h"
#include "core/circuit/erasure_model.h"
#include "core/circuit/instruction.h"
#include "core/model/pauli_channel.h"

namespace py = pybind11;

namespace qerasure::python_bindings {

void bind_circuit(py::module_& m) {
	py::class_<PauliChannel>(m, "PauliChannel")
		.def(py::init<double, double, double>(), py::arg("p_x") = 0.0, py::arg("p_y") = 0.0,
			 py::arg("p_z") = 0.0)
		.def_readwrite("p_x", &PauliChannel::p_x)
		.def_readwrite("p_y", &PauliChannel::p_y)
		.def_readwrite("p_z", &PauliChannel::p_z);

	py::class_<circuit::TQGSpreadModel>(m, "TQGSpreadModel")
		.def(py::init<const PauliChannel&, const PauliChannel&>(),
			 py::arg("control_spread") = PauliChannel{}, py::arg("target_spread") = PauliChannel{})
		.def_readwrite("control_spread", &circuit::TQGSpreadModel::control_spread)
		.def_readwrite("target_spread", &circuit::TQGSpreadModel::target_spread);

	py::class_<circuit::ErasureModel>(m, "ErasureModel")
		.def(py::init<uint32_t, const PauliChannel&, const PauliChannel&, const circuit::TQGSpreadModel&>(),
			 py::arg("max_persistence") = UINT32_MAX, py::arg("onset") = PauliChannel{},
			 py::arg("reset") = PauliChannel{}, py::arg("spread") = circuit::TQGSpreadModel{})
		.def(py::init<uint32_t, const PauliChannel&, const PauliChannel&, const PauliChannel&,
					  const PauliChannel&>(),
			 py::arg("max_persistence"), py::arg("onset"), py::arg("reset"),
			 py::arg("control_spread"), py::arg("target_spread"))
		.def(py::init<uint32_t, const PauliChannel&, const PauliChannel&, const PauliChannel&>(),
			 py::arg("max_persistence"), py::arg("onset"), py::arg("reset"),
			 py::arg("cx_spread"))
		.def_readwrite("max_persistence", &circuit::ErasureModel::max_persistence)
		.def_readwrite("onset", &circuit::ErasureModel::onset)
		.def_readwrite("reset", &circuit::ErasureModel::reset)
		.def_readwrite("spread", &circuit::ErasureModel::spread)
		.def_readwrite("check_false_negative_prob", &circuit::ErasureModel::check_false_negative_prob)
		.def_readwrite("check_false_positive_prob", &circuit::ErasureModel::check_false_positive_prob);

	py::enum_<circuit::OpCode>(m, "OpCode")
		.value("H", circuit::OpCode::H)
		.value("CX", circuit::OpCode::CX)
		.value("M", circuit::OpCode::M)
		.value("R", circuit::OpCode::R)
		.value("MR", circuit::OpCode::MR)
		.value("DETECTOR", circuit::OpCode::DETECTOR)
		.value("OBSERVABLE_INCLUDE", circuit::OpCode::OBSERVABLE_INCLUDE)
		.value("X_ERROR", circuit::OpCode::X_ERROR)
		.value("Z_ERROR", circuit::OpCode::Z_ERROR)
		.value("DEPOLARIZE1", circuit::OpCode::DEPOLARIZE1)
		.value("ERASE", circuit::OpCode::ERASE)
		.value("ERASE2", circuit::OpCode::ERASE2)
		.value("ERASE2_ANY", circuit::OpCode::ERASE2_ANY)
		.value("EC", circuit::OpCode::EC)
		.value("ECR", circuit::OpCode::ECR)
		.value("COND_ER", circuit::OpCode::COND_ER)
		.export_values();

	py::class_<circuit::Instruction>(m, "Instruction")
		.def_readonly("op", &circuit::Instruction::op)
		.def_readonly("targets", &circuit::Instruction::targets)
		.def_readonly("arg", &circuit::Instruction::arg);

	py::class_<circuit::ErasureCircuit>(m, "ErasureCircuit")
		.def(py::init<>())
		.def("append", &circuit::ErasureCircuit::append, py::arg("op"), py::arg("targets"),
			 py::arg("arg") = 0.0)
		.def("safe_append", &circuit::ErasureCircuit::safe_append, py::arg("op"),
			 py::arg("targets"), py::arg("arg") = 0.0)
		.def("append_detector", &circuit::ErasureCircuit::append_detector, py::arg("rec_lookbacks"))
		.def("append_observable_include", &circuit::ErasureCircuit::append_observable_include,
			 py::arg("rec_lookbacks"))
		.def("from_string", &circuit::ErasureCircuit::from_string, py::arg("circuit_str"))
		.def("from_file", &circuit::ErasureCircuit::from_file, py::arg("filepath"))
		.def("to_string", &circuit::ErasureCircuit::to_string)
		.def_property_readonly("instructions", &circuit::ErasureCircuit::instructions)
		.def("__str__", &circuit::ErasureCircuit::to_string);
}

}  // namespace qerasure::python_bindings
