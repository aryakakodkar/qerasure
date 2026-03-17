#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/circuit/rail_surface_compile.h"

namespace py = pybind11;

namespace qerasure::python_bindings {

void bind_rail_surface_compile(py::module_& m) {
  py::class_<circuit::RailSurfaceCompiledProgram>(m, "RailSurfaceCompiledProgram")
      .def(py::init<
               const circuit::ErasureCircuit&,
               const circuit::ErasureModel&,
               uint32_t,
               uint32_t>(),
           py::arg("circuit"),
           py::arg("model"),
           py::arg("distance"),
           py::arg("rounds"))
      .def_property_readonly("base_program", &circuit::RailSurfaceCompiledProgram::base_program)
      .def_property_readonly("distance", &circuit::RailSurfaceCompiledProgram::distance)
      .def_property_readonly("rounds", &circuit::RailSurfaceCompiledProgram::rounds)
      .def_property_readonly("num_data_qubits", &circuit::RailSurfaceCompiledProgram::num_data_qubits)
      .def_property_readonly("x_anc_offset", &circuit::RailSurfaceCompiledProgram::x_anc_offset)
      .def_property_readonly("z_anc_offset", &circuit::RailSurfaceCompiledProgram::z_anc_offset)
      .def_property_readonly("num_z_ancillas", &circuit::RailSurfaceCompiledProgram::num_z_ancillas)
      .def_property_readonly("num_detectors", &circuit::RailSurfaceCompiledProgram::num_detectors)
      .def_property_readonly(
          "check_event_to_qubit",
          &circuit::RailSurfaceCompiledProgram::check_event_to_qubit)
      .def_property_readonly(
          "check_event_to_op_index",
          &circuit::RailSurfaceCompiledProgram::check_event_to_op_index)
      .def("is_data_qubit", &circuit::RailSurfaceCompiledProgram::is_data_qubit, py::arg("qubit"))
      .def(
          "data_qubit_schedule_type",
          &circuit::RailSurfaceCompiledProgram::data_qubit_schedule_type,
          py::arg("data_qubit"))
      .def(
          "data_qubit_is_boundary",
          &circuit::RailSurfaceCompiledProgram::data_qubit_is_boundary,
          py::arg("data_qubit"))
      .def("op_round", &circuit::RailSurfaceCompiledProgram::op_round, py::arg("op_index"))
      .def(
          "data_z_ancilla_slots",
          &circuit::RailSurfaceCompiledProgram::data_z_ancilla_slots,
          py::arg("data_qubit"))
      .def(
          "round_detector_index",
          &circuit::RailSurfaceCompiledProgram::round_detector_index,
          py::arg("round_index"),
          py::arg("z_ancilla"))
      .def(
          "interaction_op_for_data_z_ancilla",
          &circuit::RailSurfaceCompiledProgram::interaction_op_for_data_z_ancilla,
          py::arg("data_qubit"),
          py::arg("z_ancilla"),
          py::arg("round_index"));
}

}  // namespace qerasure::python_bindings
