#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "qerasure/core/code/rotated_surface_code.h"
#include "qerasure/core/lowering/lowering.h"
#include "qerasure/core/noise/noise_params.h"
#include "qerasure/core/sim/erasure_simulator.h"
#include "qerasure/core/translation/stim_translation.h"

namespace py = pybind11;

namespace {

py::list gates_to_python(const qerasure::RotatedSurfaceCode& code) {
  const auto& gates_flat = code.gates();
  const std::size_t gates_per_step = code.gates_per_step();

  py::list result;
  for (std::size_t s = 0; s < 4; ++s) {
    py::list step_gates;
    for (std::size_t j = 0; j < gates_per_step; ++j) {
      const std::size_t idx = s * gates_per_step + j;
      if (idx < gates_flat.size()) {
        const auto& gate = gates_flat[idx];
        step_gates.append(py::make_tuple(gate.first, gate.second));
      }
    }
    result.append(step_gates);
  }
  return result;
}

py::dict coord_to_index_to_python(const qerasure::RotatedSurfaceCode& code) {
  const auto& coords = code.index_to_coord();
  py::dict out;
  for (std::size_t idx = 0; idx < coords.size(); ++idx) {
    py::tuple coord = py::make_tuple(coords[idx].first, coords[idx].second);
    out[coord] = idx;
  }
  return out;
}

py::dict index_to_coord_to_python(const qerasure::RotatedSurfaceCode& code) {
  const auto& coords = code.index_to_coord();
  py::dict out;
  for (std::size_t idx = 0; idx < coords.size(); ++idx) {
    py::tuple coord = py::make_tuple(coords[idx].first, coords[idx].second);
    out[py::int_(idx)] = coord;
  }
  return out;
}

std::string noise_repr(const qerasure::NoiseParams& params) {
  std::string repr = "NoiseParams(";

  const auto channels = {
      qerasure::NoiseChannel::kSingleQubitDepolarize,
      qerasure::NoiseChannel::kTwoQubitDepolarize,
      qerasure::NoiseChannel::kMeasurementError,
      qerasure::NoiseChannel::kSingleQubitErasure,
      qerasure::NoiseChannel::kTwoQubitErasure,
      qerasure::NoiseChannel::kErasureCheckError,
  };

  bool first = true;
  for (const auto channel : channels) {
    if (!first) {
      repr += ", ";
    }
    first = false;
    repr += std::string(qerasure::NoiseParams::to_string(channel));
    repr += "=";
    repr += std::to_string(params.get(channel));
  }

  repr += ")";
  return repr;
}

}  // namespace

PYBIND11_MODULE(qerasure_python, m) {
  m.doc() = "Python bindings for the qerasure code library";

  py::class_<qerasure::RotatedSurfaceCode>(m, "RotatedSurfaceCode")
      .def(py::init<std::size_t>(), py::arg("distance"))
      .def_property_readonly("distance", &qerasure::RotatedSurfaceCode::distance)
      .def_property_readonly("num_qubits", &qerasure::RotatedSurfaceCode::num_qubits)
      .def_property_readonly("gates", &gates_to_python)
      .def_property_readonly("coord_to_index", &coord_to_index_to_python)
      .def_property_readonly("index_to_coord", &index_to_coord_to_python)
      .def_property_readonly("partner_map", &qerasure::RotatedSurfaceCode::partner_map)
      .def_property_readonly("data_to_x_ancilla_slots",
                             &qerasure::RotatedSurfaceCode::data_to_x_ancilla_slots)
      .def_property_readonly("data_to_z_ancilla_slots",
                             &qerasure::RotatedSurfaceCode::data_to_z_ancilla_slots)
      .def_property_readonly("x_anc_offset", &qerasure::RotatedSurfaceCode::x_anc_offset)
      .def_property_readonly("z_anc_offset", &qerasure::RotatedSurfaceCode::z_anc_offset)
      .def_property_readonly("gates_per_step", &qerasure::RotatedSurfaceCode::gates_per_step);

  py::enum_<qerasure::NoiseChannel>(m, "NoiseChannel")
      .value("SINGLE_QUBIT_DEPOLARIZE", qerasure::NoiseChannel::kSingleQubitDepolarize)
      .value("TWO_QUBIT_DEPOLARIZE", qerasure::NoiseChannel::kTwoQubitDepolarize)
      .value("MEASUREMENT_ERROR", qerasure::NoiseChannel::kMeasurementError)
      .value("SINGLE_QUBIT_ERASURE", qerasure::NoiseChannel::kSingleQubitErasure)
      .value("TWO_QUBIT_ERASURE", qerasure::NoiseChannel::kTwoQubitErasure)
      .value("ERASURE_CHECK_ERROR", qerasure::NoiseChannel::kErasureCheckError)
      .export_values();

  py::class_<qerasure::NoiseParams>(m, "NoiseParams")
      .def(py::init<>())
      .def("set", py::overload_cast<const std::string&, double>(&qerasure::NoiseParams::set),
           py::arg("key"), py::arg("value"))
      .def("set", py::overload_cast<qerasure::NoiseChannel, double>(&qerasure::NoiseParams::set),
           py::arg("channel"), py::arg("value"))
      .def("get", py::overload_cast<const std::string&>(&qerasure::NoiseParams::get, py::const_),
           py::arg("key"))
      .def("get", py::overload_cast<qerasure::NoiseChannel>(&qerasure::NoiseParams::get, py::const_),
           py::arg("channel"))
      .def("__repr__", &noise_repr);

  py::enum_<qerasure::ErasureQubitSelection>(m, "ErasureQubitSelection")
      .value("ALL_QUBITS", qerasure::ErasureQubitSelection::ALL_QUBITS)
      .value("DATA_QUBITS", qerasure::ErasureQubitSelection::DATA_QUBITS)
      .value("X_ANCILLAS", qerasure::ErasureQubitSelection::X_ANCILLAS)
      .value("Z_ANCILLAS", qerasure::ErasureQubitSelection::Z_ANCILLAS)
      .value("EXPLICIT", qerasure::ErasureQubitSelection::EXPLICIT)
      .export_values();

  py::class_<qerasure::ErasureSimParams>(m, "ErasureSimParams")
      .def(py::init<const qerasure::RotatedSurfaceCode&, const qerasure::NoiseParams&, std::size_t,
                    std::size_t, std::optional<std::uint32_t>, qerasure::ErasureQubitSelection,
                    std::vector<std::size_t>>(),
           py::arg("code"), py::arg("noise"), py::arg("qec_rounds"), py::arg("shots"),
           py::arg("seed") = py::none(),
           py::arg("erasure_selection") = qerasure::ErasureQubitSelection::ALL_QUBITS,
           py::arg("erasable_qubits") = std::vector<std::size_t>{})
      .def_readonly("code", &qerasure::ErasureSimParams::code)
      .def_readonly("noise", &qerasure::ErasureSimParams::noise)
      .def_readonly("qec_rounds", &qerasure::ErasureSimParams::qec_rounds)
      .def_readonly("shots", &qerasure::ErasureSimParams::shots)
      .def_readonly("seed", &qerasure::ErasureSimParams::seed)
      .def_readonly("erasure_selection", &qerasure::ErasureSimParams::erasure_selection)
      .def_readonly("erasable_qubits", &qerasure::ErasureSimParams::erasable_qubits);

  py::enum_<qerasure::EventType>(m, "EventType")
      .value("ERASURE", qerasure::EventType::ERASURE)
      .value("RESET", qerasure::EventType::RESET)
      .value("CHECK_ERROR", qerasure::EventType::CHECK_ERROR)
      .export_values();

  py::class_<qerasure::ErasureSimEvent>(m, "ErasureSimEvent")
      .def_readonly("qubit_idx", &qerasure::ErasureSimEvent::qubit_idx)
      .def_readonly("event_type", &qerasure::ErasureSimEvent::event_type);

  py::class_<qerasure::ErasureSimResult>(m, "ErasureSimResult")
      .def_readonly("qec_rounds", &qerasure::ErasureSimResult::qec_rounds)
      .def_readonly("sparse_erasures", &qerasure::ErasureSimResult::sparse_erasures)
      .def_readonly("erasure_timestep_offsets", &qerasure::ErasureSimResult::erasure_timestep_offsets);

  py::class_<qerasure::ErasureSimulator>(m, "ErasureSimulator")
      .def(py::init<qerasure::ErasureSimParams>(), py::arg("params"))
      .def("simulate", &qerasure::ErasureSimulator::simulate);

  py::enum_<qerasure::PauliError>(m, "PauliError")
      .value("NO_ERROR", qerasure::PauliError::NO_ERROR)
      .value("X_ERROR", qerasure::PauliError::X_ERROR)
      .value("Z_ERROR", qerasure::PauliError::Z_ERROR)
      .value("Y_ERROR", qerasure::PauliError::Y_ERROR)
      .value("DEPOLARIZE", qerasure::PauliError::DEPOLARIZE)
      .export_values();

  py::enum_<qerasure::LoweredEventOrigin>(m, "LoweredEventOrigin")
      .value("SPREAD", qerasure::LoweredEventOrigin::SPREAD)
      .value("RESET", qerasure::LoweredEventOrigin::RESET)
      .export_values();

  py::enum_<qerasure::PartnerSlot>(m, "PartnerSlot")
      .value("X_1", qerasure::PartnerSlot::X_1)
      .value("X_2", qerasure::PartnerSlot::X_2)
      .value("Z_1", qerasure::PartnerSlot::Z_1)
      .value("Z_2", qerasure::PartnerSlot::Z_2)
      .export_values();

  py::enum_<qerasure::SpreadInstructionType>(m, "SpreadInstructionType")
      .value("X_ERROR", qerasure::SpreadInstructionType::X_ERROR)
      .value("Y_ERROR", qerasure::SpreadInstructionType::Y_ERROR)
      .value("Z_ERROR", qerasure::SpreadInstructionType::Z_ERROR)
      .value("DEPOLARIZE1", qerasure::SpreadInstructionType::DEPOLARIZE1)
      .value("COND_X_ERROR", qerasure::SpreadInstructionType::COND_X_ERROR)
      .value("COND_Y_ERROR", qerasure::SpreadInstructionType::COND_Y_ERROR)
      .value("COND_Z_ERROR", qerasure::SpreadInstructionType::COND_Z_ERROR)
      .value("ELSE_X_ERROR", qerasure::SpreadInstructionType::ELSE_X_ERROR)
      .value("ELSE_Y_ERROR", qerasure::SpreadInstructionType::ELSE_Y_ERROR)
      .value("ELSE_Z_ERROR", qerasure::SpreadInstructionType::ELSE_Z_ERROR)
      .export_values();

  py::class_<qerasure::SpreadTargetOp>(m, "SpreadTargetOp")
      .def(py::init<>())
      .def(py::init<qerasure::PauliError, qerasure::PartnerSlot>(), py::arg("error_type"),
           py::arg("slot"))
      .def_readwrite("error_type", &qerasure::SpreadTargetOp::error_type)
      .def_readwrite("slot", &qerasure::SpreadTargetOp::slot);

  py::class_<qerasure::SpreadInstruction>(m, "SpreadInstruction")
      .def(py::init<>())
      .def_readonly("type", &qerasure::SpreadInstruction::type)
      .def_readonly("probability", &qerasure::SpreadInstruction::probability)
      .def_readonly("target_slot", &qerasure::SpreadInstruction::target_slot);

  py::class_<qerasure::SpreadProgram>(m, "SpreadProgram")
      .def(py::init<>())
      .def("append", &qerasure::SpreadProgram::append, py::arg("stim_like_program"))
      .def("add_instruction",
           py::overload_cast<qerasure::SpreadInstructionType, double, qerasure::PartnerSlot>(
               &qerasure::SpreadProgram::add_instruction),
           py::arg("type"), py::arg("probability"), py::arg("target"))
      .def("add_x_error", &qerasure::SpreadProgram::add_x_error, py::arg("probability"),
           py::arg("target"))
      .def("add_y_error", &qerasure::SpreadProgram::add_y_error, py::arg("probability"),
           py::arg("target"))
      .def("add_z_error", &qerasure::SpreadProgram::add_z_error, py::arg("probability"),
           py::arg("target"))
      .def("add_depolarize1", &qerasure::SpreadProgram::add_depolarize1, py::arg("probability"),
           py::arg("target"))
      .def("add_cond_x_error", &qerasure::SpreadProgram::add_cond_x_error, py::arg("probability"),
           py::arg("target"))
      .def("add_cond_y_error", &qerasure::SpreadProgram::add_cond_y_error, py::arg("probability"),
           py::arg("target"))
      .def("add_cond_z_error", &qerasure::SpreadProgram::add_cond_z_error, py::arg("probability"),
           py::arg("target"))
      .def("add_else_x_error", &qerasure::SpreadProgram::add_else_x_error, py::arg("probability"),
           py::arg("target"))
      .def("add_else_y_error", &qerasure::SpreadProgram::add_else_y_error, py::arg("probability"),
           py::arg("target"))
      .def("add_else_z_error", &qerasure::SpreadProgram::add_else_z_error, py::arg("probability"),
           py::arg("target"))
      .def("add_error_channel", &qerasure::SpreadProgram::add_error_channel, py::arg("probability"),
           py::arg("targets"))
      .def("add_correlated_error", &qerasure::SpreadProgram::add_correlated_error,
           py::arg("probability"), py::arg("targets"))
      .def("add_else_correlated_error", &qerasure::SpreadProgram::add_else_correlated_error,
           py::arg("probability"), py::arg("targets"))
      .def_readwrite("instructions", &qerasure::SpreadProgram::instructions);

  py::class_<qerasure::LoweredErrorParams>(m, "LoweredErrorParams")
      .def(py::init<>())
      .def_readwrite("error_type", &qerasure::LoweredErrorParams::error_type)
      .def_readwrite("probability", &qerasure::LoweredErrorParams::probability);

  py::class_<qerasure::LoweredErrorEvent>(m, "LoweredErrorEvent")
      .def_readonly("qubit_idx", &qerasure::LoweredErrorEvent::qubit_idx)
      .def_readonly("error_type", &qerasure::LoweredErrorEvent::error_type)
      .def_readonly("origin", &qerasure::LoweredErrorEvent::origin);

  py::class_<qerasure::LoweringParams>(m, "LoweringParams")
      .def(py::init<const qerasure::SpreadProgram&>(), py::arg("default_program"))
      .def(py::init<const qerasure::SpreadProgram&, const qerasure::LoweredErrorParams&>(),
           py::arg("default_program"), py::arg("reset"))
      .def(py::init<const qerasure::LoweredErrorParams&, const qerasure::LoweredErrorParams&>(),
           py::arg("reset"), py::arg("ancillas"))
      .def(py::init<const qerasure::LoweredErrorParams&, const qerasure::LoweredErrorParams&,
                    const qerasure::LoweredErrorParams&>(),
           py::arg("reset"), py::arg("x_ancillas"), py::arg("z_ancillas"))
      .def(py::init<const qerasure::LoweredErrorParams&,
                    const std::pair<qerasure::LoweredErrorParams, qerasure::LoweredErrorParams>&,
                    const std::pair<qerasure::LoweredErrorParams, qerasure::LoweredErrorParams>&>(),
           py::arg("reset"), py::arg("x_ancillas"), py::arg("z_ancillas"))
      .def("set_default_data_program", &qerasure::LoweringParams::set_default_data_program,
           py::arg("program"))
      .def("set_data_qubit_program", &qerasure::LoweringParams::set_data_qubit_program,
           py::arg("data_qubit_idx"), py::arg("program"))
      .def_readwrite("reset_params_", &qerasure::LoweringParams::reset_params_)
      .def_readwrite("x_ancilla_params_", &qerasure::LoweringParams::x_ancilla_params_)
      .def_readwrite("z_ancilla_params_", &qerasure::LoweringParams::z_ancilla_params_)
      .def_readwrite("default_data_program", &qerasure::LoweringParams::default_data_program)
      .def_readwrite("per_data_program_overrides",
                     &qerasure::LoweringParams::per_data_program_overrides);

  py::class_<qerasure::LoweringResult>(m, "LoweringResult")
      .def_readonly("qec_rounds", &qerasure::LoweringResult::qec_rounds)
      .def_readonly("sparse_cliffords", &qerasure::LoweringResult::sparse_cliffords)
      .def_readonly("clifford_timestep_offsets", &qerasure::LoweringResult::clifford_timestep_offsets)
      .def_readonly("check_error_round_flags", &qerasure::LoweringResult::check_error_round_flags)
      .def_readonly("erasure_round_flags", &qerasure::LoweringResult::erasure_round_flags)
      .def_readonly("reset_round_qubits", &qerasure::LoweringResult::reset_round_qubits);

  py::class_<qerasure::Lowerer>(m, "Lowerer")
      .def(py::init<const qerasure::RotatedSurfaceCode&, const qerasure::LoweringParams&>(),
           py::arg("code"), py::arg("params"))
      .def("lower", &qerasure::Lowerer::lower);

  m.def("build_surf_stabilizer_circuit", &qerasure::build_surf_stabilizer_circuit,
        py::arg("code"), py::arg("qec_rounds"),
        "Generate a Stim-format rotated-surface stabilizer circuit string.");
  m.def("build_surface_code_stim_circuit", &qerasure::build_surface_code_stim_circuit,
        py::arg("code"), py::arg("qec_rounds"),
        "Generate a Stim-format rotated-surface-code circuit string.");
  m.def("build_logical_stabilizer_circuit", &qerasure::build_logical_stabilizer_circuit,
        py::arg("code"), py::arg("lowering_result"), py::arg("shot_index") = 0,
        "Generate a Stim-format logical stabilizer circuit with injected lowered errors.");
  m.def("build_logically_equivalent_erasure_stim_circuit",
        &qerasure::build_logically_equivalent_erasure_stim_circuit, py::arg("code"),
        py::arg("lowering_result"), py::arg("shot_index") = 0,
        "Generate a Stim-format circuit with deterministic lowered-erasure errors injected by timestep.");
  m.def("build_virtual_decoder_stim_circuit", &qerasure::build_virtual_decoder_stim_circuit,
        py::arg("code"), py::arg("qec_rounds"), py::arg("lowering_params"),
        py::arg("lowering_result"), py::arg("shot_index") = 0,
        py::arg("two_qubit_erasure_probability"),
        py::arg("condition_on_erasure_in_round") = true,
        "Generate a Stim-format virtual decoder circuit with probabilistic spread injection.");
}
