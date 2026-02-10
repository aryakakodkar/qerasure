#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "qerasure/core/code/rotated_surface_code.h"
#include "qerasure/core/noise/noise_params.h"
#include "qerasure/core/sim/erasure_simulator.h"

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

  py::class_<qerasure::ErasureSimParams>(m, "ErasureSimParams")
      .def(py::init<const qerasure::RotatedSurfaceCode&, const qerasure::NoiseParams&, std::size_t,
                    std::size_t, std::optional<std::uint32_t>>(),
           py::arg("code"), py::arg("noise"), py::arg("qec_rounds"), py::arg("shots"),
           py::arg("seed") = py::none())
      .def_readonly("code", &qerasure::ErasureSimParams::code)
      .def_readonly("noise", &qerasure::ErasureSimParams::noise)
      .def_readonly("qec_rounds", &qerasure::ErasureSimParams::qec_rounds)
      .def_readonly("shots", &qerasure::ErasureSimParams::shots)
      .def_readonly("seed", &qerasure::ErasureSimParams::seed);

  py::enum_<qerasure::EventType>(m, "EventType")
      .value("ERASURE", qerasure::EventType::ERASURE)
      .value("RESET", qerasure::EventType::RESET)
      .value("CHECK_ERROR", qerasure::EventType::CHECK_ERROR)
      .export_values();

  py::class_<qerasure::ErasureSimEvent>(m, "ErasureSimEvent")
      .def_readonly("qubit_idx", &qerasure::ErasureSimEvent::qubit_idx)
      .def_readonly("event_type", &qerasure::ErasureSimEvent::event_type);

  py::class_<qerasure::ErasureSimResult>(m, "ErasureSimResult")
      .def_readonly("sparse_erasures", &qerasure::ErasureSimResult::sparse_erasures)
      .def_readonly("erasure_timestep_offsets", &qerasure::ErasureSimResult::erasure_timestep_offsets);

  py::class_<qerasure::ErasureSimulator>(m, "ErasureSimulator")
      .def(py::init<qerasure::ErasureSimParams>(), py::arg("params"))
      .def("simulate", &qerasure::ErasureSimulator::simulate);
}
