#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "qerasure/code/code.h"
#include "qerasure/noise/noise.h"
#include "qerasure/simulators/erasure_simulator.h"

namespace py = pybind11;

namespace {

py::list gates_to_python(const RotatedSurfaceCode& code) {
    const auto& gates_flat = code.gates();
    const std::size_t d = code.distance();
    const std::size_t gates_per_step = 2 + 3 * (d - 2) + (d - 2) * (d - 2);

    py::list result;
    for (std::size_t s = 0; s < 4; s++) {
        py::list step_gates;
        for (std::size_t j = 0; j < gates_per_step; j++) {
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

py::dict coord_to_index_to_python(const RotatedSurfaceCode& code) {
    const auto& m = code.coord_to_index();
    py::dict out;
    for (const auto& kv : m) {
        py::tuple coord = py::make_tuple(kv.first.first, kv.first.second);
        out[coord] = kv.second;
    }
    return out;
}

py::dict index_to_coord_to_python(const RotatedSurfaceCode& code) {
    const auto& m = code.index_to_coord();
    py::dict out;
    for (const auto& kv : m) {
        py::tuple coord = py::make_tuple(kv.second.first, kv.second.second);
        out[py::int_(kv.first)] = coord;
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(qerasure_python, m) {
    m.doc() = "Python bindings for the qerasure code library";

    py::class_<RotatedSurfaceCode>(m, "RotatedSurfaceCode")
        .def(py::init<std::size_t>(), py::arg("distance"))
        .def_property_readonly("distance", &RotatedSurfaceCode::distance)
        .def_property_readonly("num_qubits", &RotatedSurfaceCode::num_qubits)
        .def_property_readonly("gates", &gates_to_python)
        .def_property_readonly("coord_to_index", &coord_to_index_to_python)
        .def_property_readonly("index_to_coord", &index_to_coord_to_python)
        .def_property_readonly("partner_map", &RotatedSurfaceCode::partner_map)
        .def_property_readonly("x_anc_offset", &RotatedSurfaceCode::x_anc_offset)
        .def_property_readonly("z_anc_offset", &RotatedSurfaceCode::z_anc_offset);

    py::class_<NoiseParams>(m, "NoiseParams")
        .def(py::init<>())
        .def("set", &NoiseParams::set, py::arg("key"), py::arg("value"))
        .def("get", &NoiseParams::get, py::arg("key"))
        .def("__repr__", [](const NoiseParams& params) {
            std::string repr = "NoiseParams(";
            std::unordered_map<std::string, double> probs = params.get_all();
            for (const auto& kv : probs) {
                repr += kv.first + "=" + std::to_string(kv.second) + ", ";
            }
            if (!probs.empty()) {
                repr.pop_back(); // Remove last space
                repr.pop_back(); // Remove last comma
            }
            repr += ")";
            return repr;
        });

    py::class_<ErasureSimParams>(m, "ErasureSimParams")
        .def(py::init<const RotatedSurfaceCode&, const NoiseParams&, std::size_t, std::size_t>(),
             py::arg("code"), py::arg("noise"), py::arg("qec_rounds"), py::arg("shots"))
        .def_readonly("code", &ErasureSimParams::code)
        .def_readonly("noise", &ErasureSimParams::noise)
        .def_readonly("qec_rounds", &ErasureSimParams::qec_rounds)
        .def_readonly("shots", &ErasureSimParams::shots);

    py::enum_<EventType>(m, "EventType")
        .value("ERASURE", EventType::ERASURE)
        .value("RESET", EventType::RESET)
        .value("CHECK_ERROR", EventType::CHECK_ERROR)
        .export_values();

    py::class_<ErasureSimEvent>(m, "ErasureSimEvent")
        .def_readonly("qubit_idx", &ErasureSimEvent::qubit_idx)
        .def_readonly("event_type", &ErasureSimEvent::event_type);

    py::class_<ErasureSimResult>(m, "ErasureSimResult")
        .def_readonly("sparse_erasures", &ErasureSimResult::sparse_erasures)
        .def_readonly("erasure_timestep_offsets", &ErasureSimResult::erasure_timestep_offsets);

    py::class_<ErasureSimulator>(m, "ErasureSimulator")
        .def(py::init<const ErasureSimParams&>(), py::arg("params"))
        .def("simulate", &ErasureSimulator::simulate);
}
