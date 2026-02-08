#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "qerasure/code/code.h"
#include "qerasure/noise/noise.h"

namespace py = pybind11;

PYBIND11_MODULE(qerasure_python, m) {
    m.doc() = "Python bindings for the qerasure code library";

    py::class_<RotatedSurfaceCode>(m, "RotatedSurfaceCode")
        .def(py::init<std::size_t>(), py::arg("distance"))
        .def_property_readonly("distance", &RotatedSurfaceCode::distance)
        .def_property_readonly("num_qubits", &RotatedSurfaceCode::num_qubits)
        .def_property_readonly("gates", [](const RotatedSurfaceCode& code) {
            const auto& gates_flat = code.gates();
            std::size_t distance = code.distance();
            std::size_t gates_per_step = 2 + 3 * (distance - 2) + (distance - 2) * (distance - 2);
            
            py::list result;
            for (size_t step = 0; step < 4; step++) {
                py::list step_gates;
                for (size_t i = 0; i < gates_per_step; i++) {
                    size_t idx = step * gates_per_step + i;
                    if (idx < gates_flat.size()) {
                        const auto& gate = gates_flat[idx];
                        step_gates.append(py::make_tuple(gate.first, gate.second));
                    }
                }
                result.append(step_gates);
            }
            return result;
        })
        .def_property_readonly("coord_to_index", [](const RotatedSurfaceCode& code) {
            const auto& m = code.coord_to_index();
            py::dict d;
            for (const auto& kv : m) {
                py::tuple coord = py::make_tuple(kv.first.first, kv.first.second);
                d[coord] = kv.second;
            }
            return d;
        })
        .def_property_readonly("index_to_coord", [](const RotatedSurfaceCode& code) {
            const auto& m = code.index_to_coord();
            py::dict d;
            for (const auto& kv : m) {
                py::tuple coord = py::make_tuple(kv.second.first, kv.second.second);
                d[py::int_(kv.first)] = coord;
            }
            return d;
        })
        .def_property_readonly("x_anc_offset", &RotatedSurfaceCode::x_anc_offset)
        .def_property_readonly("z_anc_offset", &RotatedSurfaceCode::z_anc_offset);

        py::class_<NoiseParams>(m, "NoiseParams")
            .def(py::init<>())
            .def_readonly("p_single_qubit_depolarize", &NoiseParams::p_single_qubit_depolarize)
            .def_readonly("p_two_qubit_depolarize", &NoiseParams::p_two_qubit_depolarize)
            .def_readonly("p_measurement_error", &NoiseParams::p_measurement_error)
            .def_readonly("p_single_qubit_erasure", &NoiseParams::p_single_qubit_erasure)
            .def_readonly("p_two_qubit_erasure", &NoiseParams::p_two_qubit_erasure)
            .def_readonly("p_erasure_check_error", &NoiseParams::p_erasure_check_error);

        m.def("build_noise_model", &build_noise_model, py::arg("params") = NoiseParams{});
}
