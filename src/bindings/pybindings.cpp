#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "qerasure/code/code.h"

namespace py = pybind11;

PYBIND11_MODULE(qerasure_python, m) {
    m.doc() = "Python bindings for the qerasure code library";

    py::class_<RotatedSurfaceCode>(m, "RotatedSurfaceCode")
        .def(py::init<std::size_t>(), py::arg("distance"))
        .def_property_readonly("distance", &RotatedSurfaceCode::distance)
        .def_property_readonly("num_qubits", &RotatedSurfaceCode::num_qubits)
        .def_property_readonly("stabilizers", &RotatedSurfaceCode::stabilizers)
        .def_property_readonly("gates", &RotatedSurfaceCode::gates)
        .def_property_readonly("coord_to_index", [](const RotatedSurfaceCode& code) {
            const auto& m = code.coord_to_index();
            py::dict d;
            for (const auto& kv : m) {
                d[py::tuple(py::cast(kv.first))] = kv.second;
            }
            return d;
        })
        .def_property_readonly("index_to_coord", &RotatedSurfaceCode::index_to_coord)
        .def_property_readonly("x_anc_offset", &RotatedSurfaceCode::x_anc_offset)
        .def_property_readonly("z_anc_offset", &RotatedSurfaceCode::z_anc_offset);
}
