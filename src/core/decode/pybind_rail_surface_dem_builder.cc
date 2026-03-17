#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/decode/rail_surface_dem_builder.h"

namespace py = pybind11;

namespace qerasure::python_bindings {

void bind_rail_surface_dem_builder(py::module_& m) {
  py::class_<decode::RailSurfaceDemBuilder>(m, "RailSurfaceDemBuilder")
      .def(
          py::init<const circuit::RailSurfaceCompiledProgram&>(),
          py::arg("program"),
          py::keep_alive<1, 2>())
      .def(
          "build_decoded_circuit",
          [](const decode::RailSurfaceDemBuilder& decoder,
             const std::vector<uint8_t>& check_results,
             const std::vector<uint8_t>& detector_samples,
             bool verbose) {
            py::gil_scoped_release release;
            return decoder.build_decoded_circuit(&check_results, &detector_samples, verbose);
          },
          py::arg("check_results"),
          py::arg("detector_samples"),
          py::arg("verbose") = false)
      .def(
          "build_decoded_circuit_text",
          [](const decode::RailSurfaceDemBuilder& decoder,
             const std::vector<uint8_t>& check_results,
             const std::vector<uint8_t>& detector_samples,
             bool verbose) {
            py::gil_scoped_release release;
            return decoder.build_decoded_circuit_text(&check_results, &detector_samples, verbose);
          },
          py::arg("check_results"),
          py::arg("detector_samples"),
          py::arg("verbose") = false);
}

}  // namespace qerasure::python_bindings
