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
          py::arg("verbose") = false)
      .def(
          "calibration_rows",
          [](const decode::RailSurfaceDemBuilder& decoder,
             const std::vector<uint8_t>& check_results,
             const std::vector<uint8_t>& detector_samples) {
            std::vector<decode::RailSurfaceDemBuilder::CalibrationRow> rows;
            {
              py::gil_scoped_release release;
              rows = decoder.calibration_rows(&check_results, &detector_samples);
            }
            py::list out;
            for (const auto& row : rows) {
              py::dict item;
              item["check_event_index"] = py::int_(row.check_event_index);
              item["data_qubit"] = py::int_(row.data_qubit);
              item["check_op_index"] = py::int_(row.check_op_index);
              item["check_round"] = py::int_(row.check_round);
              item["onset_op_index"] = py::int_(row.onset_op_index);
              item["prior_mass"] = py::float_(row.prior_mass);
              item["evidence_likelihood"] = py::float_(row.evidence_likelihood);
              item["posterior_mass"] = py::float_(row.posterior_mass);
              item["schedule_type"] = py::int_(row.schedule_type);
              item["boundary_data_qubit"] = py::bool_(row.boundary_data_qubit);
              out.append(std::move(item));
            }
            return out;
          },
          py::arg("check_results"),
          py::arg("detector_samples"));
}

}  // namespace qerasure::python_bindings
