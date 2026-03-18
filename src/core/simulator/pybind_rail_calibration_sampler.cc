#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>

#include "core/simulator/rail_calibration_sampler.h"

namespace py = pybind11;

namespace qerasure::python_bindings {

void bind_rail_calibration_sampler(py::module_& m) {
  py::class_<simulator::RailCalibrationSampler>(m, "RailCalibrationSampler")
      .def(
          py::init<const circuit::RailSurfaceCompiledProgram&>(),
          py::arg("program"),
          py::keep_alive<1, 2>())
      .def(
          "sample_syndromes",
          [](simulator::RailCalibrationSampler& sampler,
             uint32_t num_shots,
             uint32_t seed,
             uint32_t num_threads) {
            simulator::RailCalibrationSampleBatch sampled;
            {
              py::gil_scoped_release release;
              sampled = sampler.sample_syndromes(num_shots, seed, num_threads);
            }

            py::array_t<uint8_t> dets({sampled.num_shots, sampled.num_detectors});
            py::array_t<uint8_t> obs({sampled.num_shots, sampled.num_observables});
            py::array_t<uint8_t> checks({sampled.num_shots, sampled.num_checks});
            py::array_t<int32_t> onset_ops({sampled.num_shots, sampled.num_checks});

            std::memcpy(
                dets.mutable_data(),
                sampled.detector_samples.data(),
                sampled.detector_samples.size() * sizeof(uint8_t));
            std::memcpy(
                obs.mutable_data(),
                sampled.observable_flips.data(),
                sampled.observable_flips.size() * sizeof(uint8_t));
            std::memcpy(
                checks.mutable_data(),
                sampled.check_flags.data(),
                sampled.check_flags.size() * sizeof(uint8_t));
            std::memcpy(
                onset_ops.mutable_data(),
                sampled.latent_onset_ops.data(),
                sampled.latent_onset_ops.size() * sizeof(int32_t));
            return py::make_tuple(
                std::move(dets),
                std::move(obs),
                std::move(checks),
                std::move(onset_ops));
          },
          py::arg("num_shots"),
          py::arg("seed"),
          py::arg("num_threads") = 1)
      .def(
          "sample_exact_shot",
          [](simulator::RailCalibrationSampler& sampler, uint32_t seed, uint32_t shot) {
            auto sampled = sampler.sample_exact_shot(seed, shot);
            return py::make_tuple(
                std::get<0>(sampled).str(),
                std::get<1>(sampled),
                std::get<2>(sampled));
          },
          py::arg("seed"),
          py::arg("shot"));
}

}  // namespace qerasure::python_bindings
