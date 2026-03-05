#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <stdexcept>

#include "core/simulator/stream_sampler.h"

namespace py = pybind11;

namespace qerasure::python_bindings {

void bind_stream_sampler(py::module_& m) {
	py::class_<simulator::StreamSampler>(m, "StreamSampler")
		.def(py::init<const circuit::CompiledErasureProgram&>(), py::arg("program"),
			 py::keep_alive<1, 2>())
		.def(
			"sample_syndromes",
			[](simulator::StreamSampler& sampler, uint32_t num_shots, uint32_t seed,
			   uint32_t num_threads) {
				const simulator::SyndromeSampleBatch sampled =
					sampler.sample_syndromes(num_shots, seed, num_threads);

				py::array_t<uint8_t> dets(
					{sampled.num_shots, sampled.num_detectors});
				py::array_t<uint8_t> obs(
					{sampled.num_shots, sampled.num_observables});
				py::array_t<uint8_t> checks(
					{sampled.num_shots, sampled.num_checks});

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

				return py::make_tuple(std::move(dets), std::move(obs), std::move(checks));
			},
			py::arg("num_shots"), py::arg("seed"), py::arg("num_threads") = 1)
		.def(
			"sample_with_callback",
			[](simulator::StreamSampler& sampler, uint32_t num_shots, uint32_t seed,
			   py::object callback, uint32_t num_threads) {
				if (callback.is_none()) {
					py::gil_scoped_release release;
					sampler.sample_with_callback(
						num_shots,
						seed,
						[](const stim::Circuit&, const std::vector<uint8_t>&) {},
						num_threads);
					return;
				}

				if (num_threads != 1) {
					throw std::invalid_argument(
						"Python callbacks require num_threads=1 in StreamSampler.sample.");
				}

				py::function python_callback = callback.cast<py::function>();
				sampler.sample_with_callback(
					num_shots,
					seed,
					[python_callback](const stim::Circuit& circuit,
									  const std::vector<uint8_t>& check_results) {
						py::gil_scoped_acquire acquire;
						python_callback(circuit.str(), check_results);
					},
					num_threads);
			},
			py::arg("num_shots"), py::arg("seed"), py::arg("callback") = py::none(),
			py::arg("num_threads") = 1);
}

}  // namespace qerasure::python_bindings
