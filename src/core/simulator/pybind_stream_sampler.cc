#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

#include "core/simulator/stream_sampler.h"

namespace py = pybind11;

namespace qerasure::python_bindings {

void bind_stream_sampler(py::module_& m) {
	py::class_<simulator::StreamSampler>(m, "StreamSampler")
		.def(py::init<const circuit::CompiledErasureProgram&>(), py::arg("program"),
			 py::keep_alive<1, 2>())
		.def(
			"sample",
			[](simulator::StreamSampler& sampler, uint32_t num_shots, uint32_t seed,
			   py::object callback, uint32_t num_threads) {
				if (callback.is_none()) {
					py::gil_scoped_release release;
					sampler.sample(
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
				sampler.sample(
					num_shots,
					seed,
					[python_callback](const stim::Circuit& circuit,
									  const std::vector<uint8_t>& check_results) {
						py::gil_scoped_acquire acquire;
						python_callback(circuit, check_results);
					},
					num_threads);
			},
			py::arg("num_shots"), py::arg("seed"), py::arg("callback") = py::none(),
			py::arg("num_threads") = 1);
}

}  // namespace qerasure::python_bindings
