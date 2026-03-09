#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>

#include "core/decode/surf_dem_builder.h"

namespace py = pybind11;

namespace qerasure::python_bindings {

void bind_surf_dem_builder(py::module_& m) {
	py::class_<decode::SurfDemBuilder>(m, "SurfDemBuilder")
		.def(py::init<const circuit::CompiledErasureProgram&>(), py::arg("program"),
			 py::keep_alive<1, 2>())
		.def(
			"build_decoded_circuit",
			[](const decode::SurfDemBuilder& decoder,
			   const std::vector<uint8_t>& check_results,
			   bool verbose) {
				py::gil_scoped_release release;
				return decoder.build_decoded_circuit(&check_results, verbose).str();
			},
			py::arg("check_results"), py::arg("verbose") = false)
		.def(
			"build_decoded_circuit_text",
			[](const decode::SurfDemBuilder& decoder,
			   const std::vector<uint8_t>& check_results,
			   bool verbose) {
				py::gil_scoped_release release;
				return decoder.build_decoded_circuit_text(&check_results, verbose);
			},
			py::arg("check_results"), py::arg("verbose") = false)
		.def(
			"find_probability_violations",
			[](const decode::SurfDemBuilder& decoder,
			   const std::vector<uint8_t>& check_results) {
				decode::SpreadInjectionBuckets buckets;
				{
					py::gil_scoped_release release;
					buckets = decoder.compute_spread_injections(&check_results, /*verbose=*/false);
				}
				std::vector<py::dict> out;
				for (uint32_t op_index = 0; op_index < buckets.size(); ++op_index) {
					for (const auto& event : buckets[op_index]) {
						const double p_x = std::clamp(event.p_x, 0.0, 1.0);
						const double p_y = std::clamp(event.p_y, 0.0, 1.0);
						const double p_z = std::clamp(event.p_z, 0.0, 1.0);
						const double total = p_x + p_y + p_z;
						if (total <= 1.0) {
							continue;
						}
						py::dict v;
						v["op_index"] = py::int_(event.op_index);
						v["target_qubit"] = py::int_(event.target_qubit);
						v["p_x"] = py::float_(p_x);
						v["p_y"] = py::float_(p_y);
						v["p_z"] = py::float_(p_z);
						v["sum"] = py::float_(total);
						out.push_back(std::move(v));
					}
				}
				return out;
			},
			py::arg("check_results"));
}

}  // namespace qerasure::python_bindings
