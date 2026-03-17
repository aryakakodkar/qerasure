#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace qerasure::python_bindings {
void bind_circuit(py::module_& m);
void bind_compile(py::module_& m);
void bind_rail_surface_compile(py::module_& m);
void bind_surf_gen(py::module_& m);
void bind_surf_dem_builder(py::module_& m);
void bind_rail_surface_dem_builder(py::module_& m);
void bind_stream_sampler(py::module_& m);
void bind_rail_stream_sampler(py::module_& m);
}  // namespace qerasure::python_bindings

PYBIND11_MODULE(qerasure_python, m) {
  m.doc() = "Python bindings for qerasure circuit-model functionality";

  qerasure::python_bindings::bind_circuit(m);
  qerasure::python_bindings::bind_compile(m);
  qerasure::python_bindings::bind_rail_surface_compile(m);
  qerasure::python_bindings::bind_surf_gen(m);
  qerasure::python_bindings::bind_surf_dem_builder(m);
  qerasure::python_bindings::bind_rail_surface_dem_builder(m);
  qerasure::python_bindings::bind_stream_sampler(m);
  qerasure::python_bindings::bind_rail_stream_sampler(m);
}
