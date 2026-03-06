cmake_minimum_required(VERSION 3.16)
project(qerasure LANGUAGES CXX)

if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

FetchContent_Declare(
  stim
  GIT_REPOSITORY https://github.com/quantumlib/Stim.git
  GIT_TAG v1.15.0
)
FetchContent_GetProperties(stim)
if(NOT stim_POPULATED)
  FetchContent_Populate(stim)
endif()

file(STRINGS "${stim_SOURCE_DIR}/file_lists/source_files_no_main" STIM_SOURCE_FILES_REL)
list(TRANSFORM STIM_SOURCE_FILES_REL PREPEND "${stim_SOURCE_DIR}/")

add_library(qerasure_stim STATIC ${STIM_SOURCE_FILES_REL})
target_include_directories(qerasure_stim PUBLIC "${stim_SOURCE_DIR}/src")
target_compile_features(qerasure_stim PUBLIC cxx_std_20)

add_library(qerasure
    src/core/code/rotated_surface_code.cpp
    src/core/circuit/circuit.cc
    src/core/circuit/compile.cc
    src/core/decode/surf_dem_builder.cc
    src/core/gen/surf.cc
    src/core/simulator/erasure_sampler.cc
    src/core/simulator/injector.cc
    src/core/simulator/sim_internal_utils.cc
    src/core/simulator/stream_sampler.cc
)
target_link_libraries(qerasure PRIVATE qerasure_stim)
target_compile_features(qerasure PUBLIC cxx_std_20)

target_include_directories(qerasure
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
)

add_executable(erasure_sampler_bench
    benchmarks/erasure_sampler_bench.cpp
)
target_link_libraries(erasure_sampler_bench PRIVATE qerasure)

add_executable(bench_stream_sampler
    benchmarks/bench_stream_sampler.cpp
)
target_link_libraries(bench_stream_sampler PRIVATE qerasure qerasure_stim)

add_executable(bench_stream_decode
    benchmarks/bench_stream_decode.cpp
)
target_link_libraries(bench_stream_decode PRIVATE qerasure qerasure_stim)

include(CTest)
if(BUILD_TESTING)
  add_executable(sampler_injector_trace_test
      tests/sampler_injector_trace_test.cpp
  )
  target_link_libraries(sampler_injector_trace_test PRIVATE qerasure qerasure_stim)
  add_test(NAME sampler_injector_trace_test COMMAND sampler_injector_trace_test)

  add_executable(hmm_decoder_stream_test
      tests/hmm_decoder_stream_test.cpp
  )
  target_link_libraries(hmm_decoder_stream_test PRIVATE qerasure qerasure_stim)
  add_test(NAME hmm_decoder_stream_test COMMAND hmm_decoder_stream_test)

  add_executable(surf_dem_builder_trace_test
      tests/surf_dem_builder_trace_test.cpp
  )
  target_link_libraries(surf_dem_builder_trace_test PRIVATE qerasure qerasure_stim)
  add_test(NAME surf_dem_builder_trace_test COMMAND surf_dem_builder_trace_test)

  add_executable(stream_decode_dem_test
      tests/stream_decode_dem_test.cpp
  )
  target_link_libraries(stream_decode_dem_test PRIVATE qerasure qerasure_stim)
  add_test(NAME stream_decode_dem_test COMMAND stream_decode_dem_test)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/build/_deps/pybind11-src/CMakeLists.txt")
  add_subdirectory("${CMAKE_SOURCE_DIR}/build/_deps/pybind11-src" pybind11-local)
else()
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.0
  )
  FetchContent_MakeAvailable(pybind11)
endif()

pybind11_add_module(qerasure_python
    src/core/circuit/pybind_module.cc
    src/core/circuit/pybind_circuit.cc
    src/core/circuit/pybind_compile.cc
    src/core/decode/pybind_surf_dem_builder.cc
    src/core/gen/pybind_surf.cc
    src/core/simulator/pybind_stream_sampler.cc
)

target_link_libraries(qerasure_python PRIVATE qerasure qerasure_stim)
target_include_directories(qerasure_python PRIVATE ${CMAKE_SOURCE_DIR}/src)
