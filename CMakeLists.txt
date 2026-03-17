cmake_minimum_required(VERSION 3.16)
project(qerasure LANGUAGES CXX)

if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Required when linking static libraries into the pybind11 shared module.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)
option(QERASURE_BUILD_PYTHON "Build Python extension module (qerasure_python)." OFF)
set(QERASURE_STIM_SOURCE_DIR "" CACHE PATH "Path to a local Stim source checkout (optional).")
set(QERASURE_PYBIND11_SOURCE_DIR "" CACHE PATH "Path to a local pybind11 source checkout (optional).")

if(QERASURE_STIM_SOURCE_DIR AND EXISTS "${QERASURE_STIM_SOURCE_DIR}/file_lists/source_files_no_main")
  set(stim_SOURCE_DIR "${QERASURE_STIM_SOURCE_DIR}")
else()
  FetchContent_Declare(
    stim
    GIT_REPOSITORY https://github.com/quantumlib/Stim.git
    GIT_TAG v1.15.0
  )
  FetchContent_GetProperties(stim)
  if(NOT stim_POPULATED)
    FetchContent_Populate(stim)
  endif()
endif()

file(STRINGS "${stim_SOURCE_DIR}/file_lists/source_files_no_main" STIM_SOURCE_FILES_REL)
list(TRANSFORM STIM_SOURCE_FILES_REL PREPEND "${stim_SOURCE_DIR}/")

add_library(qerasure_stim STATIC ${STIM_SOURCE_FILES_REL})
target_include_directories(qerasure_stim PUBLIC "${stim_SOURCE_DIR}/src")
target_compile_features(qerasure_stim PUBLIC cxx_std_20)
# Ensure cross-module RTTI/typeinfo for Stim types is visible to pybind casters.
set_target_properties(qerasure_stim PROPERTIES
  CXX_VISIBILITY_PRESET default
  VISIBILITY_INLINES_HIDDEN OFF
)

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
set_target_properties(qerasure PROPERTIES
  CXX_VISIBILITY_PRESET default
  VISIBILITY_INLINES_HIDDEN OFF
)

target_include_directories(qerasure
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
)

set(QERASURE_BENCHMARK_SOURCES
    benchmarks/bench_stream_sampler.cpp
    benchmarks/bench_stream_decode.cpp
    benchmarks/bench_surf_dem_profile.cpp
)

foreach(benchmark_source IN LISTS QERASURE_BENCHMARK_SOURCES)
  if(NOT EXISTS "${CMAKE_SOURCE_DIR}/${benchmark_source}")
    message(STATUS "Skipping missing benchmark source: ${benchmark_source}")
    continue()
  endif()
  get_filename_component(benchmark_name "${benchmark_source}" NAME_WE)
  add_executable(${benchmark_name}
      ${benchmark_source}
  )
  target_link_libraries(${benchmark_name} PRIVATE qerasure qerasure_stim)
endforeach()

include(CTest)
if(BUILD_TESTING)
  find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
  if(EXISTS "${CMAKE_SOURCE_DIR}/.venv/bin/python")
    set(QERASURE_TEST_PYTHON_EXECUTABLE "${CMAKE_SOURCE_DIR}/.venv/bin/python")
  else()
    set(QERASURE_TEST_PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")
  endif()
  set(QERASURE_TEST_SOURCES
      tests/hmm_decoder_stream_test.cpp
      tests/skippable_reweight_test.cpp
      tests/surf_dem_builder_trace_test.cpp
      tests/surf_tail_correction_test.cpp
      tests/surf_rounds_per_check_test.cpp
      tests/surf_post_clifford_pauli_test.cpp
      tests/stream_decode_dem_test.cpp
      tests/observable_flip_artifact_test.cpp
      tests/observable_flip_artifact_ancilla_test.cpp
      tests/observable_flip_artifact_ancilla_no_spread_test.cpp
  )

  foreach(test_source IN LISTS QERASURE_TEST_SOURCES)
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/${test_source}")
      message(STATUS "Skipping missing test source: ${test_source}")
      continue()
    endif()
    get_filename_component(test_name "${test_source}" NAME_WE)
    add_executable(${test_name}
        ${test_source}
    )
    target_link_libraries(${test_name} PRIVATE qerasure qerasure_stim)
    add_test(NAME ${test_name} COMMAND ${test_name})
  endforeach()

  add_test(
    NAME python_surface_batch_decode_failure_artifact_test
    COMMAND ${QERASURE_TEST_PYTHON_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/tests/python_surface_batch_decode_failure_artifact_test.py
  )
endif()

if(QERASURE_BUILD_PYTHON)
  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    if(QERASURE_PYBIND11_SOURCE_DIR AND EXISTS "${QERASURE_PYBIND11_SOURCE_DIR}/CMakeLists.txt")
      add_subdirectory("${QERASURE_PYBIND11_SOURCE_DIR}" pybind11-local)
    elseif(EXISTS "${CMAKE_BINARY_DIR}/_deps/pybind11-src/CMakeLists.txt")
      add_subdirectory("${CMAKE_BINARY_DIR}/_deps/pybind11-src" pybind11-local)
    else()
      FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.0
      )
      FetchContent_MakeAvailable(pybind11)
    endif()
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
  # Important for returning Stim C++ types (stim::Circuit) into Python across
  # extension-module boundaries on some toolchains.
  set_target_properties(qerasure_python PROPERTIES
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF
  )
  target_compile_options(qerasure_python PRIVATE -fvisibility=default)
endif()
