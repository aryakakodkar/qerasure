cmake_minimum_required(VERSION 3.16)
project(qerasure LANGUAGES CXX)

if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_GPERFTOOLS "Enable gperftools profiling support" OFF)

include(FetchContent)

# Pull Stim automatically and link against its C++ library for translation support.
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

set(QERASURE_EXTRA_LIBS "")
set(QERASURE_EXTRA_INCLUDE_DIRS "")
if(ENABLE_GPERFTOOLS)
  find_library(PROFILER_LIB profiler)
  find_path(GPERFTOOLS_INCLUDE_DIR gperftools/profiler.h
      PATHS /opt/homebrew/include /usr/local/include /usr/include)

  if(PROFILER_LIB AND GPERFTOOLS_INCLUDE_DIR)
    list(APPEND QERASURE_EXTRA_LIBS ${PROFILER_LIB})
    list(APPEND QERASURE_EXTRA_INCLUDE_DIRS ${GPERFTOOLS_INCLUDE_DIR})
    message(STATUS "gperftools enabled: ${PROFILER_LIB}")
  else()
    message(FATAL_ERROR "ENABLE_GPERFTOOLS=ON but gperftools was not found")
  endif()
endif()

add_library(qerasure
    src/core/code/rotated_surface_code.cpp
    src/core/noise/noise_params.cpp
    src/core/sim/erasure_simulator.cpp
    src/core/lowering/lowering.cpp
    src/core/translation/stim_translation.cpp
    src/core/translation/virtual_stim_translation.cpp
)
target_link_libraries(qerasure PRIVATE qerasure_stim)
target_compile_features(qerasure PUBLIC cxx_std_20)

target_include_directories(qerasure
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

if(EXISTS "${CMAKE_SOURCE_DIR}/apps/run_sim.cpp")
  add_executable(run_sim
      apps/run_sim.cpp
  )
  target_link_libraries(run_sim PRIVATE qerasure)
endif()

add_executable(bench_erasure_sim
    benchmarks/bench_erasure_sim.cpp
)
target_link_libraries(bench_erasure_sim PRIVATE qerasure ${QERASURE_EXTRA_LIBS})
if(QERASURE_EXTRA_INCLUDE_DIRS)
  target_include_directories(bench_erasure_sim PRIVATE ${QERASURE_EXTRA_INCLUDE_DIRS})
endif()

add_executable(bench_erasure_and_lower
    benchmarks/bench_erasure_and_lower.cpp
)
target_link_libraries(bench_erasure_and_lower PRIVATE qerasure ${QERASURE_EXTRA_LIBS})
if(QERASURE_EXTRA_INCLUDE_DIRS)
  target_include_directories(bench_erasure_and_lower PRIVATE ${QERASURE_EXTRA_INCLUDE_DIRS})
endif()

add_executable(bench_stim_translation
    benchmarks/bench_stim_translation.cpp
)
target_link_libraries(bench_stim_translation PRIVATE qerasure qerasure_stim)

include(CTest)
if(BUILD_TESTING)
  add_executable(smoke_sim_test
      tests/smoke_sim_test.cpp
  )
  target_link_libraries(smoke_sim_test PRIVATE qerasure)
  add_test(NAME smoke_sim_test COMMAND smoke_sim_test)

  add_executable(lowering_standard_test
      tests/lowering_standard_test.cpp
  )
  target_link_libraries(lowering_standard_test PRIVATE qerasure)
  add_test(NAME lowering_standard_test COMMAND lowering_standard_test)

  add_executable(statistical_convergence_test
      tests/statistical_convergence_test.cpp
  )
  target_link_libraries(statistical_convergence_test PRIVATE qerasure)

  add_executable(stim_translation_test
      tests/stim_translation_test.cpp
  )
  target_link_libraries(stim_translation_test PRIVATE qerasure)
  add_test(NAME stim_translation_test COMMAND stim_translation_test)
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
    src/interop/python/pybindings.cpp
)

target_link_libraries(qerasure_python PRIVATE qerasure)
target_include_directories(qerasure_python PRIVATE ${CMAKE_SOURCE_DIR}/include)
