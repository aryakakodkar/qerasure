cmake_minimum_required(VERSION 3.16)
project(qerasure LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_GPERFTOOLS "Enable gperftools profiling support" OFF)

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
)

target_include_directories(qerasure
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

add_executable(run_sim
    apps/run_sim.cpp
)
target_link_libraries(run_sim PRIVATE qerasure)

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

include(CTest)
if(BUILD_TESTING)
  add_executable(smoke_sim_test
      tests/smoke_sim_test.cpp
  )
  target_link_libraries(smoke_sim_test PRIVATE qerasure)
  add_test(NAME smoke_sim_test COMMAND smoke_sim_test)
endif()

include(FetchContent)
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
