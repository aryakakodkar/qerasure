cmake_minimum_required(VERSION 3.16)
project(qerasure LANGUAGES CXX)

# Use a modern C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -------------------------
# Library target
# -------------------------
add_library(qerasure
    src/code/code.cpp
    src/noise/noise.cpp
    src/simulators/erasure_simulator.cpp
)

# Let users include headers as: #include "qerasure/code/RotatedSurfaceCode.hpp"
target_include_directories(qerasure
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

# -------------------------
# Executable target
# -------------------------
add_executable(run_sim
    apps/run_sim.cpp
)

target_link_libraries(run_sim
    PRIVATE qerasure
)

# -------------------------
# Benchmark executable with gperftools support
# -------------------------
# Check for gperftools
find_library(PROFILER_LIB profiler)
find_path(GPERFTOOLS_INCLUDE_DIR gperftools/profiler.h
    PATHS /opt/homebrew/include /usr/local/include /usr/include
)
option(ENABLE_GPERFTOOLS "Enable gperftools profiling support" OFF)

if(ENABLE_GPERFTOOLS AND PROFILER_LIB AND GPERFTOOLS_INCLUDE_DIR)
    message(STATUS "gperftools found: ${PROFILER_LIB}")
    message(STATUS "gperftools headers: ${GPERFTOOLS_INCLUDE_DIR}")
    add_executable(benchmark_erasure_sim
        tests/benchmark_erasure_sim.cpp
    )
    target_link_libraries(benchmark_erasure_sim
        PRIVATE qerasure ${PROFILER_LIB}
    )
    target_include_directories(benchmark_erasure_sim PRIVATE ${GPERFTOOLS_INCLUDE_DIR})
    target_compile_definitions(benchmark_erasure_sim PRIVATE ENABLE_GPERFTOOLS)
    message(STATUS "Benchmark executable configured with gperftools support")
elseif(ENABLE_GPERFTOOLS)
    message(WARNING "gperftools requested but not found. Install libgoogle-perftools-dev (Ubuntu) or google-perftools (macOS)")
    add_executable(benchmark_erasure_sim
        tests/benchmark_erasure_sim.cpp
    )
    target_link_libraries(benchmark_erasure_sim
        PRIVATE qerasure
    )
    message(STATUS "Benchmark executable configured without gperftools support")
else()
    add_executable(benchmark_erasure_sim
        tests/benchmark_erasure_sim.cpp
    )
    target_link_libraries(benchmark_erasure_sim
        PRIVATE qerasure
    )
    message(STATUS "Benchmark executable configured without gperftools support (use -DENABLE_GPERFTOOLS=ON to enable)")
endif()

# --- Fetch pybind11 at configure time ---
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.11.0  # pick a stable tag; adjust as you wish
)
FetchContent_MakeAvailable(pybind11)

# --- add the python extension module ---
pybind11_add_module(qerasure_python
    src/bindings/pybindings.cpp
)

# Link the C++ library into the module
target_link_libraries(qerasure_python PRIVATE qerasure)

# ensure the extension can find headers
target_include_directories(qerasure_python PRIVATE ${CMAKE_SOURCE_DIR}/include)
