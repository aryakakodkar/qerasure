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
