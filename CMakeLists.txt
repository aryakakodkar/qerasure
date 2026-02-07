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
