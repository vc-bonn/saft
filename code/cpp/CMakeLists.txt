cmake_minimum_required(VERSION 3.27)

project(Solver LANGUAGES C CXX CUDA VERSION 0.1.0)

option(SOLVER_BUILD_BINDINGS "Build the Python bindings" ON)

find_package(charonload REQUIRED)

if(charonload_FOUND)
    charonload_add_torch_library(solver_cpp STATIC)
    target_sources(solver_cpp PRIVATE src/solver.cpp)
    target_include_directories(solver_cpp PUBLIC include)
    target_link_libraries(solver_cpp PUBLIC CUDA::cusparse CUDA::cublas)

    charonload_add_torch_library(${TORCH_EXTENSION_NAME} MODULE)

    target_sources(${TORCH_EXTENSION_NAME} PRIVATE python/bindings.cpp)

    target_link_libraries(${TORCH_EXTENSION_NAME} PRIVATE solver_cpp)
endif()