cmake_minimum_required(VERSION 3.18)
project(haloExchangeHeaderOnly LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# ========== Set Paths ==========
set(CONDA_ROOT "$ENV{HOME}/miniconda3_ppc64/envs/halo_test")
set(MPI_C_COMPILER    "${CONDA_ROOT}/bin/mpicc")
set(MPI_CXX_COMPILER  "${CONDA_ROOT}/bin/mpicxx")   # optional, for C++

message(STATUS "CONDA_ROOT: ${CONDA_ROOT}")

find_library(MPI_LIB
    NAMES mpi            # looks for libmpi.so / libmpi.a
    HINTS ${CONDA_ROOT}/lib
    PATHS /usr/lib64 /usr/lib /usr/local/lib
    REQUIRED)

find_path(MPI_INCLUDE_DIR
    NAMES mpi.h
    HINTS ${CONDA_ROOT}/include
    PATHS /usr/include /usr/include /usr/local/include
    REQUIRED)

message(STATUS "Found MPI library   : ${MPI_LIB}")
message(STATUS "Found MPI includes  : ${MPI_INCLUDE_DIR}")

# ========== Provide C and C++ compiler ==========
set(CMAKE_C_COMPILER   "${MPI_C_COMPILER}"   CACHE FILEPATH "C compiler")
set(CMAKE_CXX_COMPILER "${MPI_CXX_COMPILER}" CACHE FILEPATH "C++ compiler")

find_package(CUDAToolkit REQUIRED)

find_library(NCCL_LIB
    NAMES nccl
    HINTS ${CONDA_ROOT}/lib
    PATHS /usr/lib64 /usr/lib /usr/local/lib
    REQUIRED)            # libnccl.so

find_path(NCCL_INCLUDE_DIR
    NAMES nccl.h
    HINTS ${CONDA_ROOT}/include
    PATHS /usr/include /usr/include /usr/local/include
    REQUIRED)    # nccl.h

message(STATUS "Found NCCL library   : ${NCCL_LIB}")
message(STATUS "Found NCCL includes  : ${NCCL_INCLUDE_DIR}")

# ========== Set Compute Capability ==========
set(CUDA_ARCH "70")  # Change to match your GPU
message(STATUS "Building for CUDA architecture tesla V100 with CUDA_ARCH: ${CUDA_ARCH}")

# ========== Source Files ==========
# This project is now self-contained: only main.cu is needed
# All kernel functions are integrated into main.cu
# The halo_lib.hpp is header-only (no external compilation needed)
add_executable(halo_test
    main.cu
)

# ========== Include Directories ==========
target_include_directories(halo_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${NCCL_INCLUDE_DIR}
    ${MPI_INCLUDE_DIR}
)

# ========== Link Directories ==========
target_link_libraries(halo_test PRIVATE
    ${NCCL_LIB}           # libnccl.so
    ${MPI_LIB}            # MPI library
    CUDA::cudart          # CUDA runtime (libcudart.so)
)

# ========== CUDA Properties ==========
set_property(TARGET halo_test PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET halo_test PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})

set_target_properties(halo_test PROPERTIES
    BUILD_RPATH "${CONDA_ROOT}/lib"
    INSTALL_RPATH "${CONDA_ROOT}/lib"
)
