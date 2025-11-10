cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_ARCHITECTURES 89)
project(Boltzmann_harmonic3D_cuda LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


set(Kokkos_SKIP_GIT_VERSION ON CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_CUDA      ON CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_SERIAL    ON CACHE BOOL "" FORCE)
set(Kokkos_ARCH_ADA89       ON CACHE BOOL "" FORCE)


include(FetchContent)
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG        v1.12.0
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(spdlog)

set(Kokkos_CXX_FLAGS "-w" CACHE STRING "" FORCE)   
set(Kokkos_CUDA_FLAGS "-w" CACHE STRING "" FORCE)


add_subdirectory(${CMAKE_SOURCE_DIR}/external/kokkos kokkos_build)
add_subdirectory(cuda_wrap)
add_subdirectory(host_utils)