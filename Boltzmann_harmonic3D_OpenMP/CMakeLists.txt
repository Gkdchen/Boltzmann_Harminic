cmake_minimum_required(VERSION 3.18)
project(Boltzmann_harmonic3D LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(EXTERNAL_DIR "${CMAKE_SOURCE_DIR}/external")


include(cmake/build_or_find_kokkos.cmake)


set(SOL2_DIR "${EXTERNAL_DIR}/sol2")
add_library(sol2 INTERFACE)
target_include_directories(sol2 INTERFACE ${SOL2_DIR})

set(LUA_DIR "${EXTERNAL_DIR}/Lua")
add_library(Lua INTERFACE)
target_include_directories(Lua INTERFACE ${LUA_DIR})


find_package(spdlog REQUIRED CONFIG)


add_executable(boltzmann_harmonic3d
    src/main.cpp
    src/configs.cpp
    src/Boltzmann_harmonic3D.cpp
)


target_link_libraries(boltzmann_harmonic3d PRIVATE
    Kokkos::kokkos
    lua
    sol2
    spdlog::spdlog
)


target_include_directories(boltzmann_harmonic3d PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)


if(KOKKOS_LBM_BACKEND STREQUAL "OpenMP")
    target_compile_options(boltzmann_harmonic3d PRIVATE -fopenmp)
    target_link_options(boltzmann_harmonic3d PRIVATE -fopenmp)
endif()
