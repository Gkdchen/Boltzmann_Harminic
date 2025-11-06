#pragma once

#include "Kokkos_Core.hpp"
#include "Kokkos_Macros.hpp"
namespace lbm_kokkos {

#ifdef USE_FLOAT
using Precision = float;
#else
using Precision = double;
#endif

using Device = Kokkos::DefaultExecutionSpace;

using DataArray2D = Kokkos::View<Precision **, Device>;
using DataArray3D = Kokkos::View<Precision ***, Device>;
using DataArray4D = Kokkos::View<Precision ****, Device>;
using DataVectorD = Kokkos::View<Precision *, Device>;
using DataArray2I = Kokkos::View<int **, Device>;
using DataVectorI = Kokkos::View<int *, Device>;
using SubData4D   = Kokkos::Subview<DataArray4D, unsigned, unsigned, unsigned,
                                    std::remove_const_t<decltype(Kokkos::ALL)>>;
using SubData3D   = Kokkos::Subview<DataArray3D, unsigned, unsigned,
                                    std::remove_const_t<decltype(Kokkos::ALL)>>;
using team_policy = Kokkos::TeamPolicy<>;
using member_type = Kokkos::TeamPolicy<>::member_type;
}  // namespace lbm_kokkos
// last index is hydro variable
// n-1 first indexes are space (i,j,k,....)

/**
 * Retrieve cartesian coordinate from index, using memory layout information.
 *
 * for each execution space define a prefered layout.
 * Prefer left layout  for CUDA execution space.
 * Prefer right layout for OpenMP execution space.
 *
 * These function will eventually disappear.
 * We still need then as long as parallel_reduce does not accept MDRange policy.
 */

/* 2D */

// KOKKOS_INLINE_FUNCTION
// void index2coord(size_t index, int &i, int &j, size_t Nx, size_t Ny) {
// #ifdef KOKKOS_ENABLE_CUDA
//   j = index / Nx;
//   i = index - j * Nx;
// #else
//   i = index / Ny;
//   j = index - i * Ny;
// #endif
// }

// KOKKOS_INLINE_FUNCTION
// size_t coord2index(int i, int j, size_t Nx, size_t Ny) {
// #ifdef KOKKOS_ENABLE_CUDA
//   return i + Nx * j;  // left layout
// #else
//   return j + Ny * i;  // right layout
// #endif
// }

// /* 3D */

// KOKKOS_INLINE_FUNCTION
// void index2coord(size_t index, int &i, int &j, int &k, size_t Nx, size_t Ny,
//                  size_t Nz) {
// #ifdef KOKKOS_ENABLE_CUDA
//   size_t NxNy = Nx * Ny;
//   k           = index / NxNy;
//   j           = (index - k * NxNy) / Nx;
//   i           = index - j * Nx - k * NxNy;
// #else
//   size_t ny_nz = Ny * Nz;
//   i            = index / ny_nz;
//   j            = (index - i * ny_nz) / Nz;
//   k            = index - j * Nz - i * ny_nz;
// #endif
// }

// KOKKOS_INLINE_FUNCTION
// size_t coord2index(int i, int j, int k, size_t Nx, size_t Ny, size_t Nz) {
// #ifdef KOKKOS_ENABLE_CUDA
//   return i + Nx * j + Nx * Ny * k;  // left layout
// #else
//   return k + Nz * j + Nz * Ny * i;  // right layout
// #endif
// }
// class NotImplementedException : public std::logic_error {
//  public:
//   NotImplementedException()
//       : std::logic_error{"Function not yet implemented."} {}
// };
// }  // namespace lbm_kokkos
