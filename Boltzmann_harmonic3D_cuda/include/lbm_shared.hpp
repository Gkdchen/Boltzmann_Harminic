#pragma once

#include <Kokkos_Core.hpp>
#include "Kokkos_Macros.hpp"
namespace lbm_kokkos {

#ifdef USE_FLOAT
using Precision = float;
#else
using Precision = double;
#endif

using Device    = Kokkos::CudaSpace;
using Host      = Kokkos::HostSpace;
using ExecSpace = Kokkos::Cuda;


using DataVectorI = Kokkos::View<int *, Device>;
using DataArray2I = Kokkos::View<int **, Device>;
using DataValue0D = Kokkos::View<Precision , Device>;
using DataVectorD = Kokkos::View<Precision *, Device>;
using DataArray2D = Kokkos::View<Precision **, Device>;
using DataArray3D = Kokkos::View<Precision ***, Device>;
using DataArray4D = Kokkos::View<Precision ****, Device>;


using SubData4D   = Kokkos::Subview<DataArray4D, unsigned, unsigned, unsigned,
                                    std::remove_const_t<decltype(Kokkos::ALL)>>;
using SubData3D   = Kokkos::Subview<DataArray3D, unsigned, unsigned,
                                    std::remove_const_t<decltype(Kokkos::ALL)>>;
using team_policy = Kokkos::TeamPolicy<>;
using member_type = Kokkos::TeamPolicy<>::member_type;
}  // namespace lbm_kokkos

