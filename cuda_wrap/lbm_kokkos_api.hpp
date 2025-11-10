#pragma once
#include <memory>

namespace lbm_kokkos {
struct LBMParams {//get parameters
    int Nx{}, Ny{}, Nz{}, Np{};
    double source{}, lamda{}, tau_f{};
    int save_iter{}, max_iter{}, display_iter{};
};

class LBM;
std::unique_ptr<LBM> create_lbm(const LBMParams& p);
void kokkos_init(int argc, char* argv[]);
void kokkos_finalize();
void lbm_run_kokkos_part(LBMParams& p);
} // namespace lbm_kokkos