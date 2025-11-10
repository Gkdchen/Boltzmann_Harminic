#include "lbm_kokkos_api.hpp"
#include "lbm_kokkos_harmonic3D_impl.cuh" 
#include <iostream>

namespace lbm_kokkos {
void kokkos_init(int argc, char* argv[]) { Kokkos::initialize(argc, argv); }
void kokkos_finalize()                   { Kokkos::finalize(); }

std::unique_ptr<LBM> create_lbm(const LBMParams& p) {
    auto h = std::make_unique<LBM>();
    h->datainit(p);
    return h;
}

void lbm_run_kokkos_part(LBMParams& p) {
    auto lbm = create_lbm(p);
    Kokkos::fence();
    lbm->allocate_views();
    Kokkos::fence();
    lbm->init();//initiation for the order-reduced polyharmonic equation                               
    Kokkos::fence();
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////repeat this procedure if k is larger than 2                                                       /////
//////but notice the source term g should be replaced by U for each reduction of order                  /////
    for (int n = 0; ; ++n) {                                                                            /////
        lbm->NN = n;                                                                                    /////
        lbm->evolution();                                                                               /////
        Kokkos::fence();                                                                                /////
        if (n % p.display_iter == 0) {                                                                             /////
            lbm->Error();                                                                               /////
            Kokkos::fence();                                                                            /////
            double err = lbm->h_err();                                                                  /////
            std::cout << "The " << n << "th computation result:" << std::endl;                          /////
            std::cout << "The max absolute error is:" << std::scientific << err << '\n';                /////
            if (err < 1e-15) break;   //convergence condition                                                                  /////
        }                                                                                               /////
    }                                                                                                   /////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
    lbm->init1();//initiation for the harmonic equation (finally reduced)
    for (int n = 0; ; ++n) {
        lbm->NN = n;
        lbm->evolution1();//solving the harmonic equation (finally reduced)
        Kokkos::fence();
        lbm->rsme();
        Kokkos::fence();
        double rsme_val=lbm->h_RSME();
        lbm->h_R(n) = sqrt(rsme_val/(p.Np*p.Np));//calculate mean square root error
        if (n % p.display_iter == 0) {
            lbm->Error1();  
            Kokkos::fence();
            double err = lbm->h_err();
            std::cout << "The " << n << "th computation result:" << std::endl;
            std::cout << "The max absolute error is:" << std::scientific << err << '\n';
            if (err < 1e-15) {//convergence condition
                lbm->output_init();
                lbm->output_x1(n);
                lbm->output_y1(n);
                lbm->output_z1(n);
                lbm->output_x2(n);
                lbm->output_y2(n);
                lbm->output_z2(n);
                lbm->output_subtractionx1();
                lbm->output_subtractionx2();
                lbm->output_subtractiony1();
                lbm->output_subtractiony2();
                lbm->output_subtractionz1();
                lbm->output_subtractionz2();
                lbm->output_error();
                break;
            }
        }
    }
}
} // namespace lbm_kokkos