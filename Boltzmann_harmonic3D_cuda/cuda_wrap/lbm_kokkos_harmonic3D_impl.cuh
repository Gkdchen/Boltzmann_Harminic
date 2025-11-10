#pragma once
#include "lbm_kokkos_api.hpp"
#include <Kokkos_Core.hpp>
#include "Kokkos_Core_fwd.hpp"
#include "Kokkos_Macros.hpp"
#include <fstream>
#include <iostream>
#include <utility>

#include "lbm_shared.hpp"

namespace lbm_kokkos {
class LBM {
    public:

    int NN,Nx,Ny,Nz,Np,save_iter;
    const int Q = 13;

    Precision source{};//if source is 0 it is the polyharmonic equation without source term.
    Precision Lamda{};//constant for building equilibrium distribution fucntion
    Precision tau_f{};//single relaxation time
    
    void datainit(const LBMParams& p);
    void allocate_views();
    void setup();
    void init();//initial procedure for the order-reduced polyharmonic equation
    void init1();//initial procedure for the harmonic equation 
    void evolution();//evolution procedure for the order-reduced polyharmonic equation
    void evolution1();//evolution procedure for the harmonic equation (final reduced)
    void output_init();//copy device data view to hostspace
    void rsme();
    void output_x1(int m);//output a slice on y-z plane at x=0.2
    void output_y1(int m);//output a slice on x-z plane at y=0.2
    void output_z1(int m);//output a slice on x-y plane at z=0.2
    void output_x2(int m);//output a slice on y-z plane at x=0.8
    void output_y2(int m);//output a slice on x-z plane at y=0.8
    void output_z2(int m);//output a slice on x-y plane at z=0.8
    void Error();//relative error for the order-reduced polyharmonic equation
    void Error1();//relative error for harmonic equation (final reduced)
    void output_error();//output mean square root error
    void output_subtractionx1();//output a slice of subtraction between nuemrical results and analytic solution on y-z plane at x=0.2
    void output_subtractionx2();//output a slice of subtraction between nuemrical results and analytic solution on x-z plane at y=0.2
    void output_subtractiony1();//output a slice of subtraction between nuemrical results and analytic solution on x-y plane at z=0.2
    void output_subtractiony2();//output a slice of subtraction between nuemrical results and analytic solution on y-z plane at x=0.8
    void output_subtractionz1();//output a slice of subtraction between nuemrical results and analytic solution on x-z plane at y=0.8
    void output_subtractionz2();//output a slice of subtraction between nuemrical results and analytic solution on x-y plane at z=0.8
    
    const Precision pi = 3.1415926;
/////////host mirror of device views/////////
    DataArray3D::HostMirror h_U{};      /////
    DataArray3D::HostMirror h_V{};      /////
    DataArray3D::HostMirror h_U0{};     /////
    DataArray3D::HostMirror h_U1{};     /////
    DataArray2I::HostMirror h_e{};      /////
    DataVectorD::HostMirror h_w{};      /////
    DataVectorD::HostMirror h_w_bar{};  /////
    DataVectorD::HostMirror h_R{};      /////
    DataValue0D::HostMirror h_temp1{};  /////
    DataValue0D::HostMirror h_temp2{};  /////
    DataValue0D::HostMirror h_err{};    /////
    DataValue0D::HostMirror h_RSME{};   /////
/////////////////////////////////////////////

    DataArray3D U{};//solution of the order-reduced polyharmonic equation obtained by using LBM solver(fucntion phsi(x,y,z) in the paper)
    DataArray3D V{};//analytic solution of the polyharmonic equation
    DataArray3D U0{};//solution of the previous iteration step, used to calculate relative error for each iteration step
    DataArray3D U1{};//solution of the harmonic equation obtained by using LBM solver (final reduced, fucntion h(x,y,z) in the paper)
    DataArray4D f{};//distribution fucntion before evolution (streaming and collision)
    DataArray4D F{};//distribution fucntion after collision
    DataArray4D M{};//distribution fucntion after streaming
    DataArray4D feq{};//equilibrium distribution fucntion 
    DataArray3D g{};//source term of the target PDEs (polyharmonic equation)
    DataArray2I e{};//unit discrete velocities
    DataVectorD w{};//weight coefficients for each duscrete direction
    DataVectorD w_bar{};//weight coefficients for evolution equation, see Ref.37 in the paper
    DataVectorD R{};//mean square root error for each evolution step
    
    DataValue0D err{};
    DataValue0D RSME{};
    DataValue0D temp1{};
    DataValue0D temp2{};

    private:
    int i,j,k,z;




};



}