#include <spdlog/spdlog.h>
#include <sol/sol.hpp>
#include <iomanip>        // std::setiosflags
#include <cmath>  
#include <csignal>
#include <filesystem>
#include <memory>

//include some function hpp
#include "configs.hpp"
#include "Kokkos_Core.hpp"
#include "lbm_kokkos_harmonic3D.hpp"


/**
 * @brief The entry point of the program.
 *
 * @param argc The number of command-line arguments.
 * @param argv An array of command-line arguments.
 * @return int The exit status of the program.
 */

int main(int argc, char* argv[]) {
    int i,j,z,n,Nx,Ny,Nz,Np;
    float pi = 3.1415926;
    const int Q = 13;
  std::string config_file;
  if (argc <= 1) {
    spdlog::error(
        "Please give the config file path as argument, like "
        "`./config/config3dMRT.json`");
    return 1;
  }

  config_file = argv[1];//命令行传输config文件名
  sol::state lua;
  Kokkos::initialize(argc, argv);
  {
    auto configs = lbm_kokkos::LBMConfigs();
    lua.open_libraries(sol::lib::base, sol::lib::coroutine, sol::lib::string,
                       sol::lib::io, sol::lib::math);

    std::filesystem::path file_path(config_file);
    if (std::filesystem::exists(file_path)) {
      lua.script_file(config_file);
      lua.script_file(file_path.parent_path().parent_path() / "src" /
                      "validation.lua");
    } else {
      spdlog::error("Config file {} does not exist", config_file);
      return 1;
    }
    configs.setup(lua);
    std::filesystem::path log_folder_path =
        file_path.parent_path().parent_path() / "logs" /
        configs.experiment_.experiment_name_ / configs.solver_.solver_name_;

    log_folder_path /= std::to_string(configs.flow_.source_);
    lbm_kokkos::LBM LBM;
     {
  LBM.Nx = Nx = configs.grid_.nx_;
  LBM.Ny = Ny = configs.grid_.ny_;
  LBM.Nz = Nz = configs.grid_.nz_;
  LBM.Np = Np = configs.grid_.np_;
  LBM.source = configs.flow_.source_;
  LBM.Lamda  = configs.flow_.lamda_;
  LBM.tau_f  = configs.flow_.tau_f_;
  LBM.U    = lbm_kokkos::DataArray3D ("U", Nx+1, Ny+1,Nz+1);//+1,0-Nx1,Nx+1
  LBM.V    = lbm_kokkos::DataArray3D ("V", Nx+1, Ny+1,Nz+1);
  LBM.U0   = lbm_kokkos::DataArray3D ("U0", Nx+1, Ny+1,Nz+1);//+1,0-Nx1,Nx+1
  LBM.U1   = lbm_kokkos::DataArray3D ("U1", Nx+1, Ny+1,Nz+1);
  LBM.f    = lbm_kokkos::DataArray4D ("f", Nx+1, Ny+1,Nz+1,Q);//+1,0-Nx1,Nx+1
  LBM.F    = lbm_kokkos::DataArray4D ("F", Nx+1, Ny+1,Nz+1,Q);
  LBM.M    = lbm_kokkos::DataArray4D ("M", Nx+1, Ny+1,Nz+1,Q);//+1,0-Nx1,Nx+1
  LBM.feq  = lbm_kokkos::DataArray4D ("feq", Nx+1, Ny+1,Nz+1,Q);
  LBM.g    = lbm_kokkos::DataArray3D ("g", Nx+1, Ny+1,Nz+1);//+1,0-Nx1,Nx+1
  LBM.w    = lbm_kokkos::DataVectorD ("w", Q);
  LBM.w_bar= lbm_kokkos::DataVectorD ("w_bar", Q);
  LBM.R    = lbm_kokkos::DataVectorD ("R", configs.solver_.save_iter_);//+1,0-Nx1,Nx+1
  LBM.error= lbm_kokkos::DataVectorD ("error",1);
  LBM.e    = lbm_kokkos::DataArray2I ("e", Q , 3);
  }
    ////initiation for the order-reduced polyharmonic equation
        Kokkos::parallel_for("init_V", 
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {Nx+1, Ny+1, Nz+1}),
            KOKKOS_LAMBDA (int i, int j, int z) {
                LBM.V(i, j, z) = cos(pi * static_cast<double>(i) / Np) *
                              sin(pi * static_cast<double>(j) / Np) *
                              sin(pi * static_cast<double>(z) / Np);
            }
        );
        
        // 等待初始化完成
        Kokkos::fence();
        
        // 调用初始化函数
        LBM.init();//initiation for the order-reduced polyharmonic equation
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////repeat this procedure if k is larger than 2                                                                                   /////
//////but notice the source term g should be replaced by U for each reduction of order                                              /////
        int n;                                                                                                                      /////   
        for(n = 0; ; n++) {                                                                                                         /////
            // 演化步骤                                                                                                              /////
            LBM.NN=n;                                                                                                               /////
            LBM.evolution();                                                                                                        /////
                                                                                                                                    /////                                                                                                                                                   
            if(n % 100 == 0) {                                                                                                      /////
                // 计算误差                                                                                                          /////
                LBM.Error();                                                                                                        /////
                // 在主机上打印结果                                                                                                   /////
                Kokkos::fence(); // 确保所有设备计算完成                                                                               /////          
                // 获取误差值（假设 error 是 Kokkos::View<double>）                                                                    /////
                auto h_err = Kokkos::create_mirror_view(LBM.error);                                                                 /////
                Kokkos::deep_copy(h_err, LBM.error);                                                                                /////
                double host_error = h_err(0);                                                                                       /////
                std::cout << "The " << n << "th computation result:" << std::endl;                                                  /////
                std::cout << "The max absolute error is:" << std::setiosflags(std::ios::scientific) << host_error << std::endl;     /////
                if(host_error < 1.0e-15) {                                                                                          /////
                    break;                                                                                                          /////
                }                                                                                                                   /////
            }                                                                                                                       /////        }
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        LBM.init1();//initiation for the harmonic equation (finally reduced)
        for(n = 0; ; n++) {
            // 第二阶段演化
            LBM.NN=n;
            LBM.evolution1();//solving the harmonic equation (finally reduced)
            
            double RSME = 0;
            
            //calculate mean square root error
            Kokkos::parallel_reduce("compute_RSME",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {Nx+1, Ny+1, Nz+1}),
                KOKKOS_LAMBDA (int i, int j, int z, double& local_RSME) {
                    double diff = LBM.U1(i, j, z) - LBM.V(i, j, z);
                    local_RSME += diff * diff;
                }, RSME
            );
            
            // 计算 R[n]

                    LBM.R(n) = sqrt(RSME / (Np * Np));
     
            
            if(n % 100 == 0) {
                // 计算误差
                LBM.Error1();
                Kokkos::fence();
                auto h_err = Kokkos::create_mirror_view(LBM.error);
                Kokkos::deep_copy(h_err, LBM.error);
                double host_error = h_err(0);
                
                std::cout << "The " << n << "th computation result:" << std::endl;
                std::cout << "The max absolute error is:" << std::setiosflags(std::ios::scientific) << host_error << std::endl;
                
                if(host_error < 1.0e-15)//convergence condition
                {
                    LBM.output_x1(n);
                    LBM.output_y1(n);
                    LBM.output_z1(n);
                    LBM.output_x2(n);
                    LBM.output_y2(n);
                    LBM.output_z2(n);
                    LBM.output_subtractionx1();
                    LBM.output_subtractionx2();
                    LBM.output_subtractiony1();
                    LBM.output_subtractiony2();
                    LBM.output_subtractionz1();
                    LBM.output_subtractionz2();
                    LBM.output_error();
                    break;
                }
            }
        
        }
   

#ifdef USE_WANDB
    if (configs.experiment_.wandb) {
      wandbcpp::finish();
    }
#endif
  }
  Kokkos::finalize();
  return 0;
}
}
