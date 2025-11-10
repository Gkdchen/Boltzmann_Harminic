#include <spdlog/spdlog.h>
#include <iomanip>
#include <cmath>
#include <filesystem>
#include <memory>

#include "lbm_kokkos_api.hpp"

#include "configs.hpp"

int main(int argc, char* argv[])
{
    if (argc <= 1) {
        spdlog::error("Please give the config file path");
        return 1;
    }
    std::string config_file = argv[1];

    sol::state lua;
    lbm_kokkos::kokkos_init(argc, argv);
    {
        auto configs = lbm_kokkos::LBMConfigs();
        lua.open_libraries(sol::lib::base, sol::lib::coroutine, sol::lib::string,
                   sol::lib::io, sol::lib::math);
        std::filesystem::path file_path(config_file);
        if (!std::filesystem::exists(file_path)) {
            spdlog::error("Config file {} not found", config_file);
            return 1;
        }
        lua.script_file(config_file);
        configs.setup(lua);
        lbm_kokkos::LBMParams LBMParams;
////////////////get parameters from config.lua////////////////
        LBMParams.Nx=configs.grid_.nx_;
        LBMParams.Ny=configs.grid_.ny_;
        LBMParams.Nz=configs.grid_.nz_;
        LBMParams.Np=configs.grid_.np_;
        LBMParams.source=configs.flow_.source_;
        LBMParams.lamda=configs.flow_.lamda_;
        LBMParams.tau_f=configs.flow_.tau_f_;
        LBMParams.save_iter=configs.solver_.save_iter_;
        LBMParams.display_iter=configs.solver_.display_iter_;
        LBMParams.max_iter=configs.solver_.max_iter_;
///////////////////////////////////////////////////////////////
        lbm_run_kokkos_part(LBMParams);
    }
    lbm_kokkos::kokkos_finalize();
    return 0;
}