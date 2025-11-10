#include "configs.hpp"

#include <vector>

#include "sol/sol.hpp"
#ifdef USE_WANDB
#include "wandbcpp.hpp"
#endif
namespace lbm_kokkos {

void lbm_kokkos::LBMConfigs::setup(const sol::state &lua) {
  auto config        = lua["config"];
  auto grid          = config["grid"];
  grid_.nx_          = grid["nx"];
  grid_.ny_          = grid["ny"];
  grid_.nz_          = grid["nz"].get_or(1);
  grid_.np_          = grid["np"];
  grid_.total_nodes_ = grid_.nx_ * grid_.ny_ * grid_.nz_;

  auto solver              = config["solver"];
  solver_.solver_name_     = solver["solver_name"];
  solver_.save_iter_       = solver["save_iter"];
  solver_.display_iter_    = solver["display_iter"];
  solver_.max_iter_        = solver["max_iter"];
  solver_.total_nodes_     = grid_.total_nodes_;

  auto flow       = config["flow"];
  flow_.source_   = flow["source"];
  flow_.lamda_    = flow["lamda"].get_or(1.0);
  flow_.tau_f_    = flow["tau_f"];

  auto experiment              = config["experiments"];
  experiment_.experiment_name_ = experiment["name"];

  solver_.iter_   = 0;

};

LBMConfigs::~LBMConfigs() { custom_ = sol::lua_nil; };
}  // namespace lbm_kokkos
