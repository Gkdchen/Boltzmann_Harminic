#pragma once
#include <cstddef>
#include <map>
#include <vector>

#include "lbm_shared.hpp"
#include "sol/sol.hpp"


namespace lbm_kokkos {
struct LBMConfigs {

  struct GridConfig {
    size_t nx_{};
    size_t ny_{};
    size_t nz_{};
    size_t np_{};
    size_t total_nodes_{};
  };

  struct FlowConfig {
    Precision source_{};
    Precision lamda_{};
    Precision tau_f_{};
  };
  struct SolverConfig {
    std::string solver_name_{};
    size_t iter_{};
    size_t max_iter_{};
    size_t save_iter_{};
    size_t display_iter_{};
    size_t total_nodes_{};
  };
  struct ExperimentalConfig {
    std::string experiment_name_{};
  };

  GridConfig grid_               = GridConfig();
  FlowConfig flow_               = FlowConfig();
  SolverConfig solver_           = SolverConfig();
  ExperimentalConfig experiment_ = ExperimentalConfig();
  std::vector<std::map<std::string, std::vector<int>>> watch_points_;
  sol::table custom_;

  LBMConfigs() = default;
  ~LBMConfigs();
  virtual void setup(const sol::state &lua);
};
}  // namespace lbm_kokkos
