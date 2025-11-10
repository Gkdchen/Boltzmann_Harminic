# LBM Kokkos CUDA Code User Guide

## Introduction  
LBM Kokkos CUDA is a 3D Harmonic Lattice Boltzmann solver accelerated by NVIDIA GPUs through the Kokkos portability framework.

## System Requirements  
- **OS**: Ubuntu 22.04 or newer  
- **Compiler**: GCC ≥ 12 (C++17 capable)  
- **Dependencies**: `gcc`, `cmake`, `git`, `libspdlog-dev`, `cuda-toolkit`

## Build & Run  
1. **Configure & compile**  
   ```shell
   cmake -DKOKKOS_LBM_BACKEND=CUDA \
         -DKokkos_ENABLE_CUDA=ON \
         -DKokkos_ENABLE_SERIAL=ON \
         -DCMAKE_BUILD_TYPE=Release
   cd build/
   make -j
   ```

2. **Launch the simulation**  
   ```shell
   cd ..
   ./build/host_util/host_util ./config/config.lua
   ```

3. **Edit settings**  
   Modify `./config/config.lua` to change parameters, e.g.  
   ```lua
   config = {
     grid = {
       nx = 100,  -- x-direction grid points
       ny = 100,  -- y-direction grid points
       nz = 100,  -- z-direction grid points
       np = 100   -- points per unit length
     },
     solver = {
       solver_name = "3d",
       save_iter   = 1000000,  --large enough to save RSME
       display_iter= 100,      --output error
       max_iter    = 10000000
     },
     flow = {
       source = 1.0,
       lamda  = 1.0/3.0,
       tau_f  = 1.0   -- numerical parameter
     },
     experiments = {
       name = "harmonic"
     }
   }
   ```

## Output Files  
- `3D-Laplace*.dat`: cross-sectional field data  
- `subtraction*.dat`: residual history  

Follow the steps above to experience the LBM Kokkos CUDA harmonic solver.