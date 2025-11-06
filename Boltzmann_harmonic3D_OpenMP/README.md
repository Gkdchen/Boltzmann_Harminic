# LBM Kokkos OpenMP Code User Guide

## Introduction  
LBM Kokkos OpenMP is a 3D Harmonic Lattice Boltzmann solver that uses the Kokkos OpenMP backend for efficient shared-memory parallelism.

## System Requirements  
- **OS**: Ubuntu 22.04 or newer  
- **Compiler**: GCC ≥ 12 (C++17 capable)  
- **Dependencies**: `gcc`, `cmake`, `git`, `libspdlog-dev`, `kokkos`, `Lua`, `sol2`

## Build & Run  
1. **Configure and compile**  
   ```shell
   cd build/
   cmake .. -DKOKKOS_LBM_BACKEND=OpenMP \
            -DKokkos_ENABLE_DEBUG=ON \
            -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
            -DKOKKOS_ENABLE_BOUNDS_CHECK=ON \
            -DCMAKE_BUILD_TYPE=Release
   make -j
   ```

2. **Launch the simulation**  
   ```shell
   cd ..
   ./build/boltzmann_harmonic3d ./config/config.lua
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
       save_iter   = 1000000,
       display_iter= 100,
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

4. **Optional OpenMP tuning**  
   ```shell
   export OMP_PROC_BIND=spread
   export OMP_PLACES=threads
   ```

## Output Files  
- `3D-Laplace*.dat`: cross-sectional field data  
- `subtraction*.dat`: residual history  

Follow the steps above to experience the LBM Kokkos OpenMP harmonic solver.