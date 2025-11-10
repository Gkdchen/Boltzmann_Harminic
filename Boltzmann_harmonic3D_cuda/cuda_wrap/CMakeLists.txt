add_library(kokkos_cuda_wrap STATIC
    kokkos_launcher.cu
    lbm_kokkos_api.cpp 
)

# 【1】链接 Kokkos 核心库
target_link_libraries(kokkos_cuda_wrap PRIVATE kokkoscore)

# 【2】把头文件目录暴露给 host_utils
target_include_directories(kokkos_cuda_wrap
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    ${CMAKE_SOURCE_DIR}/include
    # 【3】让 host 也能通过依赖拿到 Kokkos 头文件
    $<TARGET_PROPERTY:kokkoscore,INTERFACE_INCLUDE_DIRECTORIES>
)

# 【4】device lambda 必需标志
target_compile_options(kokkos_cuda_wrap PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>
)