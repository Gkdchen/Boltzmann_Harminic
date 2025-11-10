add_library(kokkos_cuda_wrap STATIC
    kokkos_launcher.cu
    lbm_kokkos_api.cpp 
)


target_link_libraries(kokkos_cuda_wrap PRIVATE kokkoscore)


target_include_directories(kokkos_cuda_wrap
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    ${CMAKE_SOURCE_DIR}/include

    $<TARGET_PROPERTY:kokkoscore,INTERFACE_INCLUDE_DIRECTORIES>
)


target_compile_options(kokkos_cuda_wrap PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>
)