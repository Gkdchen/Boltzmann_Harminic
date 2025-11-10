set(EXTERNAL_DIR "${CMAKE_SOURCE_DIR}/external")


set(LUA_SRC_DIR "${EXTERNAL_DIR}/Lua/lua-5.4.6/src")
file(GLOB LUA_SOURCES "${EXTERNAL_DIR}/Lua/lua-5.4.6/src/*.c")
list(FILTER LUA_SOURCES EXCLUDE REGEX "lua.c|luac.c")
add_library(lua_static STATIC ${LUA_SOURCES})
set_target_properties(lua_static PROPERTIES LINKER_LANGUAGE C)
target_include_directories(lua_static PUBLIC
    ${EXTERNAL_DIR}/Lua/lua-5.4.6/include
)


add_executable(host_utils
    src/main.cpp
    src/configs.cpp
)


target_include_directories(host_utils PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${EXTERNAL_DIR}/sol2
     $<TARGET_PROPERTY:kokkoscore,INTERFACE_INCLUDE_DIRECTORIES>
)


target_link_libraries(host_utils PRIVATE
    kokkos_cuda_wrap
    lua_static       
    spdlog::spdlog
)


target_compile_options(host_utils PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>
)