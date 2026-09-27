set(TARGET ltx2-capi-smoke)

add_executable(${TARGET} main.c)
if(APPLE)
    sd_set_macos_rpaths(${TARGET})
endif()
target_link_libraries(${TARGET} PRIVATE stable-diffusion)
target_compile_features(${TARGET} PUBLIC c_std_11)
