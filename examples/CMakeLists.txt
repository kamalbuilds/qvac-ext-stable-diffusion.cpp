include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(cli)
add_subdirectory(server)

option(SD_BUILD_LTX2_CAPI_SMOKE "sd: build the ltx2-capi-smoke link/ABI test" ON)
if(SD_BUILD_LTX2_CAPI_SMOKE)
    add_subdirectory(ltx2-capi-smoke)
endif()