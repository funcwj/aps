macro(add_demo_command command_name)
    add_executable(${command_name} "${command_name}.cc")
    target_link_libraries(${command_name} ${APS_TORCH_LIB} ${APS_UTILS_LIB} ${TORCH_LIBRARIES})
endmacro()

add_demo_command(rt_enh_transformer)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/demo/real_time_enhancement/cpp/bin)
