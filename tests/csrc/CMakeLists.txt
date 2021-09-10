enable_testing()

macro(add_aps_test test_name)
    add_executable(${test_name} "${test_name}.cc")
    target_link_libraries(${test_name} ${APS_TORCH_LIB} ${APS_UTILS_LIB} ${TORCH_LIBRARIES})
    add_test(NAME ${test_name} COMMAND ${test_name})
endmacro()

add_aps_test(test-wav)
add_aps_test(test-fft)
add_aps_test(test-utils-stft)
add_aps_test(test-torch-stft)
add_aps_test(test-pipeline)
add_aps_test(test-arg)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/tests/csrc/bin)
