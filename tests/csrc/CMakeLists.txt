cmake_minimum_required(VERSION 3.4)

enable_testing()

macro(add_aps_test test_name)
    add_executable(${test_name} "${test_name}.cc")
    target_link_libraries(${test_name} ${APS_BASE} ${APS_TORCH} ${TORCH_LIBRARIES})
    add_test(NAME ${test_name} COMMAND ${test_name})
endmacro()

add_aps_test(test-wav)
add_aps_test(test-fft)
add_aps_test(test-stft)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/tests/csrc/bin)
