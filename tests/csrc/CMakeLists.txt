cmake_minimum_required(VERSION 3.4)

add_executable(test-wav test-wav.cc)
add_executable(test-fft test-fft.cc)
add_executable(test-stft test-stft.cc)

target_link_libraries(test-wav ${APS_LIB} ${TORCH_LIBRARIES})
target_link_libraries(test-fft ${APS_LIB} ${TORCH_LIBRARIES})
target_link_libraries(test-stft ${APS_LIB} ${TORCH_LIBRARIES})

set_property(TARGET test-wav PROPERTY CXX_STANDARD 14)
set_property(TARGET test-fft PROPERTY CXX_STANDARD 14)
set_property(TARGET test-stft PROPERTY CXX_STANDARD 14)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/tests/csrc/bin)
