cmake_minimum_required(VERSION 3.4)

set(APS_BASE_SRC ${CMAKE_SOURCE_DIR}/csrc/utils/wav.cc
                 ${CMAKE_SOURCE_DIR}/csrc/utils/io.cc
                 ${CMAKE_SOURCE_DIR}/csrc/utils/fft.cc
                 ${CMAKE_SOURCE_DIR}/csrc/utils/math.cc
                 ${CMAKE_SOURCE_DIR}/csrc/utils/window.cc
                 ${CMAKE_SOURCE_DIR}/csrc/utils/stft.cc)

set(APS_TORCH_SRC ${CMAKE_SOURCE_DIR}/csrc/transform/stft.cc)

add_library(${APS_BASE} SHARED ${APS_BASE_SRC})
add_library(${APS_TORCH} SHARED ${APS_TORCH_SRC})
