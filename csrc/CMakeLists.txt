cmake_minimum_required(VERSION 3.4)

set(APS_SRC ${CMAKE_SOURCE_DIR}/csrc/io/wav.cc
            ${CMAKE_SOURCE_DIR}/csrc/io/basic.cc)

add_library(${APS_LIB} SHARED ${APS_SRC})
