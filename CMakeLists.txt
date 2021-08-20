cmake_minimum_required(VERSION 3.4)
project("APS speech processing toolkit")

set(CMAKE_CXX_STANDARD 11)

MESSAGE(STATUS "Detect OS: ${CMAKE_SYSTEM}")

include_directories(${CMAKE_SOURCE_DIR}/csrc)
link_directories(${CMAKE_SOURCE_DIR}/lib)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)
set(APS_LIB aps)

add_subdirectory(csrc)
add_subdirectory(tests/csrc)
