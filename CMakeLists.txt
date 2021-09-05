cmake_minimum_required(VERSION 3.4)
project("APS speech processing toolkit")

MESSAGE(STATUS "Detect OS: ${CMAKE_SYSTEM}")

find_package(Torch REQUIRED)

set(APS_BASE aps_base)
set(APS_TORCH aps_torch)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)

include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/csrc)
link_directories(${CMAKE_SOURCE_DIR}/lib)

add_subdirectory(csrc)
add_subdirectory(tests/csrc)
