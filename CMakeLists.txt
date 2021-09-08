cmake_minimum_required(VERSION 3.14)
project("APS: A Speech Processing Toolkit")

include(FetchContent)
MESSAGE(STATUS "Detect OS: ${CMAKE_SYSTEM}")

set(APS_UTILS_LIB aps_utils)
set(APS_TORCH_LIB aps_torch)
set(CMAKE_CXX_STANDARD 14)

# for libtorch
set(TORCH_VERSION "1.8.0")
set(LIBTORCH_ZIP "libtorch-shared-with-deps-${TORCH_VERSION}%2Bcpu.zip")
set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/${LIBTORCH_ZIP}")
set(LIBTORCH_HASH "2bb925b0a271777ccc69e1af8bb24d730900ce5d4742882d70f2bf61783bd879")
FetchContent_Declare(libtorch
  URL      ${LIBTORCH_URL}
  URL_HASH ${URL_HASH}
)
FetchContent_MakeAvailable(libtorch)
find_package(Torch REQUIRED PATHS ${libtorch_SOURCE_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(${TORCH_INCLUDE_DIRS})

# for csrc
include_directories(${CMAKE_SOURCE_DIR}/csrc)
link_directories(${CMAKE_SOURCE_DIR}/lib)

add_subdirectory(csrc)
add_subdirectory(tests/csrc)
add_subdirectory(demo/real_time_enhancement/cpp)
