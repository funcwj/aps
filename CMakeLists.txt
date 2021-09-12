cmake_minimum_required(VERSION 3.14)
project("APS: A Speech Processing Toolkit")
message(STATUS "Detect OS: ${CMAKE_SYSTEM}")

if(WIN32)
  message(FATAL_ERROR  "Now doesn't support Windows")
endif()

option(BUILD_TESTS "Build the test commands" ON)
option(BUILD_DEMOS "Build the demo commands" ON)

include(FetchContent)

set(APS_UTILS_LIB aps_utils)
set(APS_TORCH_LIB aps_torch)
set(CMAKE_CXX_STANDARD 14)

# for libtorch
set(TORCH_VERSION "1.8.0")
if(UNIX AND NOT APPLE)
  set(LIBTORCH_ZIP "libtorch-shared-with-deps-${TORCH_VERSION}%2Bcpu.zip")
  set(LIBTORCH_HASH "2bb925b0a271777ccc69e1af8bb24d730900ce5d4742882d70f2bf61783bd879")
else()
  set(LIBTORCH_ZIP "libtorch-macos-${TORCH_VERSION}.zip")
  set(LIBTORCH_HASH "d080f1f5c0b128d72de4453018d61bbf21afb67a452cc43b4f0a9099b7cf0c7f")
endif()
set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/${LIBTORCH_ZIP}")
message(STATUS "Get libtorch: ${LIBTORCH_URL} ...")
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
if(BUILD_TESTS)
  add_subdirectory(tests/csrc)
endif(BUILD_TESTS)
if(BUILD_DEMOS)
  add_subdirectory(demos/real_time_enhancement/cpp)
endif(BUILD_DEMOS)
