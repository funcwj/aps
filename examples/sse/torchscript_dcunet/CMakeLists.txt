cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project("Speech Enhancement Example Using TorchScript")

find_package(Torch REQUIRED)

find_path(SOX_INCLUDE_DIR sox.h ${SOX_ROOT} /usr/local/include)
find_library(SOX_LIBRARY sox ${SOX_ROOT} /usr/local/lib)

include_directories(${SOX_INCLUDE_DIR})
add_executable(dcunet-enhan dcunet-enhan.cc)
target_link_libraries(dcunet-enhan "${TORCH_LIBRARIES}" ${SOX_LIBRARY})
set_property(TARGET dcunet-enhan PROPERTY CXX_STANDARD 14)
