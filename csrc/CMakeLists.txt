set(APS_UTILS_SRC ${CMAKE_SOURCE_DIR}/csrc/utils/wav.cc
                  ${CMAKE_SOURCE_DIR}/csrc/utils/io.cc
                  ${CMAKE_SOURCE_DIR}/csrc/utils/fft.cc
                  ${CMAKE_SOURCE_DIR}/csrc/utils/math.cc
                  ${CMAKE_SOURCE_DIR}/csrc/utils/window.cc
                  ${CMAKE_SOURCE_DIR}/csrc/utils/stft.cc
                  ${CMAKE_SOURCE_DIR}/csrc/utils/args.cc)

set(APS_TORCH_SRC ${CMAKE_SOURCE_DIR}/csrc/base/stft.cc
                  ${CMAKE_SOURCE_DIR}/csrc/base/pipeline.cc
                  ${CMAKE_SOURCE_DIR}/csrc/enh/time_frequency.cc
                  ${CMAKE_SOURCE_DIR}/csrc/enh/dfsmn.cc
                  ${CMAKE_SOURCE_DIR}/csrc/enh/transformer.cc)
# SHARED didn't work for APPLE
add_library(${APS_UTILS_LIB} STATIC ${APS_UTILS_SRC})
add_library(${APS_TORCH_LIB} STATIC ${APS_TORCH_SRC})
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)