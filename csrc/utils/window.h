// Copyright 2021 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

#ifndef CSRC_UTILS_WINDOW_H_
#define CSRC_UTILS_WINDOW_H_

#include "utils/log.h"
#include "utils/math.h"

class WindowFunction {
 public:
  static void Generate(const std::string& name, float* window,
                       int32_t window_len, bool periodic = true);

 private:
  static void Hanning(float* window, int32_t window_len, bool periodic = true);
  static void Hamming(float* window, int32_t window_len, bool periodic = true);
  static void Rectangular(float* window, int32_t window_len);
  static void SqrtHanning(float* window, int32_t window_len, bool periodic = true);
  static void Blackman(float* window, int32_t window_len, bool periodic = true);
  static void Bartlett(float* window, int32_t window_len, bool periodic = true);
};

#endif
