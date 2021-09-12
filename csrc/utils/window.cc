// Copyright 2021 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

#include "utils/window.h"

namespace aps {

void WindowFunction::Hanning(float* window, int32_t window_len, bool periodic) {
  float a = PI2 / (periodic ? window_len : window_len - 1.0f);
  for (int32_t i = 0; i < window_len; i++)
    window[i] = 0.50 - 0.50 * cosf(a * i);
}

void WindowFunction::Hamming(float* window, int32_t window_len, bool periodic) {
  float a = PI2 / (periodic ? window_len : window_len - 1.0f);
  for (int32_t i = 0; i < window_len; i++)
    window[i] = 0.54 - 0.46 * cosf(a * i);
}

void WindowFunction::Rectangular(float* window, int32_t window_len) {
  for (int32_t i = 0; i < window_len; i++) window[i] = 1.0f;
}

void WindowFunction::SqrtHanning(float* window, int32_t window_len,
                                 bool periodic) {
  Hanning(window, window_len, periodic);
  for (int32_t i = 0; i < window_len; i++) window[i] = sqrtf(window[i]);
}

void WindowFunction::Blackman(float* window, int32_t window_len,
                              bool periodic) {
  float a = PI2 / (periodic ? window_len : window_len - 1.0f);
  for (int32_t i = 0; i < window_len; i++)
    window[i] = 0.42 - 0.5 * cosf(a * i) + 0.08 * cosf(2 * a * i);
}
void WindowFunction::Bartlett(float* window, int32_t window_len,
                              bool periodic) {
  float a = 2.0f / (periodic ? window_len : window_len - 1.0f);
  for (int32_t i = 0; i < window_len; i++)
    window[i] = 1.0f - fabsf(i * a - 1.0f);
}

void WindowFunction::Generate(const std::string& name, float* window,
                              int32_t window_len, bool periodic) {
  ASSERT(window_len >= 0);
  memset(window, 0, sizeof(float) * window_len);
  if (name == "hann")
    Hanning(window, window_len, periodic);
  else if (name == "sqrthann")
    SqrtHanning(window, window_len, periodic);
  else if (name == "hamm")
    Hamming(window, window_len, periodic);
  else if (name == "rect")
    Rectangular(window, window_len);
  else if (name == "blackman")
    Blackman(window, window_len, periodic);
  else if (name == "bartlett")
    Bartlett(window, window_len, periodic);
  else
    LOG_FAIL << "Unknown type of the window: " << name;
}

}  // namespace aps
