// Copyright 2018 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

#include "utils/math.h"

namespace aps {

int32_t RoundUpToNearestPowerOfTwo(int32_t n) {
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return n + 1;
}

float Log32(float linear) { return logf(std::fmax(EPS_F32, linear)); }

bool StringToInt32(const std::string &str, int32_t *out) {
  size_t end = 0;
  int32_t n;
  LOG_INFO << "for int32 " << str;
  n = std::stoi(str, &end);
  if (end != str.size()) return false;
  *out = n;
  return true;
}

bool StringToFloat(const std::string &str, float *out) {
  size_t end = 0;
  double n;
  LOG_INFO << "for float " << str;
  n = std::stod(str, &end);
  if (end != str.size()) return false;
  *out = static_cast<float>(n);
  return true;
}

}  // namespace aps
