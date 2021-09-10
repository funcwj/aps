// Copyright 2018 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

#ifndef CSRC_UTILS_MATH_H_
#define CSRC_UTILS_MATH_H_

#include <cmath>
#include <limits>
#include <string>
#include "utils/log.h"

namespace aps {

const float EPS_F32 = std::numeric_limits<float>::epsilon();
const float PI = acosf32(-1);
const float PI2 = PI * 2;

const int32_t MAX_INT32 = std::numeric_limits<int32_t>::max();
const int16_t MAX_INT16 = std::numeric_limits<int16_t>::max();
const int8_t MAX_INT8 = std::numeric_limits<int8_t>::max();

#define REAL_PART(complex_values, index) (complex_values[(index) << 1])
#define IMAG_PART(complex_values, index) (complex_values[((index) << 1) + 1])

int32_t RoundUpToNearestPowerOfTwo(int32_t n);

bool StringToInt32(const std::string &str, int32_t *out);
bool StringToFloat(const std::string &str, float *out);

}  // namespace aps

#endif  // CSRC_UTILS_MATH_H_
