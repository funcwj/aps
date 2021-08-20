// Copyright 2018 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

#ifndef CSRC_UTILS_MATH_H_
#define CSRC_UTILS_MATH_H_

#include <cmath>
#include <limits>

const float EPS_F32 = std::numeric_limits<float>::epsilon();

const int32_t MAX_INT32 = std::numeric_limits<int32_t>::max();
const int16_t MAX_INT16 = std::numeric_limits<int16_t>::max();
const int8_t MAX_INT8 = std::numeric_limits<int8_t>::max();

#endif  // CSRC_UTILS_MATH_H_
