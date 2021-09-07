// Copyright 2018 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
// 2018.7 in https://github.com/funcwj/asr-utils/blob/master/fft/fft-computer.h

#ifndef CSRC_UTILS_FFT_H_
#define CSRC_UTILS_FFT_H_

#include <utility>
#include <vector>
#include <algorithm>
#include "utils/math.h"
#include "utils/log.h"

// Class for FFT computation

class FFTComputer {
 public:
  explicit FFTComputer(int32_t register_size) : register_size_(register_size) {
    ASSERT(RoundUpToNearestPowerOfTwo(register_size) == register_size);
    int32_t table_size = register_size >> 1;
    cos_table_.resize(table_size);
    sin_table_.resize(table_size);
    // pre-compute cos/sin values for FFT
    for (int32_t k = 0; k < table_size; k++) {
      cos_table_[k] = cosf32(PI * k / table_size);
      sin_table_[k] = sinf32(PI * k / table_size);
    }
    // for RealFFT data cache
    fft_cache_.resize(register_size);
  }

  // Compute (inverse)FFT values
  // cplx_values: [R0, I0, R1, I1, ... R(N - 1), I(N - 1)]
  // num_samples: length of cplx_values(N)
  void ComplexFFT(float *cplx_values, int32_t num_samples, bool invert);

  // Compute RealFFT values
  // src: [R0, R1, ..., R(N - 1)]
  void RealFFT(float *src, int32_t num_samples, bool invert);

 private:
  // Required 2^N
  int32_t register_size_;
  // Precomputed values
  std::vector<float> cos_table_, sin_table_, fft_cache_;
  // BitReverse for complex values
  void ComplexBitReverse(float *cplx_values, int32_t num_values);
};

#endif  // CSRC_UTILS_FFT_H_
