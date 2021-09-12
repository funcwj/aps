
// Copyright 2018 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

#include "utils/fft.h"

namespace aps {

void FFTComputer::ComplexBitReverse(float *cplx_values, int32_t num_values) {
  for (int32_t j = 0, i = 0; i < num_values - 1; i++) {
    if (i < j) {
      std::swap(REAL_PART(cplx_values, i), REAL_PART(cplx_values, j));
      std::swap(IMAG_PART(cplx_values, i), IMAG_PART(cplx_values, j));
    }
    int32_t m = num_values >> 1;
    while (j >= m) {
      j = j - m;
      m = m >> 1;
    }
    j = j + m;
  }
}

void FFTComputer::ComplexFFT(float *cplx_values, int32_t num_samples,
                             bool invert) {
  int32_t n = num_samples >> 1, s = register_size_ / n;

  ComplexBitReverse(cplx_values, n);

  int32_t i, j, m = 1, cnt, inc, k;
  float WR, WI, Ri, Ii, Rj, Ij;

  while (m < n) {
    cnt = 0, inc = n / (m << 1);
    while (cnt < inc) {
      i = cnt * m * 2;
      for (int t = 0; t < m; t++, i++) {
        j = i + m, k = t * inc;
        // WR = cos(PI * k * 2 / n), WI = sin(PI * k * 2 / n);
        WR = cos_table_[k * s],
        WI = (invert ? sin_table_[k * s] : -sin_table_[k * s]);
        Rj = REAL_PART(cplx_values, j), Ij = IMAG_PART(cplx_values, j);
        Ri = REAL_PART(cplx_values, i), Ii = IMAG_PART(cplx_values, i);
        REAL_PART(cplx_values, i) = Ri + WR * Rj - WI * Ij;
        IMAG_PART(cplx_values, i) = Ii + WR * Ij + WI * Rj;
        REAL_PART(cplx_values, j) = Ri - WR * Rj + WI * Ij;
        IMAG_PART(cplx_values, j) = Ii - WR * Ij - WI * Rj;
      }
      cnt++;
    }
    m = m << 1;
  }

  if (invert) {
    for (i = 0; i < num_samples; i++) cplx_values[i] = cplx_values[i] / n;
  }
}

void FFTComputer::RealFFT(float *src, int32_t num_samples, bool invert) {
  if (num_samples != register_size_) {
    LOG_FAIL << "Assert num_samples == register_size_ failed, " << num_samples
             << " vs " << register_size_;
  }

  int32_t n = num_samples >> 1, s = register_size_ / num_samples;
  std::copy(src, src + num_samples, fft_cache_.begin());
  if (!invert) {
    ComplexFFT(fft_cache_.data(), num_samples, invert);
  }

  float FR, FI, GR, GI, YR, YI, CYR, CYI, cosr, sinr;

  for (int r = 1; r < n; r++) {
    YR = REAL_PART(fft_cache_, r), CYR = REAL_PART(fft_cache_, n - r);
    YI = IMAG_PART(fft_cache_, r), CYI = IMAG_PART(fft_cache_, n - r);
    FR = (YR + CYR) / 2, FI = (YI - CYI) / 2;
    GR = (YI + CYI) / 2, GI = (CYR - YR) / 2;
    cosr = invert ? -cos_table_[r * s] : cos_table_[r * s];
    sinr = sin_table_[r * s];
    REAL_PART(src, r) = FR + cosr * GR - sinr * GI;
    IMAG_PART(src, r) = FI + cosr * GI + sinr * GR;
  }
  FR = REAL_PART(fft_cache_, 0);
  GR = IMAG_PART(fft_cache_, 0);
  REAL_PART(src, 0) = invert ? (FR + GR) * 0.5 : FR + GR;
  IMAG_PART(src, 0) = invert ? (FR - GR) * 0.5 : FR - GR;
  if (invert) {
    ComplexFFT(src, num_samples, invert);
  }
}

}  // namespace aps
