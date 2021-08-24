// Copyright 2021 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

#include "transform/stft.h"

void STFTBase::SquareWindow(float* dst) {
  for (size_t n = 0; n < window_len_; n++) dst[n] = window_[n] * window_[n];
}

void STFTBase::AddWindow(float* src, int32_t src_length) {
  ASSERT(src_length == window_len_);
  for (size_t n = 0; n < src_length; n++) src[n] *= window_[n];
}

void STFTBase::RealFFT(float* src, int32_t src_length, bool invert) {
  fft_computer_->RealFFT(src, src_length, invert);
}

void StreamingSTFT::Compute(float* src, int32_t src_length, float* dst) {
  memcpy(dst, src, sizeof(float) * src_length);
  if (pre_emphasis_ > 0) {
    for (int32_t n = src_length - 1; n > 0; n--)
      dst[n] -= pre_emphasis_ * dst[n - 1];
  }
  AddWindow(dst, src_length);
  RealFFT(dst, src_length, false);
}

void StreamingiSTFT::Compute(float* src, int32_t src_length, float* dst) {
  memcpy(dst, src, sizeof(float) * src_length);
  RealFFT(dst, src_length, true);
  AddWindow(dst, src_length);
  // need post processing
  SquareWindow(win_denorm_);
  for (int32_t n = 0; n < overlap_len_; n++) {
    dst[n] += wav_cache_[n];
    win_denorm_[n] += win_cache_[n];
  }
  memcpy(wav_cache_, dst, sizeof(float) * overlap_len_);
  memcpy(win_cache_, win_denorm_, sizeof(float) * overlap_len_);
  for (int32_t n = 0; n < src_length; n++) dst[n] /= (win_denorm_[n] + EPS_F32);
}

void StreamingiSTFT::Flush(float* dst) {
  for (int32_t n = 0; n < overlap_len_; n++)
    dst[n] = wav_cache_[n] / (win_cache_[n] + EPS_F32);
}
