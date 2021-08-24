// Copyright 2021 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

#include "transform/stft.h"

void STFTBase::SquareOfWindow(float* dst) {
  for (size_t n = 0; n < frame_len_; n++) dst[n] = window_[n] * window_[n];
}

void STFTBase::AddWindow(float* src, int32_t src_length) {
  ASSERT(src_length == frame_len_);
  for (size_t n = 0; n < src_length; n++) src[n] *= window_[n];
}

void STFTBase::RealFFT(float* src, int32_t src_length, bool invert) {
  fft_computer_->RealFFT(src, src_length, invert);
}

void StreamingSTFT::Compute(float* src, int32_t src_length, float* dst) {
  ASSERT(src_length <= FFTSize());
  memset(dst, 0, sizeof(float) * FFTSize());
  memcpy(dst, src, sizeof(float) * src_length);
  AddWindow(dst, src_length);
  RealFFT(dst, FFTSize(), false);
}

void StreamingiSTFT::Compute(float* src, int32_t src_length, float* dst) {
  ASSERT(src_length <= FFTSize());
  memcpy(dst, src, sizeof(float) * FFTSize());
  RealFFT(dst, FFTSize(), true);
  AddWindow(dst, src_length);
  Normalization(dst, src_length);
}

void StreamingiSTFT::Normalization(float* src, int32_t src_length) {
  SquareOfWindow(win_denorm_);
  for (int32_t n = 0; n < overlap_len_; n++) {
    src[n] += wav_cache_[n];
    win_denorm_[n] += win_cache_[n];
  }
  memcpy(wav_cache_, src + FrameHop(), sizeof(float) * overlap_len_);
  memcpy(win_cache_, win_denorm_ + FrameHop(), sizeof(float) * overlap_len_);
  for (int32_t n = 0; n < src_length; n++) src[n] /= (win_denorm_[n] + EPS_F32);
}

void StreamingiSTFT::Flush(float* dst) {
  for (int32_t n = 0; n < overlap_len_; n++)
    dst[n] = wav_cache_[n] / (win_cache_[n] + EPS_F32);
}
