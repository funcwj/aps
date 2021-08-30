// Copyright 2021 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

#include "transform/stft.h"

void STFTBase::Windowing(float* frame, int32_t frame_len) {
  ASSERT(frame_len == frame_len_);
  for (size_t n = 0; n < frame_len; n++) frame[n] *= window_[n];
}

void STFTBase::RealFFT(float* data_ptr, int32_t data_len, bool invert) {
  fft_computer_->RealFFT(data_ptr, data_len, invert);
}

void StreamingSTFT::Compute(float* frame, int32_t frame_len, float* stft) {
  ASSERT(frame_len <= FFTSize());
  memset(stft, 0, sizeof(float) * FFTSize());
  memcpy(stft, frame, sizeof(float) * frame_len);
  Windowing(stft, frame_len);
  RealFFT(stft, FFTSize(), false);
}

void StreamingiSTFT::Compute(float* stft, int32_t frame_len, float* frame) {
  ASSERT(frame_len <= FFTSize());
  memcpy(frame, stft, sizeof(float) * FFTSize());
  RealFFT(frame, FFTSize(), true);
  Windowing(frame, frame_len);
  Normalization(frame, frame_len);
}

void StreamingiSTFT::Normalization(float* frame, int32_t frame_len) {
  ASSERT(frame_len == FrameLength());
  for (int32_t n = 0; n < frame_len; n++)
    win_denorm_[n] = window_[n] * window_[n];
  for (int32_t n = 0; n < overlap_len_; n++) {
    frame[n] += wav_cache_[n];
    win_denorm_[n] += win_cache_[n];
  }
  std::copy(frame + FrameHop(), frame + frame_len, wav_cache_.begin());
  std::copy(win_denorm_.begin() + FrameHop(), win_denorm_.end(),
            win_cache_.begin());
  for (int32_t n = 0; n < frame_len; n++)
    frame[n] /= (win_denorm_[n] + EPS_F32);
}

void StreamingiSTFT::Flush(float* frame) {
  for (int32_t n = 0; n < overlap_len_; n++)
    frame[n] = wav_cache_[n] / (win_cache_[n] + EPS_F32);
}
