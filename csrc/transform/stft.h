// Copyright 2021 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

#ifndef CSRC_TRANSFORM_STFT_H_
#define CSRC_TRANSFORM_STFT_H_

#include "utils/fft.h"
#include "utils/log.h"
#include "utils/math.h"
#include "utils/window.h"

class STFTBase {
 public:
  STFTBase(int32_t frame_len = 512, int32_t frame_hop = 256,
           const std::string& window = "hann")
      : frame_len_(frame_len), frame_hop_(frame_hop), window_str_(window) {
    window_len_ = RoundUpToNearestPowerOfTwo(frame_len);
    fft_computer_ = new FFTComputer(window_len_);
    window_ = new float[window_len_];
    memset(window_, 0, sizeof(float) * window_len_);
    window_gen_.Generate(window, window_ + (window_len_ - frame_len) / 2,
                         frame_len, true);
  }

  int32_t WindowLength() const { return window_len_; }
  int32_t FrameShift() const { return frame_hop_; }

  void SquareWindow(float* dst);
  void AddWindow(float* src, int32_t src_length);
  void RealFFT(float* src, int32_t src_length, bool invert);

  ~STFTBase() {
    if (window_) delete[] window_;
    if (fft_computer_) delete fft_computer_;
  }

 private:
  float* window_;
  std::string window_str_;
  int32_t frame_len_, frame_hop_, window_len_;
  FFTComputer* fft_computer_;
  WindowFunction window_gen_;
};

class StreamingSTFT : public STFTBase {
 public:
  StreamingSTFT(int32_t frame_len = 512, int32_t frame_hop = 256,
                const std::string& window = "hann", float pre_emphasis = 0.0)
      : STFTBase(frame_len, frame_hop, window), pre_emphasis_(pre_emphasis) {}

  void Compute(float* src, int32_t window_length, float* dst);

 private:
  float pre_emphasis_;
};

class StreamingiSTFT : public STFTBase {
 public:
  StreamingiSTFT(int32_t frame_len = 512, int32_t frame_hop = 256,
                 const std::string& window = "hann")
      : STFTBase(frame_len, frame_hop, window) {
    overlap_len_ = WindowLength() - FrameShift();
    wav_cache_ = new float[overlap_len_];
    win_cache_ = new float[overlap_len_];
    memset(wav_cache_, 0, sizeof(float) * overlap_len_);
    memset(win_cache_, 0, sizeof(float) * overlap_len_);
    win_denorm_ = new float[WindowLength()];
  }

  ~StreamingiSTFT() {
    if (win_cache_) delete[] win_cache_;
    if (wav_cache_) delete[] wav_cache_;
    if (win_denorm_) delete[] win_denorm_;
  }

  void Compute(float* src, int32_t window_length, float* dst);
  void Flush(float* dst);

 private:
  float* wav_cache_;
  float* win_cache_;
  float* win_denorm_;
  int32_t overlap_len_;
};

#endif  // CSRC_TRANSFORM_STFT_H_
