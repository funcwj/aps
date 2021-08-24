// Copyright 2021 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

#ifndef CSRC_TRANSFORM_STFT_H_
#define CSRC_TRANSFORM_STFT_H_

#include <string>

#include "utils/fft.h"
#include "utils/log.h"
#include "utils/math.h"
#include "utils/window.h"

/*
For mode == "librosa", window_len must be 2^N, which is used for front-end
models, e.g., speech enhancement, speech separation. mode == "kaldi" is used for
ASR tasks
*/
class STFTBase {
 public:
  STFTBase(int32_t window_len = 512, int32_t frame_hop = 256,
           const std::string& window = "hann",
           const std::string& mode = "librosa")
      : frame_hop_(frame_hop), win_str_(window), mode_(mode) {
    ASSERT(mode == "librosa" || mode == "kaldi");
    // due to implementation of FFT, we need fft_size as 2^N
    fft_size_ = RoundUpToNearestPowerOfTwo(window_len);
    fft_computer_ = new FFTComputer(fft_size_);
    window_ = new float[fft_size_];
    memset(window_, 0, sizeof(float) * fft_size_);
    // librosa & kaldi is different
    window_gen_.Generate(
        window, window_ + (mode == "kaldi" ? 0 : (fft_size_ - window_len) / 2),
        window_len, true);
    frame_len_ = mode == "librosa" ? fft_size_ : window_len;
  }

  // return the frame length used for STFT
  int32_t FrameLength() const { return frame_len_; }
  // frame hop size
  int32_t FrameHop() const { return frame_hop_; }
  // FFT size, must be 2^N
  int32_t FFTSize() const { return fft_size_; }

  void SquareOfWindow(float* dst);
  void AddWindow(float* src, int32_t src_length);
  void RealFFT(float* src, int32_t src_length, bool invert);

  ~STFTBase() {
    if (window_) delete[] window_;
    if (fft_computer_) delete fft_computer_;
  }

 private:
  float* window_;
  std::string win_str_, mode_;
  int32_t frame_len_, frame_hop_, fft_size_;
  FFTComputer* fft_computer_;
  WindowFunction window_gen_;
};

class StreamingSTFT : public STFTBase {
 public:
  StreamingSTFT(int32_t frame_len = 512, int32_t frame_hop = 256,
                const std::string& window = "hann",
                const std::string& mode = "librosa")
      : STFTBase(frame_len, frame_hop, window, mode) {}

  void Compute(float* src, int32_t window_length, float* dst);
};

class StreamingiSTFT : public STFTBase {
 public:
  StreamingiSTFT(int32_t frame_len = 512, int32_t frame_hop = 256,
                 const std::string& window = "hann",
                 const std::string& mode = "librosa")
      : STFTBase(frame_len, frame_hop, window, mode) {
    overlap_len_ = FrameLength() - FrameHop();
    wav_cache_ = new float[overlap_len_];
    win_cache_ = new float[overlap_len_];
    memset(wav_cache_, 0, sizeof(float) * overlap_len_);
    memset(win_cache_, 0, sizeof(float) * overlap_len_);
    win_denorm_ = new float[FrameLength()];
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
  void Normalization(float* src, int32_t src_length);
};

#endif  // CSRC_TRANSFORM_STFT_H_
