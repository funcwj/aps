// Copyright 2021 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

#ifndef CSRC_UTILS_STFT_H_
#define CSRC_UTILS_STFT_H_

#include <algorithm>
#include <string>
#include <vector>

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
      : frame_hop_(frame_hop), mode_(mode) {
    ASSERT(mode == "librosa" || mode == "kaldi");
    // due to implementation of FFT, we need fft_size as 2^N
    fft_size_ = RoundUpToNearestPowerOfTwo(window_len);
    fft_computer_ = new FFTComputer(fft_size_);
    window_.resize(fft_size_, 0);
    // librosa & kaldi is different
    window_gen_.Generate(
        window,
        window_.data() + (mode == "kaldi" ? 0 : (fft_size_ - window_len) / 2),
        window_len, true);
    frame_len_ = mode == "librosa" ? fft_size_ : window_len;
  }

  // return the frame length used for STFT
  int32_t FrameLength() const { return frame_len_; }
  // frame hop size
  int32_t FrameHop() const { return frame_hop_; }
  // FFT size, must be 2^N
  int32_t FFTSize() const { return fft_size_; }

  void Windowing(float* frame, int32_t frame_len);
  void RealFFT(float* data_ptr, int32_t data_len, bool invert);

  ~STFTBase() {
    if (fft_computer_) delete fft_computer_;
  }
  // public: we need access it in iSTFT
  std::vector<float> window_;

 private:
  std::string mode_;
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

  // Run STFT for one frame
  void Compute(float* frame, int32_t frame_len, float* stft);
};

class StreamingiSTFT : public STFTBase {
 public:
  StreamingiSTFT(int32_t frame_len = 512, int32_t frame_hop = 256,
                 const std::string& window = "hann",
                 const std::string& mode = "librosa")
      : STFTBase(frame_len, frame_hop, window, mode) {
    overlap_len_ = FrameLength() - FrameHop();
    wav_cache_.resize(overlap_len_, 0);
    win_cache_.resize(overlap_len_, 0);
    win_denorm_.resize(FrameLength());
  }

  // Run iSTFT for one frame (with normalization)
  void Compute(float* stft, int32_t frame_len, float* frame);
  // Calling it at last
  void Flush(float* frame);

 private:
  std::vector<float> wav_cache_, win_cache_, win_denorm_;
  int32_t overlap_len_;
  void Normalization(float* frame, int32_t frame_len);
};

#endif  // CSRC_UTILS_STFT_H_
