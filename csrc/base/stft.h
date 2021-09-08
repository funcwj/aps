// Copyright 2021 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

#ifndef CSRC_BASE_STFT_H_
#define CSRC_BASE_STFT_H_

#include <torch/csrc/api/include/torch/nn/functional.h>
#include <torch/script.h>

#include <algorithm>
#include <string>

#include "utils/log.h"
#include "utils/math.h"

namespace aps {
namespace tf = torch::nn::functional;

// For mode == "kaldi", we always use FFT size as 2^N (for ASR tasks)
// for mode == "librosa", FFT size equals to the window length
class TorchSTFTBase {
 public:
  TorchSTFTBase(int32_t window_len = 512, int32_t frame_hop = 256,
                const std::string& window = "hann",
                const std::string& mode = "librosa")
      : frame_len_(window_len), frame_hop_(frame_hop), mode_(mode) {
    ASSERT(mode == "librosa" || mode == "kaldi");
    ASSERT(window_len % 2 == 0);
    GenerateWindow(window, window_len);
    fft_size_ =
        mode == "kaldi" ? RoundUpToNearestPowerOfTwo(window_len) : window_len;
    zero_pad_ = fft_size_ - frame_len_;
  }

  // return the frame length used for STFT
  int32_t FrameLength() const { return frame_len_; }
  // frame hop size
  int32_t FrameHop() const { return frame_hop_; }
  // FFT size
  int32_t FFTSize() const { return fft_size_; }
  // Zero padding for each frame
  int32_t ZeroPad() const { return zero_pad_; }
  // window
  torch::Tensor window_;

 private:
  std::string mode_;
  int32_t frame_len_, frame_hop_, fft_size_, zero_pad_;
  void GenerateWindow(const std::string& name, int32_t window_len);
};

class StreamingTorchSTFT : public TorchSTFTBase {
 public:
  StreamingTorchSTFT(int32_t frame_len = 512, int32_t frame_hop = 256,
                     const std::string& window = "hann",
                     const std::string& mode = "librosa")
      : TorchSTFTBase(frame_len, frame_hop, window, mode) {}

  // Run STFT for one frame
  torch::Tensor Compute(const torch::Tensor& frame);
};

class StreamingTorchiSTFT : public TorchSTFTBase {
 public:
  StreamingTorchiSTFT(int32_t frame_len = 512, int32_t frame_hop = 256,
                      const std::string& window = "hann",
                      const std::string& mode = "librosa")
      : TorchSTFTBase(frame_len, frame_hop, window, mode) {
    overlap_len_ = FrameLength() - FrameHop();
    Reset();
  }

  // Run iSTFT for one frame (with normalization)
  torch::Tensor Compute(const torch::Tensor& stft);
  // Calling it at last
  torch::Tensor Flush();
  // Reset
  void Reset();

 private:
  torch::Tensor wav_cache_, win_cache_;
  int32_t overlap_len_;
  torch::Tensor Normalization(const torch::Tensor& frame);
};

}  // namespace aps

#endif  // CSRC_BASE_STFT_H_
