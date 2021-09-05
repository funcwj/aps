// Copyright 2021 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

#ifndef CSRC_TRANSFORM_STFT_H_
#define CSRC_TRANSFORM_STFT_H_

#include <torch/csrc/api/include/torch/nn/functional.h>
#include <torch/script.h>

#include <algorithm>
#include <string>

#include "utils/log.h"
#include "utils/math.h"

namespace tf = torch::nn::functional;

class TorchSTFTBase {
 public:
  TorchSTFTBase(int32_t window_len = 512, int32_t frame_hop = 256,
                const std::string& window = "hann",
                const std::string& mode = "librosa")
      : frame_hop_(frame_hop), mode_(mode) {
    ASSERT(mode == "librosa" || mode == "kaldi");
    GenerateWindow(window, window_len);
    fft_size_ =
        mode == "kaldi" ? RoundUpToNearestPowerOfTwo(window_len) : window_len;
    if (mode == "kaldi") {
      window_ =
          tf::pad(window_, tf::PadFuncOptions({0, fft_size_ - window_len}));
    } else {
      int32_t lpad = (fft_size_ - window_len) / 2;
      window_ = tf::pad(
          window_, tf::PadFuncOptions({lpad, fft_size_ - window_len - lpad}));
    }
    frame_len_ = mode == "kaldi" ? window_len : fft_size_;
  }

  // return the frame length used for STFT
  int32_t FrameLength() const { return frame_len_; }
  // frame hop size
  int32_t FrameHop() const { return frame_hop_; }
  // FFT size
  int32_t FFTSize() const { return fft_size_; }
  // window
  torch::Tensor window_;

 private:
  std::string mode_;
  int32_t frame_len_, frame_hop_, fft_size_;
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
    wav_cache_ = torch::zeros(overlap_len_);
    win_cache_ = torch::zeros_like(wav_cache_);
  }

  // Run iSTFT for one frame (with normalization)
  torch::Tensor Compute(const torch::Tensor& stft);
  // Calling it at last
  torch::Tensor Flush();

 private:
  torch::Tensor wav_cache_, win_cache_;
  int32_t overlap_len_;
  torch::Tensor Normalization(const torch::Tensor& frame);
};

#endif  // CSRC_TRANSFORM_STFT_H_
