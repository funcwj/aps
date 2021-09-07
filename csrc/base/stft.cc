// Copyright 2021 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

#include "base/stft.h"

void TorchSTFTBase::GenerateWindow(const std::string& name,
                                   int32_t window_len) {
  if (name == "hann")
    window_ = torch::hann_window(window_len, true, torch::kFloat32);
  else if (name == "sqrthann")
    window_ = torch::hann_window(window_len, true, torch::kFloat32).sqrt();
  else if (name == "hamm")
    window_ = torch::hamming_window(window_len, true, torch::kFloat32);
  else if (name == "rect")
    window_ = torch::ones(window_len);
  else if (name == "blackman")
    window_ = torch::blackman_window(window_len, true, torch::kFloat32);
  else if (name == "bartlett")
    window_ = torch::bartlett_window(window_len, true, torch::kFloat32);
  else
    LOG_FAIL << "Unknown type of the window: " << name;
}

torch::Tensor StreamingTorchSTFT::Compute(const torch::Tensor& frame) {
  ASSERT(frame.dim() == 1);
  ASSERT(frame.size(0) == FrameLength());
  torch::Tensor frame_in = frame * window_;
  if (ZeroPad() > 0)
    frame_in = tf::pad(frame_in, tf::PadFuncOptions({0, ZeroPad()}));
  torch::Tensor stft = torch::fft_rfft(frame_in, FFTSize(), -1);
  return torch::view_as_real(stft);
}

torch::Tensor StreamingTorchiSTFT::Compute(const torch::Tensor& stft) {
  // F x 2
  ASSERT(stft.dim() == 2);
  ASSERT(stft.size(0) == FFTSize() / 2 + 1);
  torch::Tensor frame =
      torch::fft_irfft(torch::view_as_complex(stft), FFTSize(), -1);
  if (ZeroPad() > 0) frame = torch::slice(frame, 0, 0, FrameLength());
  return Normalization(frame * window_);
}

torch::Tensor StreamingTorchiSTFT::Normalization(const torch::Tensor& frame) {
  torch::Tensor denorm = window_ * window_;
  torch::slice(frame, 0, 0, overlap_len_) += wav_cache_;
  torch::slice(denorm, 0, 0, overlap_len_) += win_cache_;
  win_cache_ = torch::slice(denorm, 0, FrameHop(), FFTSize());
  wav_cache_ = torch::slice(frame, 0, FrameHop(), FFTSize());
  torch::Tensor frame_norm = frame / (denorm + EPS_F32);
  return torch::slice(frame_norm, 0, 0, FrameHop());
}

torch::Tensor StreamingTorchiSTFT::Flush() {
  return wav_cache_ / (win_cache_ + EPS_F32);
}

void StreamingTorchiSTFT::Reset() {
  wav_cache_ = torch::zeros(overlap_len_);
  win_cache_ = torch::zeros_like(wav_cache_);
}
