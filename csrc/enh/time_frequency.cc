// Copyright 2021 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

#include "enh/time_frequency.h"

Module TimeFrequencyNnet::LoadTorchScriptModule(const std::string &path) {
  Module nnet = torch::jit::load(path);
  // make sure in evaluation mode
  nnet.eval();
  // ASSERT(!nnet.is_training());
  return nnet;
}

// mix_stft: F x T x 2
// src_mask: F x T or F x T x 2
torch::Tensor TimeFrequencyNnet::Masking(const torch::Tensor &stft,
                                         const torch::Tensor &mask) {
  ASSERT(stft.dim() == 3);
  ASSERT(mask.dim() == 3 || mask.dim() == 2);
  torch::Tensor enh_stft;
  // real mask
  if (mask.dim() == 2) {
    enh_stft = stft * mask.squeeze(-1);
  } else {
    ASSERT(mask.size(-1) == 2);
    enh_stft =
        torch::view_as_complex(stft) * torch::view_as_complex(mask);
    enh_stft = torch::view_as_real(enh_stft);
  }
  return enh_stft;
}
