// Copyright 2021 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

#ifndef CSRC_ENH_TIME_FREQUENCY_H_
#define CSRC_ENH_TIME_FREQUENCY_H_

#include <torch/script.h>

#include <algorithm>
#include <string>
#include <vector>

#include "base/stft.h"
#include "utils/log.h"

namespace aps {
using Module = torch::jit::script::Module;

struct TimeFrequencyNnetOptions {
  std::string nnet;
  std::string transform;
  int32_t frame_len;
  int32_t frame_hop;
  std::string window;

  TimeFrequencyNnetOptions()
      : nnet(""),
        transform(""),
        frame_len(512),
        frame_hop(256),
        window("hann") {}
};

class TimeFrequencyNnet {
 public:
  Module LoadTorchScriptModule(const std::string &path);

  // Do time-frequency masking
  torch::Tensor Masking(const torch::Tensor &stft, const torch::Tensor &mask);

  virtual bool Process(const torch::Tensor &audio_chunk,
                       torch::Tensor *audio_enhan) = 0;

  virtual void Reset() = 0;

  virtual torch::Tensor Flush() = 0;
};

}  // namespace aps

#endif  // CSRC_ENH_TIME_FREQUENCY_H_
