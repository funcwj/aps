// Copyright 2021 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

#ifndef CSRC_BASE_PIPELINE_H_
#define CSRC_BASE_PIPELINE_H_

#include <torch/script.h>

#include <queue>
#include <string>
#include <vector>

#include "base/stft.h"
#include "utils/log.h"

namespace aps {

class PipelineBase {
 public:
  virtual void Reset() = 0;

  virtual void Process(const torch::Tensor &chunk) = 0;

  virtual bool IsDone() = 0;

  virtual torch::Tensor Pop(int32_t dim = 0) = 0;
};

class Frame : public PipelineBase {
 public:
  Frame(int32_t frame_len, int32_t frame_hop)
      : frame_len_(frame_len), frame_hop_(frame_hop) {
    Reset();
  }

  virtual void Reset() {
    while (!queue_.empty()) queue_.pop();
    cache_ = torch::zeros(0);
  }
  // Push one audio chunk
  virtual void Process(const torch::Tensor &chunk);
  // Current queue is empty
  virtual bool IsDone() { return queue_.size() == 0; }
  // Pop one frame
  virtual torch::Tensor Pop(int32_t dim = 0);

 private:
  int32_t frame_len_, frame_hop_;
  torch::Tensor cache_;
  std::queue<torch::Tensor> queue_;
};

class STFT : public Frame {
 public:
  STFT(int32_t frame_len, int32_t frame_hop, const std::string &window)
      : Frame(frame_len, frame_hop), stft_(frame_len, frame_hop, window) {}

  // Pop one frame
  virtual torch::Tensor Pop(int32_t dim = 0) {
    return stft_.Compute(Frame::Pop());
  }

 private:
  StreamingTorchSTFT stft_;
};

class Feature : public STFT {
  using Module = torch::jit::script::Module;

 public:
  Feature(const Module &feature, int32_t frame_len, int32_t frame_hop,
          const std::string &window)
      : STFT(frame_len, frame_hop, window), feature_(feature) {}

  // Pop one frame
  virtual torch::Tensor Pop(int32_t dim = 0) {
    return feature_.forward({STFT::Pop()}).toTensor();
  }

 private:
  Module feature_;
};

class Context : public PipelineBase {
 public:
  Context(int32_t lctx, int32_t rctx, int32_t chunk = 1, int32_t stride = -1)
      : lctx_(lctx), rctx_(rctx), chunk_(chunk) {
    if (stride <= 0)
      stride_ = chunk;
    else
      stride_ = stride;
    ASSERT(stride_ >= 1 && chunk >= 1);
  }

  virtual void Reset() { queue_.clear(); }

  virtual bool IsDone() { return queue_.size() < chunk_ + rctx_ + lctx_; }
  // Push one frame
  virtual void Process(const torch::Tensor &one_frame);
  // Pop one chunk if possible
  virtual torch::Tensor Pop(int32_t dim = 0);

  // Set done, add right context
  void SetDone();

 private:
  int32_t lctx_, rctx_, chunk_, stride_;
  std::vector<torch::Tensor> queue_;
};

}  // namespace aps
#endif  // CSRC_BASE_PIPELINE_H_
