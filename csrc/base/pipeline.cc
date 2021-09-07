// Copyright 2021 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

#include "base/pipeline.h"

void Frame::Process(const torch::Tensor &chunk) {
  int32_t chunk_len = chunk.size(0), n = 0, cache_len = cache_.size(0);
  if (cache_len != 0) {
    chunk_len += cache_len;
    torch::Tensor chunk_with_cache = torch::cat({cache_, chunk}, 0);
    while (n + frame_len_ < chunk_len) {
      queue_.push(torch::slice(chunk_with_cache, 0, n, n + frame_len_));
      n += frame_hop_;
    }
  } else {
    while (n + frame_len_ < chunk_len) {
      queue_.push(torch::slice(chunk, 0, n, n + frame_len_));
      n += frame_hop_;
    }
  }
  ASSERT(chunk_len - n);
  cache_ = torch::slice(chunk, 0, n, chunk_len);
}

torch::Tensor Frame::Pop(int32_t dim) {
  ASSERT(!IsDone());
  torch::Tensor frame = queue_.front();
  queue_.pop();
  return frame;
}

void Context::Process(const torch::Tensor &one_frame) {
  if (queue_.size() == 0) {
    for (int32_t c = 0; c < lctx_; c++) {
      queue_.push_back(torch::zeros_like(one_frame));
    }
  }
  queue_.push_back(one_frame);
}

void Context::SetDone() {
  for (int32_t c = 0; c < rctx_; c++) {
    queue_.push_back(torch::zeros_like(queue_[0]));
  }
}

torch::Tensor Context::Pop(int32_t dim) {
  ASSERT(!IsDone());
  std::vector<torch::Tensor> frames;
  for (int32_t c = 0; c < chunk_ + rctx_ + lctx_; c++) {
    frames.push_back(queue_[c]);
  }
  torch::Tensor chunk = torch::stack(frames, dim);
  for (int32_t c = 0; c < stride_; c++) queue_.erase(queue_.begin());
  return chunk;
}
