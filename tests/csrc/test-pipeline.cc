// Copyright 2021 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

#include <torch/script.h>

#include "base/pipeline.h"

using namespace aps;
using Tensor = torch::Tensor;

void TestFrame() {
  int32_t num_samples = 100, chunk_size = 25;
  int32_t frame_len = 20, frame_hop = 10;
  Frame frm(frame_len, frame_hop);
  for (int32_t c = 0; c < num_samples; c += chunk_size)
    frm.Process(torch::arange(c, c + chunk_size));
  while (!frm.Done()) {
    LOG_INFO << frm.Pop();
  }
}

void TestContext() {
  int32_t num_frames = 50;
  int32_t lctx = 2, rctx = 3, chunk = 2;
  Context ctx(lctx, rctx, chunk);
  Tensor buffer;
  for (int32_t t = 0; t < num_frames; t++) {
    ctx.Process(torch::ones(3) * (t + 1));
  }
  // ctx.SetDone();
  while (!ctx.Done()) {
    LOG_INFO << ctx.Pop();
  }
}

void TestToy() {
  std::queue<Tensor> q;
  for (int32_t i = 0; i < 10; i++) {
    Tensor egs = torch::ones(10) * i;
    q.push(std::move(egs));
  }
  while (!q.empty()) {
    LOG_INFO << q.front();
    q.pop();
  }
}

int main(int argc, char const *argv[]) {
  TestToy();
  TestFrame();
  TestContext();
  return 0;
}
