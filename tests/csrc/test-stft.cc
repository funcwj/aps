// Copyright 2021 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

#include <torch/script.h>

#include "transform/stft.h"

using Tensor = at::Tensor;

void TestWindow() {
  int32_t window_len[3] = {256, 400, 512};
  WindowFunction wf;
  for (int32_t i = 0; i < 3; i++) {
    int32_t wlen = window_len[i];
    Tensor win1 = torch::zeros(wlen);
    wf.Generate("hann", win1.data_ptr<float>(), wlen);
    Tensor win2 = torch::hann_window(wlen, torch::kFloat32);
    ASSERT(torch::allclose(win1, win2, 1.0e-4));
    wf.Generate("sqrthann", win1.data_ptr<float>(), wlen);
    ASSERT(torch::allclose(win1, torch::sqrt(win2), 1.0e-4));
    wf.Generate("hamm", win1.data_ptr<float>(), wlen);
    win2 = torch::hamming_window(wlen, torch::kFloat32);
    ASSERT(torch::allclose(win1, win2, 1.0e-4));
    wf.Generate("bartlett", win1.data_ptr<float>(), wlen);
    win2 = torch::bartlett_window(wlen, torch::kFloat32);
    ASSERT(torch::allclose(win1, win2, 1.0e-4));
  }
}

int main(int argc, char const *argv[]) {
  TestWindow();
  return 0;
}
