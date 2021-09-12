// Copyright 2018 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

#include <torch/script.h>
#include "utils/fft.h"

using namespace aps;
using Tensor = torch::Tensor;

void ToyRealFFT() {
  int N = 8;
  Tensor egs1 = torch::zeros(N);
  for (int i = 0; i < N / 2; i++)
    egs1[i] = 1;
  FFTComputer fft(N);
  fft.RealFFT(egs1.data_ptr<float>(), N, false);
  LOG_INFO << egs1;
}

void ToyComplexFFT() {
  int N = 16;
  Tensor egs1 = torch::zeros(N);
  for (int i = 0; i < N / 4; i++)
    egs1[i * 2] = 1;
  FFTComputer fft(N);
  fft.ComplexFFT(egs1.data_ptr<float>(), N, false);
  LOG_INFO << egs1;
}

void TestRealFFT() {
  int win_size[3] = {32, 64, 128};
  for (size_t i = 0; i < 3; i++) {
    int N = win_size[i];
    FFTComputer fft(N);
    Tensor egs1 = torch::rand(N) * 10;
    Tensor egs2 = egs1.clone();
    // my rfft
    fft.RealFFT(egs1.data_ptr<float>(), N, false);
    Tensor egs3 = torch::zeros(N + 2);
    torch::slice(egs3, 0, 0, N).copy_(egs1);
    egs3[-2] = egs3[1];
    egs3[1] = 0;
    // torch rfft
    egs2 = torch::fft_rfft(egs2, N, -1);
    egs2 = torch::view_as_real(egs2).view(-1);
    ASSERT(torch::allclose(egs3, egs2, 1.0e-05, 1.0e-06));
  }
}

void TestComplexFFT() {
  int win_size[3] = {32, 64, 128};
  for (size_t i = 0; i < 3; i++) {
    int N = win_size[i];
    FFTComputer fft(N);
    Tensor egs1 = torch::rand(N) * 10;
    Tensor egs2 = egs1.clone();
    fft.ComplexFFT(egs1.data_ptr<float>(), N, false);
    fft.ComplexFFT(egs1.data_ptr<float>(), N, true);
    ASSERT(torch::allclose(egs1, egs2, 1.0e-05, 1.0e-06));
  }
}

void TestRealiFFT() {
  int win_size[3] = {32, 64, 128};
  for (size_t i = 0; i < 3; i++) {
    int N = win_size[i];
    FFTComputer fft(N);
    Tensor egs1 = torch::rand(N) * 10;
    Tensor egs2 = egs1.clone();
    fft.RealFFT(egs1.data_ptr<float>(), N, false);
    fft.RealFFT(egs1.data_ptr<float>(), N, true);
    ASSERT(torch::allclose(egs1, egs2, 1.0e-05, 1.0e-06));
  }
}

int main(int argc, char const *argv[]) {
  TestComplexFFT();
  TestRealiFFT();
  return 0;
}
