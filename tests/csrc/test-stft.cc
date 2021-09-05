// Copyright 2021 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

#include <torch/script.h>

#include "utils/wav.h"
#include "utils/stft.h"

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

void TestSTFT() {
  const std::string src_wav = "data/transform/egs1.wav";
  const std::string dst_wav = "data/transform/copy.wav";
  WavReader wav_reader(src_wav);
  LOG_INFO << wav_reader.Info();
  WavWriter wav_writer(dst_wav, wav_reader.SampleRate(),
                       wav_reader.NumChannels(), 2);
  size_t num_samples = wav_reader.NumSamples();
  Tensor src_data = torch::zeros(num_samples);
  float *src_wav_ptr = src_data.data_ptr<float>();
  size_t read = wav_reader.Read(src_wav_ptr, num_samples);
  ASSERT(read == num_samples);
  const int32_t window_len = 400, frame_hop = 256;
  StreamingSTFT stft(window_len, frame_hop, "hann", "kaldi");
  StreamingiSTFT istft(window_len, frame_hop, "hann", "kaldi");
  int32_t fft_size = stft.FFTSize();
  int32_t frame_len = stft.FrameLength(), overlap_len = frame_len - frame_hop;

  Tensor fft = torch::zeros(fft_size);
  float *fft_ptr = fft.data_ptr<float>();

  Tensor dst_data = torch::zeros(fft_size);
  float *dst_wav_ptr = dst_data.data_ptr<float>();
  for (int32_t n = 0; n + frame_len < num_samples; n += frame_hop) {
    stft.Compute(src_wav_ptr + n, frame_len, fft_ptr);
    istft.Compute(fft_ptr, frame_len, dst_wav_ptr);
    wav_writer.Write(dst_wav_ptr, frame_hop);
  }
  istft.Flush(dst_wav_ptr);
  wav_writer.Write(dst_wav_ptr, overlap_len);
  wav_writer.Close();
}

int main(int argc, char const *argv[]) {
  TestWindow();
  TestSTFT();
  return 0;
}
