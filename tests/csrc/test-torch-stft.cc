// Copyright 2021 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

#include <torch/script.h>

#include "base/stft.h"
#include "utils/wav.h"

void TestSTFT() {
  const std::string src_wav = "data/transform/egs1.wav";
  const std::string dst_wav = "data/transform/copy.wav";
  WavReader wav_reader(src_wav);
  LOG_INFO << wav_reader.Info();
  WavWriter wav_writer(dst_wav, wav_reader.SampleRate(),
                       wav_reader.NumChannels(), 2);
  size_t num_samples = wav_reader.NumSamples();
  torch::Tensor src_data = torch::zeros(num_samples);
  float *src_wav_ptr = src_data.data_ptr<float>();
  size_t read = wav_reader.Read(src_wav_ptr, num_samples);

  ASSERT(read == num_samples);
  const int32_t window_len = 400, frame_hop = 256;
  StreamingTorchSTFT stft(window_len, frame_hop, "hann", "librosa");
  StreamingTorchiSTFT istft(window_len, frame_hop, "hann", "librosa");
  int32_t frame_len = stft.FrameLength(), fft_size = stft.FFTSize();

  torch::Tensor frame;
  for (int32_t n = 0; n + frame_len < num_samples; n += frame_hop) {
    frame = istft.Compute(
        stft.Compute(torch::slice(src_data, 0, n, n + frame_len)));
    wav_writer.Write(frame.data_ptr<float>(), frame_hop);
  }
  frame = istft.Flush();
  ASSERT(frame.size(0) == frame_len - frame_hop);
  wav_writer.Write(frame.data_ptr<float>(), frame.size(0));
  wav_writer.Close();
}

int main(int argc, char const *argv[]) {
  TestSTFT();
  return 0;
}
