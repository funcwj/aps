// Copyright 2018 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

#include <torch/script.h>

#include "utils/wav.h"

using Tensor = torch::Tensor;
const std::string wav_egs[3] = {"egs1.wav", "egs2.wav", "egs3.wav"};
const std::string prefix = "data/transform";

void TestWavIO() {
  for (size_t i = 0; i < 3; i++) {
    WavReader wav_reader(prefix + "/" + wav_egs[i]);
    int32_t num_channels = wav_reader.NumChannels();
    WavWriter wav_writer(prefix + "/copy.wav", wav_reader.SampleRate(),
                         num_channels, 2);
    LOG_INFO << wav_writer.Info();
    int32_t chunk = 3200;  // 0.2s
    size_t read = 0, write = 0;
    Tensor cache = torch::zeros(num_channels * chunk, torch::kFloat32);
    float *cache_ptr = cache.data_ptr<float>();
    while (true) {
      if (wav_reader.Done()) break;
      read = wav_reader.Read(cache_ptr, chunk);
      write = wav_writer.Write(cache_ptr, read);
      ASSERT(read == write);
    }
    wav_writer.Close();
    LOG_INFO << wav_reader.Info();
    LOG_INFO << wav_writer.Info();
    WavReader egs_reader(prefix + "/copy.wav");
    LOG_INFO << egs_reader.Info();
  }
}

int main(int argc, char const *argv[]) {
  TestWavIO();
  return 0;
}
