// Copyright 2021 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

#include "enh/transformer.h"
#include "utils/wav.h"

using namespace aps;

const int32_t wav_chunk = 16000;
const std::string nnet = "debug/avg.epoch29-38.scripted.pt";
const std::string transform = "debug/avg.epoch29-38.transform.pt";
const std::string noisy_wav = "debug/noisy.wav";
const std::string enhan_wav = "debug/enhan.wav";

int main(int argc, char const *argv[]) {
  torch::NoGradGuard no_grad;
  TimeFrequencyNnetOptions opts;
  opts.nnet = nnet;
  opts.transform = transform;
  TransformerNnet nnet(opts);

  WavReader wav_reader(noisy_wav);
  int32_t num_channels = wav_reader.NumChannels();
  WavWriter wav_writer(enhan_wav, wav_reader.SampleRate(), num_channels, 2);
  LOG_INFO << wav_writer.Info();

  torch::Tensor noisy = torch::zeros(num_channels * wav_chunk), enhan;
  while (true) {
    if (wav_reader.Done()) {
      enhan = nnet.Flush();
      wav_writer.Write(enhan.data_ptr<float>(), enhan.size(0));
      break;
    }
    int32_t read = wav_reader.Read(noisy.data_ptr<float>(), wav_chunk);
    ASSERT(read <= wav_chunk);
    LOG_INFO << "Processing " << read << " samples";
    if (nnet.Process(noisy, &enhan)) {
      wav_writer.Write(enhan.data_ptr<float>(), enhan.size(0));
    }
  }
  wav_writer.Close();
  return 0;
}
