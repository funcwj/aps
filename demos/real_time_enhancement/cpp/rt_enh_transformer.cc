// Copyright 2021 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

#include "enh/transformer.h"
#include "utils/args.h"
#include "utils/wav.h"

using namespace aps;

int main(int argc, char const *argv[]) {
  const std::string description =
      "Command to perform transformer based real time speech enhancement";
  ArgParser parser(description);

  torch::NoGradGuard no_grad;
  TimeFrequencyNnetOptions opts;

  std::string noisy_wav, enhan_wav;
  int32_t wav_chunk = 16000;

  parser.AddArgument("nnet", &opts.nnet,
                     "Exported transformer model using TorchScript", true);
  parser.AddArgument("transform", &opts.transform,
                     "Exported feature transform layer using TorchScript",
                     true);
  parser.AddArgument("noisy-wav", &noisy_wav,
                     "Path of the input noisy wave file", true);
  parser.AddArgument("enhan-wav", &enhan_wav,
                     "Path of the enhancement wave file", true);
  parser.AddArgument("wav-chunk", &enhan_wav,
                     "Wave chunk size for program to process at one time",
                     false);

  parser.ReadCommandArgs(argc, argv);

  TransformerNnet nnet(opts);

  WavReader wav_reader(noisy_wav);
  int32_t num_channels = wav_reader.NumChannels(),
          num_samples = wav_reader.NumSamples();
  WavWriter wav_writer(enhan_wav, wav_reader.SampleRate(), num_channels);
  LOG_INFO << wav_writer.Info();

  int32_t write_samples = 0, enhan_samples = 0, read_samples = 0;
  torch::Tensor noisy = torch::zeros(num_channels * wav_chunk), enhan;
  while (true) {
    if (wav_reader.Done()) break;
    read_samples = wav_reader.Read(noisy.data_ptr<float>(), wav_chunk);
    ASSERT(read_samples <= wav_chunk);
    LOG_INFO << "Processing " << read_samples << " samples ...";
    if (nnet.Process(noisy, &enhan)) {
      enhan_samples = enhan.size(0);
      wav_writer.Write(enhan.data_ptr<float>(), enhan_samples);
      write_samples += enhan_samples;
      LOG_INFO << "Write " << enhan_samples << " samples done";
    }
  }
  wav_writer.Close();
  return 0;
}
