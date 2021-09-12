// Copyright 2021 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

#include "enh/transformer.h"
#include "utils/args.h"
#include "utils/wav.h"
#include "utils/timer.h"

int main(int argc, char const *argv[]) {
  using namespace aps;

  try {
    const std::string description =
        "Command to perform real time speech enhancement using Transformer. "
        "Please using the script cmd/export_for_libtorch.py to export "
        "TorchScript models and the available real time speech enhancement "
        "models supported by the TorchScript are put under aps/rt_sse/*py";
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
    LOG_INFO << "Audio information: " << wav_reader.Info();
    const int32_t num_channels = wav_reader.NumChannels(),
                  num_samples = wav_reader.NumSamples(),
                  sample_rate = wav_reader.SampleRate();
    const float duration = num_samples / sample_rate;
    WavWriter wav_writer(enhan_wav, sample_rate, num_channels);

    int32_t write_samples = 0, read_samples = 0;
    torch::Tensor noisy = torch::zeros(num_channels * wav_chunk), enhan;

    Timer timer;
    while (true) {
      if (wav_reader.Done()) {
        enhan = nnet.Flush();
        int32_t required_samples = num_samples - write_samples;
        ASSERT(0 <= required_samples);
        ASSERT(enhan.size(0) >= required_samples);
        wav_writer.Write(enhan.data_ptr<float>(), required_samples);
        LOG_INFO << "Flush last " << required_samples << " samples done";
        break;
      }
      read_samples = wav_reader.Read(noisy.data_ptr<float>(), wav_chunk);
      ASSERT(read_samples <= wav_chunk);
      LOG_INFO << "Processing " << read_samples << " samples ...";
      if (nnet.Process(torch::slice(noisy, 0, 0, read_samples), &enhan)) {
        wav_writer.Write(enhan.data_ptr<float>(), enhan.size(0));
        write_samples += enhan.size(0);
        LOG_INFO << "Write " << enhan.size(0) << " samples done";
      }
    }
    wav_writer.Close();
    LOG_INFO << "RTF: " << timer.Elapsed() / duration;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
