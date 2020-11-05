// Copyright 2020 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

#include <sox.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <iostream>
#include <memory>

sox_sample_t expect_sr = 16000;
// same with unet training
sox_sample_t enh_chunk = 64000;
sox_sample_t hop_chunk = 64000 - 256;
sox_sample_t max_int32 = 2147483647;

using Module = torch::jit::script::Module;

void sox_read(const char *fname, std::vector<sox_sample_t> *buffer) {
  sox_format_t *sox_format = sox_open_read(fname, nullptr, nullptr, nullptr);
  sox_signalinfo_t s = sox_format->signal;
  if (s.rate != expect_sr || s.channels != 1) {
    throw std::runtime_error("Expect 16kHz, single-channel audio file");
  }
  buffer->resize(s.length);
  const uint32_t samples_read =
      sox_read(sox_format, buffer->data(), buffer->size());
  if (samples_read == 0 or samples_read != s.length) {
    throw std::runtime_error("Error in reading audio");
  }
  return;
}

void sox_dump(const char *fname, const torch::Tensor &buffer) {
  uint64_t num_samples = buffer.size(0);
  sox_signalinfo_t si = {expect_sr * 1.0, 1, sizeof(int16_t) * 8, num_samples,
                         nullptr};
  sox_format_t *sox_format =
      sox_open_write(fname, &si, nullptr, nullptr, nullptr, nullptr);
  sox_sample_t *ptr = buffer.data_ptr<sox_sample_t>();
  sox_write(sox_format, ptr, num_samples);
  return;
}

void prep_chunks(const torch::Tensor &tensor,
                 std::vector<torch::Tensor> *chunks) {
  int32_t num_samples = tensor.size(0);
  int32_t num_chunks = num_samples / hop_chunk + 1;
  for (int32_t c = 0; c < num_chunks * hop_chunk; c += hop_chunk) {
    torch::Tensor chunk =
        torch::slice(tensor, 0, c, std::min(num_samples, c + enh_chunk));
    if (c + enh_chunk > num_samples) {
      torch::Tensor zero = torch::zeros({enh_chunk}, torch::kFloat32);
      torch::slice(zero, 0, 0, num_samples - c).copy_(chunk);
      chunks->push_back(zero);
    } else {
      chunks->push_back(chunk);
    }
  }
}

void nnet_enhan(Module &nnet, const char *fname_noisy,
                const char *fname_enhan) {
  std::vector<sox_sample_t> noisy;
  auto norm = [](torch::Tensor &vec) -> float_t {
    return vec.abs().max().item().toFloat();
  };

  sox_read(fname_noisy, &noisy);
  std::cout << "Load audio from " << fname_noisy << " done" << std::endl;
  uint32_t num_samples = noisy.size();
  // sox read audio samples, normalized in [-1, 1]
  torch::Tensor noisy_wav =
      torch::from_blob(noisy.data(), {num_samples}, torch::kInt32)
          .to(torch::kFloat32);

  std::cout << "Processing ..." << std::endl;
  // prepare noisy chunks
  std::vector<torch::Tensor> chunks;
  prep_chunks(noisy_wav / float(max_int32), &chunks);
  uint32_t num_chunks = chunks.size();
  // inference
  torch::Tensor batch = torch::stack(chunks, 0);
  std::vector<torch::jit::IValue> egs;
  egs.push_back(batch);
  torch::Tensor output = nnet.forward(egs).toTensor();
  // merge enhan chunks
  torch::Tensor enhan_wav =
      torch::zeros({(num_chunks - 1) * hop_chunk + enh_chunk});
  for (uint32_t c = 0; c < num_samples; c += hop_chunk) {
    torch::slice(enhan_wav, 0, c, c + enh_chunk).copy_(output[c / hop_chunk]);
  }
  // back to int32
  enhan_wav = torch::slice(enhan_wav * float(max_int32), 0, 0, num_samples);
  enhan_wav = (enhan_wav * norm(noisy_wav) / norm(enhan_wav)).to(torch::kInt32);
  // dump audio
  sox_dump(fname_enhan, enhan_wav);
  std::cout << "Dump audio to " << fname_enhan << std::endl;
  return;
}

int main(int argc, const char *argv[]) {
  if (argc != 4) {
    std::cerr
        << "Usage: dcunet-enhan <exported-nnet> <noisy-wav> <enhan-wav>\n";
    return -1;
  }

  const char *dcunet_cpt = argv[1];
  Module dcunet;
  try {
    dcunet = torch::jit::load(dcunet_cpt);
  } catch (const c10::Error &e) {
    std::cerr << "Error in loading the dcunet checkpoint: " << dcunet_cpt
              << std::endl;
    return -1;
  }
  std::cout << "Load model from " << dcunet_cpt << " done" << std::endl;
  nnet_enhan(dcunet, argv[2], argv[3]);
  return 0;
}
