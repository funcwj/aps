// Copyright 2021 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

#include "enh/transformer.h"

namespace aps {
TransformerNnet::TransformerNnet(const TimeFrequencyNnetOptions &opts)
    : stft_(opts.frame_len, opts.frame_hop, opts.window),
      istft_(opts.frame_len, opts.frame_hop, opts.window) {
  // load nnet and feature transform module
  nnet_ = LoadTorchScriptModule(opts.nnet);
  feature_ = LoadTorchScriptModule(opts.transform);
  chunk_size_ = nnet_.attr("chunk").toInt();
  LOG_INFO << "Get chunk size = " << chunk_size_ << " from " << opts.nnet;
  // for nnet forward & stft
  feat_ctx_ = std::make_unique<Context>(Context(0, 0, chunk_size_));
  stft_ctx_ = std::make_unique<Context>(Context(0, 0, chunk_size_));
  Reset();
}

void TransformerNnet::Reset() {
  stft_.Reset();
  feat_ctx_->Reset();
  stft_ctx_->Reset();
  nnet_.run_method("reset");
}

torch::Tensor TransformerNnet::Flush() {
  int32_t padding_samples =
      (chunk_size_ - 1) * istft_.FrameHop() + istft_.FrameLength();
  torch::Tensor zero_pad = torch::zeros(padding_samples), enhan_pad;
  ASSERT(Process(zero_pad, &enhan_pad));
  return enhan_pad;
}

torch::Tensor TransformerNnet::FeatureTransform(const torch::Tensor &stft) {
  // add additional batch dim: F x 2 => 1 x F x 1 x 2
  ASSERT(stft.dim() == 2);
  torch::Tensor feats = feature_.forward({stft.view({1, -1, 1, 2})}).toTensor();
  return feats.squeeze();
}

torch::Tensor TransformerNnet::SpeechEnhancement() {
  ASSERT(!feat_ctx_->Done());
  // call step function, output TF masks
  torch::Tensor feats, masks, enhan;
  std::vector<torch::Tensor> buffer;
  while (!feat_ctx_->Done()) {
    // feats: T x F
    feats = feat_ctx_->Pop();
    // F x T (x2)
    masks = nnet_.run_method("step", feats.unsqueeze(0)).toTensor()[0];
    enhan = Masking(stft_ctx_->Pop(1), masks);
    for (int32_t t = 0; t < enhan.size(1); t++)
      buffer.push_back(
          istft_.Compute(enhan.index({torch::indexing::Slice(), t})));
  }
  return torch::cat(buffer, 0);
}

bool TransformerNnet::Process(const torch::Tensor &audio_chunk,
                              torch::Tensor *audio_enhan) {
  // do STFT
  stft_.Process(audio_chunk);
  while (!stft_.Done()) {
    torch::Tensor raw_stft = stft_.Pop();
    stft_ctx_->Process(raw_stft);
    feat_ctx_->Process(FeatureTransform(raw_stft));
  }
  if (feat_ctx_->Done()) {
    return false;
  } else {
    *audio_enhan = SpeechEnhancement();
    return true;
  }
}

}  // namespace aps
