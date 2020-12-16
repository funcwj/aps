#!/usr/bin/env bash

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eu

for x in am lm ss; do python ../cmd/train_$x.py -h; done
for x in am ss; do python ../cmd/distributed_train_$x.py -h; done
for x in wer ss_metric gmvn; do python ../cmd/compute_$x.py -h; done
for x in separate_blind decode decode_batch; do python ../cmd/$x.py -h; done

# 5.12% & 2.70%
for cer in false true; do
  ../cmd/compute_wer.py --cer $cer data/metric/asr/hyp.en.text \
    data/metric/asr/ref.en.text
  ../cmd/compute_wer.py --cer $cer data/metric/asr/hyp.zh.text \
    data/metric/asr/ref.zh.text
done

for metric in sdr pesq stoi sisnr; do
  ../cmd/compute_ss_metric.py --metric $metric \
    data/metric/sse/bss_spk1.scp,data/metric/sse/bss_spk2.scp \
    data/metric/sse/ref_spk1.scp,data/metric/sse/ref_spk2.scp
done

../cmd/compute_gmvn.py --transform asr --sr 16000 \
  data/dataloader/se/wav.1.scp data/transform/transform.yaml /dev/null
../utils/wav_duration.py --output sample data/dataloader/se/wav.1.scp -
../utils/archive_wav.py data/dataloader/se/wav.1.scp /dev/null

head data/metric/asr/ref.en.text | ../utils/tokenizer.pl --space "<space>" -
../utils/tokenizer.py --space "<space>" --unit char --dump-vocab - \
  --text-format kaldi data/metric/asr/ref.en.text /dev/null
../utils/tokenizer.py --unit word --dump-vocab /dev/null --add-units "<sos>,<eos>,<unk>" \
  --text-format kaldi data/metric/asr/ref.zh.text -
../utils/tokenizer.py --spm data/checkpoint/en.libri.unigram.spm.model --unit subword \
  --text-format kaldi data/metric/asr/ref.en.text -
