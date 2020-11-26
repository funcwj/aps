#!/usr/bin/env bash

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eu

for x in am lm ss; do python ../bin/train_$x.py -h; done
for x in am ss; do python ../bin/distributed_train_$x.py -h; done
for x in wer ss_metric gmvn; do python ../bin/compute_$x.py -h; done
for x in separate_blind decode decode_batch; do python ../bin/$x.py -h; done

# 5.12% & 2.70%
for cer in true false; do
  ../bin/compute_wer.py --cer $cer data/metric/asr/hyp.text \
    data/metric/asr/ref.text
done

for metric in sdr pesq stoi sisnr; do
  ../bin/compute_ss_metric.py --metric $metric \
    data/metric/sse/bss_spk1.scp,data/metric/sse/bss_spk2.scp \
    data/metric/sse/ref_spk1.scp,data/metric/sse/ref_spk2.scp
done

../bin/compute_gmvn.py --transform asr --sr 16000 \
  data/dataloader/ss/wav.1.scp data/gmvn/transform.yaml data/gmvn/gmvn.pt

../utils/wav_duration.py --output sample data/dataloader/ss/wav.1.scp -
../utils/archive_wav.py data/dataloader/ss/wav.1.scp /dev/null
head data/metric/asr/ref.text | ../utils/tokenizer.pl --space "<space>" -
../utils/tokenizer.py --space "<space>" --unit char --dump-vocab dict \
  --text-format kaldi data/metric/asr/ref.text /dev/null
../utils/tokenizer.py --unit word --dump-vocab dict --add-units "<sos>,<eos>,<unk>" \
  --text-format kaldi data/metric/asr/ref.text /dev/null
../utils/tokenizer.py --spm data/mdl/en.libri.unigram.spm.model --unit subword \
  --text-format kaldi data/metric/asr/ref.text -
