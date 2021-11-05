#!/usr/bin/env bash

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eu

# check import errors
for cmd in cmd/*.py; do python $cmd -h; done

egs_dir=tests/data/metric/asr

# 5.12% & 2.70%
for cer in false true; do
  cmd/compute_wer.py --cer $cer $egs_dir/hyp.en.text $egs_dir/ref.en.text
  cmd/compute_wer.py --cer $cer $egs_dir/hyp.zh.text $egs_dir/ref.zh.text
done

head $egs_dir/ref.en.text | utils/tokenizer.pl --space "<space>" -
for cmd in utils/tokenizer.py cmd/text_tokenize.py; do
  $cmd --space "<space>" --unit char --dump-vocab - \
    --text-format kaldi $egs_dir/ref.en.text /dev/null
  $cmd --unit word --dump-vocab /dev/null --add-units "<sos>,<eos>,<unk>" \
    --text-format kaldi $egs_dir/ref.zh.text -
  $cmd --spm tests/data/checkpoint/en.libri.unigram.spm.model --unit subword \
    --text-format kaldi $egs_dir/ref.en.text -
done

egs_dir=tests/data/metric/sse
for metric in sdr pesq stoi sisnr; do
  cmd/compute_ss_metric.py --metric $metric \
    $egs_dir/bss_spk1.scp,$egs_dir/bss_spk2.scp \
    $egs_dir/ref_spk1.scp,$egs_dir/ref_spk2.scp
done

egs_dir=tests/data/dataloader/se

cmd/compute_gmvn.py --transform asr --sr 16000 --num-jobs 2 \
  $egs_dir/wav.1.scp tests/data/transform/transform.yaml \
  tests/data/transform/gmvn.pt && rm tests/data/transform/gmvn.pt
cmd/check_audio.py $egs_dir/wav.1.scp -
cmd/archive_wav.py $egs_dir/wav.1.scp $egs_dir/egs.1.scp $egs_dir/egs.1.ark
cmd/extract_wav.py $egs_dir/egs.1.scp $egs_dir/egs
rm $egs_dir/egs.1.{scp,ark} && rm -rf $egs_dir/egs

utils/wav_duration.py --output time $egs_dir/wav.1.scp -
