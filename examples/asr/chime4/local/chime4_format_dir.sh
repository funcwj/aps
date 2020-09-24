#!/usr/bin/env bash

# wujian@2020

set -eu

echo "$0: Formating chime4 data dir..."

chime4_dir=data/chime4
wsj0_dir=data/wsj0
wsj0_chime4_dir=data/wsj0_chime4

for name in tr05 dt05 et05; do
  mkdir -p ./$chime4_dir/$name
  for x in wav.scp text utt2dur; do
    cat $chime4_dir/${name}_{simu,real}_noisy/$x | sort -k1 > $chime4_dir/$name/$x
  done
done

for name in train dev tst; do
  [ -d $chime4_dir/$name ] && rm -rf $chime4_dir/$name
done

mv $chime4_dir/tr05 $chime4_dir/train
mv $chime4_dir/dt05 $chime4_dir/dev
mv $chime4_dir/et05 $chime4_dir/tst

echo -e "<sos> 0\n<eos> 1\n<unk> 2" > $chime4_dir/dict
cat $chime4_dir/train/text | utils/tokenizer.pl --space "<space>" - | \
  cut -d" " -f 2- | tr ' ' '\n' | sort | \
  uniq | awk '{print $1" "NR + 2}' >> $chime4_dir/dict || exit 1

for name in train dev; do
  cat $chime4_dir/$name/text | utils/tokenizer.pl --space "<space>" - | \
    ./utils/token2idx.pl $chime4_dir/dict > $chime4_dir/$name/token || exit 1
done

mkdir -p $wsj0_chime4_dir/{train,dev}
for x in train dev; do
  for name in wav.scp utt2dur token; do
    cat $wsj0_dir/$x/$name $chime4_dir/$x/$name | sort -k1 > $wsj0_chime4_dir/$x/$name
  done
done
cp $chime4_dir/dict $wsj0_chime4_dir

echo "$0: Prepare $chime4_dir & $wsj0_chime4_dir done"
