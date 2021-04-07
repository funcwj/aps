#!/usr/bin/env bash

# Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.

set -eu

if [ $# -le 3 ]; then
  echo "Arguments should be a list of WSJ directories, see ../run.sh for example."
  exit 1;
fi


dir=$PWD/data/local/data
mkdir -p $dir
local=$PWD/local
sph2pipe=sph2pipe

if [ ! `which sph2pipe` ]; then
  echo "Could not find sph2pipe, install it first..."
  mkdir -p exp && cd exp && wget https://www.openslr.org/resources/3/sph2pipe_v2.5.tar.gz
  tar -zxf sph2pipe_v2.5.tar.gz && cd sph2pipe_v2.5
  gcc -o sph2pipe *.c -lm && cd .. && rm -rf sph2pipe_v2.5.tar.gz
  sph2pipe=$PWD/sph2pipe_v2.5/sph2pipe
  cd ..
fi

cd $dir
# Make directory of links to the WSJ disks such as 11-13.1.  This relies on the command
# line arguments being absolute pathnames.
rm -rf links/ 2>/dev/null
mkdir links/

ln -s $* links
# Do some basic checks that we have what we expected.
if [ ! -d links/11-13.1 -o ! -d links/13-34.1 -o ! -d links/11-2.1 ]; then
  echo "wsj_data_prep.sh: Spot check of command line arguments failed"
  echo "Command line arguments must be absolute pathnames to WSJ directories"
  echo "with names like 11-13.1."
  echo "Note: if you have old-style WSJ distribution,"
  echo "local/cstr_wsj_data_prep.sh may work instead, see run.sh for example."
  exit 1;
fi

# This version for SI-284
cat links/13-34.1/wsj1/doc/indices/si_tr_s.ndx \
 links/11-13.1/wsj0/doc/indices/train/tr_s_wv1.ndx | \
 $local/ndx2flist.pl  $* | sort | \
 grep -v -i 11-2.1/wsj0/si_tr_s/401 > train_si284.flist

nl=`cat train_si284.flist | wc -l`
[ "$nl" -eq 37416 ] || echo "Warning: expected 37416 lines in train_si284.flist, got $nl"

# Nov'92 (333 utts)
# These index files have a slightly different format;
# have to add .wv1
cat links/11-13.1/wsj0/doc/indices/test/nvp/si_et_20.ndx | \
  $local/ndx2flist.pl $* |  awk '{printf("%s.wv1\n", $1)}' | \
  sort > test_eval92.flist

# Dev-set for Nov'93 (503 utts)
cat links/13-34.1/wsj1/doc/indices/h1_p0.ndx | \
  $local/ndx2flist.pl $* | sort > test_dev93.flist

# Finding the transcript files:
for x in $*; do find -L $x -iname '*.dot'; done > dot_files.flist

# Convert the transcripts into our format (no normalization yet)
for x in train_si284 test_eval92 test_dev93; do
   $local/flist2scp.pl $x.flist | sort > ${x}_sph.scp
   cat ${x}_sph.scp | awk '{print $1}' | $local/find_transcripts.pl  dot_files.flist > $x.trans1
done

# Do some basic normalization steps.  At this point we don't remove OOVs--
# that will be done inside the training scripts, as we'd like to make the
# data-preparation stage independent of the specific lexicon used.
noiseword="<NOISE>";
for x in train_si284 test_eval92 test_dev93; do
   cat $x.trans1 | $local/normalize_transcript.pl $noiseword | sort > $x.txt || exit 1;
done

# Create scp's with wav's. (the wv1 in the distribution is not really wav, it is sph.)
for x in train_si284 test_eval92 test_dev93; do
  awk -v cmd=$sph2pipe'{printf("%s %s -f wav %s |\n", $1, cmd, $2);}' < ${x}_sph.scp > ${x}_wav.scp
done

echo "Data preparation succeeded"
