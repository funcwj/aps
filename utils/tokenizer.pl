#!/usr/bin/env perl

# Copyright 2018 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

# egs
# 1)  add <space> between each word for English dataset
#     cat data/libri/test_clean/text | head | ./local/tokenizer.pl --space "<space>"
# 2)  transform words to characters for Mandarin dataset
#     cat data/aishell_v1/tst/text | head | ./local/tokenizer.pl
# 3)  prepare dictionary
#     cat data/libri/train/text | head | ./local/tokenizer.pl --space "<space>" | \
#     cut -d" " -f 2- | tr ' ' '\n' | sort | uniq | awk '{print $1" "NR + 2}'

use utf8;

use open qw(:encoding(utf8));
binmode STDIN, ":utf8";
binmode STDOUT, ":utf8";
binmode STDERR, ":utf8";

$ended = 0;
$space = "";
$noise = "";

do {
  $ended = 0;
  if ($ARGV[0] eq "--space") {
    $ended = 1;
    $space = $ARGV[1];
    shift @ARGV;
    shift @ARGV;
  }
} while ($ended);

if (@ARGV < 1 || @ARGV > 2) {
  die "Usage: tokenizer.pl [--space <space-token>] text > char\n";
}

while (<>) {
  @F = split " ";
  $cs = $F[0];
  $id = 1;
  foreach $s (@F[1..$#F]) {
    @s = split("", $s);
    foreach $c (@s) {
        $cs = "$cs $c";
    }
    if ($id != $#F && $space ne "") {
      $cs = "$cs $space";
    }
    $id = $id + 1;
  }
  print "$cs\n";
}
