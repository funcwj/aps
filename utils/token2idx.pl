#!/usr/bin/env perl

# wujian@2018

use warnings;

if (@ARGV != 1) {
  die "Invalid usage: " . join(" ", @ARGV) . "\n".
      "Usage: token2idx.pl dict < text.char > token \n";
}

($dict_obj) = @ARGV;
open(M, "<$dict_obj") || die "Error opening dictionary file $dict_obj: $!";

while (<M>) {
  @A = split(" ", $_);
  @A >= 1 || die "token2idx.pl: empty item.";
  $i = shift @A;
  $o = join(" ", @A);
  $dict{$i} = $o;
}

while(<STDIN>) {
  @A = split(" ", $_);
  for ($x = 1; $x <= $#A; $x++) {
    $a = $A[$x];
    if (!defined $dict{$a}) {
      if (!defined $dict{"<unk>"}) {
        die "token2idx.pl: missing <unk> in dictionary $dict_obj";
      }
      $A[$x] = $dict{"<unk>"};
    } else {
      $A[$x] = $dict{$a};
    }
  }
  print join(" ", @A) . "\n";
}