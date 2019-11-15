#!/usr/bin/env perl
# wujian@2018

use warnings;

if(@ARGV != 1) {
  print STDERR "Invalid usage: " . join(" ", @ARGV) . "\n";
  print STDERR "Usage: token2idx.pl dict <token >index\n";
  exit(1);
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
  for ($x = 1; $x < $#A; $x++) {
    $a = $A[$x];
    if (!defined $dict{$a}) {
      $A[$x] = $dict{"<unk>"};
    } else {
      $A[$x] = $dict{$a};
    }
  }
  print join(" ", @A) . "\n";
}
