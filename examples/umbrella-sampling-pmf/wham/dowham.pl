#!/usr/bin/perl -w

$name = "PMF";
$nbin = 144;
$temp = 298;
$kB = 0.008314472;
$kT = $temp*$kB;
$kcal = 4.1868;
$kcaltokT = $kT/$kcal;
$outfile = "$name"."_"."$nbin";
$randn = 12323;
$sys = "../../../pymbar/wham-grossfield/wham/wham P360.0 -180 180 $nbin 0.0000000001 $temp 0 metadata.dat $outfile 100 $randn";
`$sys`;
open(IN,"$outfile");@lines = <IN>;close(IN);

foreach $l (@lines) {
    if ($l =~ /\#/) {
	print $l;
	next;
    }
    ($theta,$pmf,$err,$prob,$perr) = split(/\s+/,$l);
    $pmf /= $kcaltokT;
    $err /= $kcaltokT;
    printf "%8.2f%10.4f%10.4f%15.12g%15.12g\n",$theta,$pmf,$err,$prob,$perr;   
}
