#!/usr/bin/perl -wd
$pi = 3.141592674;
# Grossfield's code uses this conversion 
$kJpkcal = 4.1868; 
@datafiles = <../data/p*dihed.xvg>;

# make new datafiles
@locations = ();
foreach $f (@datafiles) {
    open(IN,"$f");@lines =<IN>;close(IN);
    @v = split(/\//,$f);
    $fl = $v[$#v];
    push(@locations,$fl);
    open(OUT,">data/$fl");
    foreach $l (@lines) {

	if ($l =~ /^@/) {
	    $l = "#".$l;
	}
	if ($l =~ /xtc/) {
	    chomp($l);
	}
	print OUT "$l";
    }
    close(OUT);
}
    
open(IN, "centers.dat");@lines =<IN>;close(IN);
open(OUT,">metadata.dat");
for ($i=0;$i<=$#lines;$i++) {
    $fname = "data/prod$i"."_dihed.xvg";
    ($loc,$spring) = split(/\s+/,$lines[$i]);
    $spring *= (($pi/180)**2)/($kJpkcal);
    print OUT "$fname   $loc $spring\n";
}
close(OUT);

