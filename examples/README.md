pymbar/examples
======

This folder contains two examples illustrating application of MBAR to
a set of harmonic oscillators, for which free energy differences and
expectations can be computed analytically. More examples can be found
in [pymbar-examples](http://github.com/choderalab/pymbar-examples/).

* `README.md` - this file
* `harmonic-oscilllators.py` - a file  
* `harmonic-oscilllators-distributions.py`

It also contains sample output from these scripts.

Usage
------

* `harmonic-oscillators.py` - runs though all of the external functions for MBAR 
  using data generated from harmonic oscillators

** `harmonic-oscillators.py_output.txt` - sample output from `harmonic-oscillators.py`

* `oscillators.pdf` - figure illustrating the overlap of the harmonic oscillators in this test.

* `oscillators.m` - Matlab script to generate oscillators.pdf

This script gives examples of how to call all externally accessible
functionality in MBAR.  Since all samples are drawn from harmoinc
oscillators, 

 * `harmonic-oscillators-distributions.py` - test driver showing the
  consistency of free energies and observable error estimates from the
  normal distribution

** `harmonic-oscillators-distributions.py_output.txt` - sample output from `harmonic-oscillators.py`

** `QQdf.pdf` - QQ plots for the free energy differences

** `QQMBARobserve.pdf` - QQ plots for the ensemble averages computing using MBAR

** `QQstandardobserve.pdf` - QQ plots for the ensemble averages computing using standard averaging (which can't be done for the lone unsampled state

** cumulative_probability_comparison_curves.pdf - another visualization comparing the standard normal distribution with the errors in the data normalized by the estimated uncertatity. 

QQ plots give a straight line if the distributions agree.  In this
case, we compare the distribution of errors from the analytical
estimate divided by the estimated uncertainty to the analytical
standard normal distribution.  In all cases except for distribution
the positions sampled from the unsampled state, the QQ plot is linear
to within noise.

The [Anderson-Darling test](http://en.wikipedia.org/wiki/Anderson%E2%80%93Darling_test) printed out in the
`harmonic-oscillators-distributions.py` code also gives a test of
normality of the error estimates.

Cutoffs for the statistic for the confidence intervals for known uncertainty and known mean are:

* 15%  1.610
* 10%  1.933
* 5%   2.492
* 2.5% 3.070
* 1%   3.857

However, since the sigma is generated using MBAR, then it has some
uncertainty, and the statistic may be slightly different. 

When the number of replicates becomes too high, there is a chance that
the Anderson-Darling metric can be too sensitive, but the current
level of 200 replicates is fine.

Again, all results except the uncertainty of the position in the
unsampled states are consistent with normal distribution of error.