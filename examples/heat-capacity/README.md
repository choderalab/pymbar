USAGE:  

For calling with MBAR error estimates:

python heat-capacity.py > heat-capacity.py.sample_out_no_bootstrap
 
For calling with bootstrap error estimates using 200 bootstraps

python heat-capacity.py -b 200 > heat-capacity.py.sample_out_bootstrap

Other options: 

* -t temperature or -t beta: controls whether temperate or beta
 derivatives are used in calculating finite differences 

* -s spacing controls the number of intermediates used compute finite
   differences (default is 200)

----------------------------------------------------------

This script takes the total energies of a simplified protein model that
are collected at 16 temperatures, and computes the heat capacity at a
finer grain in temperature.  The graph exhibits a strong maximum at
the folding transition.

The heat capacity is computed three ways: 

1) Using the fluctuations using the formula (<E^2>-<E>^2)
/(R*T^2). The uncertainty in the heat capacity is computed from
propagation of the error, though it requires a (not very good)
estimate of the number of effective samples.  CV is proportional to
the variance of the energy, and the uncertainty in the variance is
simply 2*variance divided by the same proportionality constant.

2) We also compute the heat capacity by numerical differentiation of
the heat capacity.  This can be done either with temperature (C_v =
d<E>/dT) or beta (C_V - kB beta^2 d<E>/dbeta) differences.  Which
one is used can be controlled using the -t flag, with -t temperature
being the default.

3) Finally, we compute the heat capacity in terms of the second derivative of
the free energy. This can be done either with temperature or beta
differences.  In the case of temperature differences, we have: C_V =
2*T*df/dT + T^2*d^2f/dT^2, where f is the reduced free energy -kB*ln
Q.  In the case of beta differences, we have: C_V = -k_b beta^2
df^2/d^2beta.  The error can be computed by propagation of error, but
is not currently implemented correctly. Because of complications in
central difference computation of 2nd derivatives at the endpoints, we
omit these points.

We can check the analytical uncertainty estimate by comparing to the
bootstrap error estimate, as seen in the two sample output files.  We
see that the first derivative numerical difference error estimate is
off by no more than about 10%, though the variance-derived heat
capacity error estimate is off by significantly more (usually 50% or
more), and the second derivative variance is completely wrong and
needs to be rederived.

Thanks to Tommy Knotts and Jacob Lewis (BYU) for the data set and the
first draft of the script!

