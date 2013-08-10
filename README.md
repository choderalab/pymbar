pymbar
======

Python implementation of the multistate Bennett acceptance ratio (MBAR) method for estimating expectations and free energy differences

Authors
-------
* John D. Chodera <choderaj@mskcc.org>
* Michael R. Shirts <michael.shirts@virginia.edu>

Manifest
--------

This archive contains the following files:

* `README.md` - this file
* `GPL` - a copy of the GNU General Public License version 2
* `pymbar/` - Python MBAR package
* `examples/` - examples of applications of MBAR to various types of experiments

Prerequisites
-------------

The pymbar module requires the following:

* Python 2.4 or later: http://www.python.org/
* the NumPy package: http://numpy.scipy.org/
* the SciPy package: http://www.scipy.org/
* Some optional graphing functionality in the tests requires the matplotlib library: http://matplotlib.sourceforge.net/

Many of these packages are now standard in scientific Python installations or bundles, such as [Enthought Canopy](https://www.enthought.com/products/canopy/) or [continuum.io Anaconda](http://continuum.io/).

Installation
------------

Use the provided `setup.py` to install.

To install to your default Python site-packages location:
```tcsh
# python setup.py install
```
Or to install to a different location (e.g. a local Python package repository):
```tcsh
# python setup.py --prefix=/path/to/my/site-packages/
```
The C++ helper code will automatically be built in both cases, if possible.

To build pymbar in situ, without installing to site-packages, run
```tcsh
# python setup.py build
```
and add the directory containing this file to your PYTHONPATH environment variable.
```tcsh
# For tcsh
setenv PYTHONPATH "/path/to/pymbar:$PYTHONPATH"
# For bash
export PYTHONPATH="/path/to/pymbar:$PYTHONPTH"
```


Usage
-----

In Python 2.4 or later, you can view the docstrings with `help()`:
```python
>>> from pymbar import MBAR
>>> help(MBAR)
```
See the example code in the docstrings, or find more elaborate examples in the `examples/` directory.

Basic usage involves first constructing a MBAR object, initializing it with the reduced potential from the simulation or experimental data:
```python
>>> mbar = pymbar.MBAR(u_kln, N_k)
```
Next, we extract the dimensionless free energy differences and uncertainties:
```python
>>> (Deltaf_ij_estimated, dDeltaf_ij_estimated) = mbar.getFreeEnergyDifferences()
```
or compute expectations of given observables A(x) for all states:
```python
>>> (A_k_estimated, dA_k_estimated) = mbar.computeExpectations(A_kn)
```
See the help for these individual methods for more information on exact usage.


References
----------

* Please cite the original MBAR paper:

  Shirts MR and Chodera JD. Statistically optimal analysis of samples from multiple equilibrium states. J. Chem. Phys. 129:124105 (2008).  [DOI](http://dx.doi.org/10.1063/1.2978177)
  
* Some timeseries algorithms can be found in the following reference:

  Chodera JD, Swope WC, Pitera JW, Seok C, and Dill KA. Use of the weighted histogram analysis method for the analysis of simulated and parallel tempering simulations. J. Chem. Theor. Comput. 3(1):26-41 (2007).  [DOI](http://dx.doi.org/10.1021/ct0502864)

  
Copyright notice
----------------

Copyright (c) 2006-2012 The Regents of the University of California.  All Rights Reserved.  Portions of this software are Copyright (c) 2007-2008 Stanford University and Columbia University, (c) 2008-2013 University of Virginia, and (c) 2013 Memorial Sloan-Kettering Cancer Center.

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.

Thanks
------

We would especially like to thank a large number of people for helping us identify issues and ways to improve `pymbar`, including Tommy Knotts, David Mobley, Himanshu Paliwal, Zhiqiang Tan, Patrick Varilly, Todd Gingrich, Aaron Keys, Anna Schneider, Adrian Roitberg, Nick Schafer, Thomas Speck, Troy van Voorhis, Gupreet Singh, Jason Wagoner, Gabriel Rocklin, Yannick Spill, Ilya Chorny, Greg Bowman, Vincent Voelz, Peter Kasson, Sam Moors, Carl Rogers, Josua Adelman, Javier Palacios, David Chandler, and Andrew Jewett.
