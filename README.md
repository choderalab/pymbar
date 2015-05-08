[![Build Status](https://travis-ci.org/choderalab/pymbar.png)](https://travis-ci.org/choderalab/pymbar)
[![PyPI Version](https://badge.fury.io/py/pymbar.png)](https://pypi.python.org/pypi/pymbar)
[![Downloads](https://pypip.in/d/pymbar/badge.png)](https://pypi.python.org/pypi/pymbar)
[![Binstar Badge](https://binstar.org/omnia/pymbar/badges/installer/conda.svg)](https://conda.binstar.org/omnia)

pymbar
======

Python implementation of the multistate Bennett acceptance ratio (MBAR) method for estimating expectations and free energy differences.  See our [Docs](http://pymbar.readthedocs.org/en/latest/).


Installation
------------

The easiest way to install `pymbar` is via [conda](http://conda.pydata.org), a binary package installer that comes with the [Anaconda Scientific Python distribution](https://store.continuum.io/cshop/anaconda/):
```tcsh
conda install -c https://conda.binstar.org/omnia pymbar
```

Besides conda, you can also install `pymbar` using `pip` (`pip install pymbar`) or directly from the source directory (`python setup.py install`).


Usage
-----

Basic usage involves importing pymbar, loading data, and constructing an MBAR object from the reduced potential of simulation or experimental data:

```python
>>> from pymbar import MBAR, testsystems
>>> x_k, u_kn, N_k, s_n = testsystems.HarmonicOscillatorsTestCase().sample()
>>> mbar = MBAR(u_kn, N_k)
```

Next, we extract the dimensionless free energy differences and uncertainties:

```python
>>> (Deltaf_ij_estimated, dDeltaf_ij_estimated, Theta_ij) = mbar.getFreeEnergyDifferences()
```

or compute expectations of given observables A(x) for all states:

```python
>>> (A_k_estimated, dA_k_estimated) = mbar.computeExpectations(x_k)
```
See the help for these individual methods for more information on exact usage; in Python or IPython, you can view the docstrings with `help()`.  
Additional examples can be found in [pymbar-examples](http://github.com/choderalab/pymbar-examples/).  Note that the example for free energy calculations found in pymbar-examples/alchemical-free-energy has moved to https://github.com/MobleyLab/alchemical-analysis


Prerequisites
-------------

To install and run pymbar requires the following:

* Python 2.7 or later: http://www.python.org/
* NumPy: http://numpy.scipy.org/
* SciPy: http://www.scipy.org/
* NumExpr: https://github.com/pydata/numexpr
* six
* nose
* Some optional graphing functionality in the tests requires the matplotlib library: http://matplotlib.sourceforge.net/


Authors
-------
* John D. Chodera <choderaj@mskcc.org>
* Michael R. Shirts <michael.shirts@virginia.edu>
* Kyle A. Beauchamp <beauchak@mskcc.org>


Manifest
--------

This archive contains the following files:

* `README.md` - this file
* `LICENSE` - a copy of the GNU General Public License version 2 covering this code
* `pymbar/` - Python MBAR package
* `examples/` - examples of applications of MBAR to various types of calculations
  See the README.md in that folder for more information
* `docs/` - sphinx documetation
* `devtools/` - travis CI and conda configuration files



References
----------

* Please cite the original MBAR paper:

  Shirts MR and Chodera JD. Statistically optimal analysis of samples from multiple equilibrium states. J. Chem. Phys. 129:124105 (2008).  [DOI](http://dx.doi.org/10.1063/1.2978177)

* Some timeseries algorithms can be found in the following reference:

  Chodera JD, Swope WC, Pitera JW, Seok C, and Dill KA. Use of the weighted histogram analysis method for the analysis of simulated and parallel tempering simulations. J. Chem. Theor. Comput. 3(1):26-41 (2007).  [DOI](http://dx.doi.org/10.1021/ct0502864)


License
-------

Pymbar is free software and is licensed under the GPLv2 license.


Thanks
------

We would especially like to thank a large number of people for helping us identify issues
and ways to improve `pymbar`, including Tommy Knotts, David Mobley, Himanshu Paliwal,
Zhiqiang Tan, Patrick Varilly, Todd Gingrich, Aaron Keys, Anna Schneider, Adrian Roitberg,
Nick Schafer, Thomas Speck, Troy van Voorhis, Gupreet Singh, Jason Wagoner, Gabriel Rocklin,
Yannick Spill, Ilya Chorny, Greg Bowman, Vincent Voelz, Peter Kasson, Dave Caplan, Sam Moors,
Carl Rogers, Josua Adelman, Javier Palacios, David Chandler, Andrew Jewett, Stefano Martiniani, and Antonia Mey.
