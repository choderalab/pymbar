[![Build Status](https://travis-ci.org/choderalab/pymbar.png)](https://travis-ci.org/choderalab/pymbar)
[![Anaconda Cloud Downloads Badge](https://anaconda.org/conda-forge/pymbar/badges/downloads.svg)](https://anaconda.org/conda-forge/pymbar)
[![Anaconda Cloud Badge](https://anaconda.org/conda-forge/pymbar/badges/installer/conda.svg)](https://anaconda.org/conda-forge/pymbar)
[![PyPI Version](https://badge.fury.io/py/pymbar.png)](https://pypi.python.org/pypi/pymbar)
[![DOI](https://zenodo.org/badge/9991771.svg)](https://zenodo.org/badge/latestdoi/9991771)

pymbar
======

Python implementation of the [multistate Bennett acceptance ratio (MBAR)](http://www.alchemistry.org/wiki/Multistate_Bennett_Acceptance_Ratio) method for estimating expectations and free energy differences from equilibrium samples from multiple probability densities.
See our [docs](http://pymbar.readthedocs.org/en/latest/).


Installation
------------

The easiest way to install the `pymbar` release is via [conda](http://conda.pydata.org):

```bash
conda install -c conda-forge pymbar
```
which will come with JAX to speed up the code. Or to get the non-JAX accelerated version:
```bash
conda install -c conda-forge pymbar-core
```

You can also install JAX accelerated `pymbar` from the [Python package index](https://pypi.python.org/pypi/pymbar) 
using `pip`:
```bash
pip install pymbar[jax]
```
or the non-jax-accelerated version with
```bash
pip install pymbar
```
Whether you install the JAX accelerated or non-JAX-accelerated version does not 
change any calls or how the code is run. The non-Jax version is smaller on disk due to smaller
dependencies, but may not run as fast.


The development version can be installed directly from github via `pip`:

```bash
# Get the compressed tarball
pip install https://github.com/choderalab/pymbar/archive/master.tar.gz
# Or obtain a temporary clone of the repo with git
pip install git+https://github.com/choderalab/pymbar.git
```

Usage
-----

Basic usage involves importing `pymbar` and constructing an `MBAR` object from the [reduced potential](http://www.alchemistry.org/wiki/Multistate_Bennett_Acceptance_Ratio#Reduced_potential) of simulation or experimental data.

Suppose we sample a 1D harmonic oscillator from a few thermodynamic states:
```python
>>> from pymbar import testsystems
>>> x_n, u_kn, N_k, s_n = testsystems.HarmonicOscillatorsTestCase().sample()
```
We have the `nsamples` sampled oscillator positions `x_n` (with samples from all states concatenated), [reduced potentials](http://www.alchemistry.org/wiki/Multistate_Bennett_Acceptance_Ratio#Reduced_potential) in the `(nstates,nsamples)` matrix `u_kn`, number of samples per state in the `nsamples` array `N_k`, and indices `s_n` denoting which thermodynamic state each sample was drawn from.

To analyze this data, we first initialize the `MBAR` object:

```python
>>> mbar = MBAR(u_kn, N_k)
```

Estimating dimensionless free energy differences between the sampled thermodynamic states and their associated uncertainties (standard errors) simply requires a call to `compute_free_energy_differences()`:

```python
>>> results = mbar.compute_free_energy_differences()
```

Here `results` is a dictionary with keys `Deltaf_ij`, `dDeltaf`, and `Theta`. `Deltaf_ij[i,j]` is the matrix of dimensionless free energy differences `f_j - f_i`, `dDeltaf_ij[i,j]` is the matrix of standard errors in this matrices estimate, and `Theta` is a covariance matrix that can be used to propagate error into quantities derived from the free energies.

Expectations and associated uncertainties can easily be estimated for observables `A(x)` for all states:

```python
>>> A_kn = x_kn # use position of harmonic oscillator as observable
>>> results = mbar.compute_expectations(A_kn)
```

where `results` is a dictionary with keys `mu`, `sigma`, and `Theta`, where `mu[i]` is the array of the estimate for the average of the observable for in state i, `sigma[i]` is the estimated standard deviation of the `mu` estimates,  and `Theta[i,j]` is the covariance matrix of the log weights.

See the docstring help for these individual methods for more information on exact usage; in Python or IPython, you can view the docstrings with `help()`.

JAX needs 64-bit mode
---------------------
PyMBAR needs 64-bit floats to provide reliable answers. JAX by default uses 
[32-bit (Single) bitsize](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision). 
PyMBAR will turn on JAX's 64-bit mode, which may cause issues with some separate uses of JAX in the same code as PyMBAR, 
such as existing Neural Network (NN) Models for machine learning.

If you would like JAX in 32-bit mode, and PyMBAR in the same script, instance your MBAR with the `accelerator=numpy` 
option, e.g.
```python
mbar = MBAR(..., accelerator="numpy")
```
replacing `...` with your other options.

Authors
-------
* Kyle A. Beauchamp <kyle.beauchamp@choderalab.org>
* John D. Chodera <john.chodera@choderalab.org>
* Levi N. Naden <lnaden@vt.edu>
* Michael R. Shirts <michael.shirts@colorado.edu>

References
----------

* Please cite the original MBAR paper:

  > Shirts MR and Chodera JD. Statistically optimal analysis of samples from multiple equilibrium states. J. Chem. Phys. 129:124105 (2008).  [DOI](http://dx.doi.org/10.1063/1.2978177)

* Some timeseries algorithms can be found in the following reference:

  > Chodera JD, Swope WC, Pitera JW, Seok C, and Dill KA. Use of the weighted histogram analysis method for the analysis of simulated and parallel tempering simulations. J. Chem. Theor. Comput. 3(1):26-41 (2007).  [DOI](http://dx.doi.org/10.1021/ct0502864)

* The automatic equilibration detection method provided in `pymbar.timeseries.detectEquilibration()` is described here:

  > Chodera JD. A simple method for automated equilibration detection in molecular simulations. J. Chem. Theor. Comput. 12:1799, 2016.  [DOI](http://dx.doi.org/10.1021/acs.jctc.5b00784)

License
-------

`pymbar` is free software and is licensed under the MIT license.


Thanks
------
We would especially like to thank a large number of people for helping us identify issues
and ways to improve `pymbar`, including Tommy Knotts, David Mobley, Himanshu Paliwal,
Zhiqiang Tan, Patrick Varilly, Todd Gingrich, Aaron Keys, Anna Schneider, Adrian Roitberg,
Nick Schafer, Thomas Speck, Troy van Voorhis, Gupreet Singh, Jason Wagoner, Gabriel Rocklin,
Yannick Spill, Ilya Chorny, Greg Bowman, Vincent Voelz, Peter Kasson, Dave Caplan, Sam Moors,
Carl Rogers, Josua Adelman, Javier Palacios, David Chandler, Andrew Jewett, Stefano Martiniani, and Antonia Mey.

Notes
-----
* `alchemical-analysis.py` described in [this publication](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4420631/) has been [relocated here](https://github.com/MobleyLab/alchemical-analysis).
