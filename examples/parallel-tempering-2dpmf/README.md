# Estimate 2D potential of mean force for alanine dipeptide parallel tempering data using MBAR.

This example demonstrates how MBAR [1] can be used to estimate a 2D potential of mean force from parallel tempering simulation data [2].
The system of interest is terminally-blocked alanine peptide in explicit solvent (described in [2]), and the 2D PMF is computed for a simple binning of phi and psi torsion angles.

## Protocol

* Potential energies and (phi, psi) torsions from parallel tempering simulation are read in by temperature
* Replica trajectories of potential energies and torsions are reconstructed to reflect their true temporal correlation, and then subsampled to produce statistically independent samples, collecting them again by temperature
* The `pymbar` class is initialized to compute the dimensionless free energies at each temperature using MBAR
* The torsions are binned into sequentially labeled bins in two dimensions
* The relative free energies and uncertainties of these torsion bins at the temperature of interest is estimated
* The 2D PMF is written out

## References

>  [1] Shirts MR and Chodera JD. Statistically optimal analysis of samples from multiple equilibrium states.
>  J. Chem. Phys. 129:124105, 2008
>  http://dx.doi.org/10.1063/1.2978177
>
>  [2] Chodera JD, Swope WC, Pitera JW, and Dill KA. Long-time protein folding dynamics from short-time molecular dynamics simulations.
>  Multiscale Modeling & Simulation, Special Section on Multiscale Modeling in Biology, 5(4):1214-1226, 2006.
>  http://dx.doi.org/10.1137/06065146X

## Manifest

* `parallel-tempering-2dpmf.py` - Python script to execute this data analysis example
* `data/` - data directory (see `data/README.md` for documentation of data files)
