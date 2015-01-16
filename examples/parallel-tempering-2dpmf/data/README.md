# Parallel tempering data for alanine dipeptide in explicit solvent.

## Reference

> [1] Chodera JD, Swope WC, Pitera JW, and Dill KA. Long-time protein folding dynamics from short-time molecular dynamics simulations.
> Multiscale Modeling & Simulation, Special Section on Multiscale Modeling in Biology, 5(4):1214-1226, 2006.

## Manifest

* `temperatures` - list of all temperatures (in `K`) used in simulation, from smallest to largest
* `replica-indices` - row `i`, column `j` corresponds to index (from `0..39`) of the replica at temperature index `j` at iteration `i`
* `energies/potential-energies` - row `i`, column `j` corresponds to the potential energy of snapshot `i` from temperature index `j`
* `backbone-torsions/` - trajectories of phi (`*.phi`) and psi (`*.psi`) backbone torsions at a given temperature index
* `end-to-end-distance/` - end-to-end distance trajectory at a given temperature index

## Dataset

The dataset consists of 500 production iterations of a parallel tempering simulation, with 20 ps of Hamiltonian dynamics between exchange attempts, yielding a total simulation time of 10 ns/replica.
There were 40 temperatures, spaning 273-400 K.
Data was collected by temperature, and samples here are for every 1 ps.
See Ref. [1] for more information about the protocol.

