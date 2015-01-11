Script: `alchemical_analysis.py`

An illustration of MBAR applied to alchemical free energy
calculations, and comparison of MBAR to a number of other free energy
methods described in Paliwal and Shirts, J. Chem. Theory Comp, v. 7,
4115-4134 (2011).

The dataset contained in the data/ directory is obtained from a series
of alchemical intermediates for computing the hydration free energy of
3-methylindole using a beta version of Gromacs 4.6. File output is
consistent with Gromacs 5.0. 3-methylindole is represented by its OPLS
parameters and water by TIP3P.  The paramters are from Shirts and
Pande. J. Chem. Phys. 122, 134508 (2005).

Electrostatic and van der Waals interactions are turned off in the same
simulation, over 38 total states.  Files used to generate the data are
included in the directory inputfiles/3-methylindole-38steps. The
placeholder FEP_STATE is replaced with the integer 0 to 37 to produce
the 38 input files.

To run the files for a sparser lambda spacing (11 total states),
invoke specifying the data directory to use with the command:

`python alchemical_analysis.py -d data/3-methylindole-11steps -q xvg -p dhdl -u kJ`

For the denser lambda spacing with 38 total states, run

`python alchemical_analysis.py -d data/3-methylindole-38steps -q xvg -p dhdl -u kJ`

Note that all these files were generated in gromacs with `calc-lambda-neighbors = -1`, where one calculates 
the energy at all the other states.

One can also run `alchemical_analysis.py` on files that include just the
states that are +1/-1 states from the intermediate simulated at, which
is default for GROMACS. MBAR, however, cannot be run on this
restricted data set.  The difference between these two files is
automatically recognized. For an example, try:

`python alchemical_analysis.py -d data/3-methylindole-11steps-neighbors`

Note how all the output results are the same _except_ for MBAR, which
is omitted from the calculation.

Sample outputs are provided in the files:  
```
output_38steps/screen_printout.txt
output_11steps/screen_printout.txt
output_11steps_neighbors/screen_printout.txt
output_11steps_skip_lambda/screen_printout.txt
```

The last file from the above list corresponds to the analysis that does not account for some intermediate
states. This is controlled by the -k flag. An alternative approach (no need to use the -k flag) is for the
directory with the data files to contain only those that are of interest.
