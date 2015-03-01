Script: `alchemical_analysis.py`

An illustration of MBAR applied to alchemical free energy
calculations, and comparison of MBAR to a number of other free energy
methods described in Paliwal and Shirts, J. Chem. Theory Comp, v. 7,
4115-4134 (2011).

See description in `gromacs/README.md`, `sire/README.md`, and `amber/README.md`.


Help for `alchemical_analysis.py` (obtained with `python alchemical_analysis.py -h`) is:

```Options:
  -h, --help            show this help message and exit
  -a SOFTWARE, --software=SOFTWARE
                        Package's name the data files come from: Gromacs,
                        Sire, or AMBER. Default: Gromacs.
  -c, --cfm             The Curve-Fitting-Method-based consistency inspector.
                        Default: False.
  -d DATAFILE_DIRECTORY, --dir=DATAFILE_DIRECTORY
                        Directory in which data files are stored. Default:
                        Current directory.
  -f BFORWREV, --forwrev=BFORWREV
                        Plotting the free energy change as a function of time
                        in both directions. The number of time points (an
                        integer) is to be followed the flag. Default: 0
  -g, --breakdown       Plotting the free energy differences evaluated for
                        each pair of adjacent states for all methods. Default:
                        False.
  -i UNCORR_THRESHOLD, --threshold=UNCORR_THRESHOLD
                        Perform the analysis with rather all the data if the
                        number of uncorrelated samples is found to be less
                        than this number. If 0 is given, the time series
                        analysis will not be performed at all. Default: 50.
  -k BSKIPLAMBDAINDEX, --koff=BSKIPLAMBDAINDEX
                        Give a string of lambda indices separated by '-' and
                        they will be removed from the analysis. (Another
                        approach is to have only the files of interest present
                        in the directory). Default: None.
  -m METHODS, --methods=METHODS
                        A list of the methods to esitimate the free energy
                        with. Default: [TI, TI-CUBIC, DEXP, IEXP, BAR, MBAR].
                        To add/remove methods to the above list provide a
                        string formed of the method strings preceded with +/-.
                        For example, '-ti_cubic+gdel' will turn methods into
                        [TI, DEXP, IEXP, BAR, MBAR, GDEL]. 'ti_cubic+gdel', on
                        the other hand, will call [TI-CUBIC, GDEL]. 'all'
                        calls the full list of supported methods [TI, TI-
                        CUBIC, DEXP, IEXP, GINS, GDEL, BAR, UBAR, RBAR, MBAR].
  -o OUTPUT_DIRECTORY, --out=OUTPUT_DIRECTORY
                        Directory in which the output files produced by this
                        script will be stored. Default: Same as
                        datafile_directory.
  -p PREFIX, --prefix=PREFIX
                        Prefix for datafile sets, i.e.'dhdl' (default).
  -q SUFFIX, --suffix=SUFFIX
                        Suffix for datafile sets, i.e. 'xvg' (default).
  -r DECIMAL, --decimal=DECIMAL
                        The number of decimal places the free energies are to
                        be reported with. No worries, this is for the text
                        output only; the full-precision data will be stored in
                        'results.pickle'. Default: 3.
  -s EQUILTIME, --skiptime=EQUILTIME
                        Discard data prior to this specified time as
                        'equilibration' data. Units picoseconds. Default: 0
                        ps.
  -t TEMPERATURE, --temperature=TEMPERATURE
                        Temperature in K. Default: 298 K.
  -u UNITS, --units=UNITS
                        Units to report energies: 'kJ', 'kcal', and 'kBT'.
                        Default: 'kJ'
  -v, --verbose         Verbose option. Default: False.
  -w, --overlap         Print out and plot the overlap matrix. Default: False.
  -x, --ignoreWL        Do not check whether the WL weights are equilibrated.
                        No log file needed as an accompanying input.
  -y RELATIVE_TOLERANCE, --tolerance=RELATIVE_TOLERANCE
                        Convergence criterion for the energy estimates with
                        BAR and MBAR. Default: 1e-10.
  -z INIT_WITH, --initialize=INIT_WITH
                        The initial MBAR free energy guess; either 'BAR' or
                        'zeroes'. Default: 'BAR'.
```
