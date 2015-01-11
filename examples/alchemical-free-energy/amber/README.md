There are five subdirectories in `data/` named `ti00[1-5]`, each containing `ti00[1-5].out` (which are copies of the `ti001.out`, with `clambda` edited)
and `ti00[1-5].en` (copies of `ti001.en`).

The output files were obtained by executing the script with the following options:

`python ../alchemical_analysis.py -a AMBER -d data/ -p ti*/ti -q out -u kcal -r 8 -v`

The argument that follows the `-a` flag should not necessarily be in all capitals; any combination of lower- and upper-case letters is OK.
`data` is the path to the directory with the data files.

The `-p` flag seeks for the prefix of the data file. If the data files are in multiple subdirectories,
the name of those subdirectories (in a form of the glob pattern) should preface the file prefix (like `ti*/ti` above).

Whenever the `-v` flag (verbose option) is used, the averages and RMS fluctuations computed for each quantity present in the MDEN file (`ti00X.en` in our example; no requirement for it to bear the same prefix as the MDOUT file) will be computed and displayed.

The dataset contained in the `data/` directory was generated from the files obtained from a 21-window simulation of the vdw gas-phase methanol-to-ethane transformation run by Hannes Loeffler at STFC, UK.
