"""
Extract forces and extension timeseries data from Excel spreadsheet files (.xls) provided by Michael T. Woodside

REQUIREMENTS:

* xlrd (pure Python Excel spreadsheet (.xls) file reader) - http://www.lexicon.net/sjmachin/xlrd.htm
"""

# =============================================================================================
# IMPORTS
# =============================================================================================

import sys
import bz2
import tempfile
from pathlib import Path

from numpy import zeros, float64

try:
    import xlrd  # pure Python Excel spreadsheet (.xls) file reader
except ImportError:
    print("ERROR: This example requires xlrd, run:", "  pip install xlrd", sep="\n")
    sys.exit()

# =============================================================================================
# PARAMETERS
# =============================================================================================
# location of original data (.xls files)
original_data_directory = Path("original-data")
# location for where processed data is to be stored
processed_data_directory = Path("processed-data")
# prefixes of datasets to process (filenames are f'({prefix}_data.xls')
datasets = ["20R55_4T"]

# =============================================================================================
# SUBROUTINES
# =============================================================================================

# =============================================================================================
# MAIN
# =============================================================================================


def main():
    ORIGINAL_DATA = Path(original_data_directory)
    PROCESSED_DATA = Path(processed_data_directory)
    # process all datasets
    for dataset in datasets:
        print(f"Extracting data from dataset '{dataset}'...")

        # Extract compressed Excel spreadsheet file and load it with xlrd
        with bz2.open(ORIGINAL_DATA / f"{dataset}_data.xls.bz2") as f:
            with tempfile.NamedTemporaryFile(suffix=".xls", delete=False) as temp:
                temp.write(f.read())
        workbook = xlrd.open_workbook(temp.name)

        # DEBUG
        print(
            f"Workbook contains {workbook.nsheets:d} worksheets:", workbook.sheet_names(), sep="\n"
        )

        # Get the first worksheet.
        worksheet = workbook.sheet_by_index(0)

        # DEBUG
        print(
            f"1st worksheet '{worksheet.name}' has {worksheet.ncols:d} columns and {worksheet.nrows:d} rows."
        )

        # Extract biasing forces.
        K = worksheet.ncols - 1
        biasing_force_k = zeros([K], float64)
        for k in range(K):
            biasing_force_k[k] = worksheet.cell_value(rowx=1, colx=1 + k)
        print(f"{K:d} biasing forces (in pN):", biasing_force_k, sep="\n")

        # Write biasing forces.
        filename = PROCESSED_DATA / f"{dataset}.forces"
        print(f"Writing biasing forces to '{filename}'...")
        with open(filename, "w") as outfile:
            for k in range(K):
                if k > 0:
                    outfile.write(" ")
                outfile.write(f"{biasing_force_k[k]:.2f}")

        # Read trajectories.
        T_max = worksheet.nrows - 3
        x_kt = zeros([K, T_max])
        for k in range(K):
            print(f"Reading trajectory {k + 1:d} / {K + 1:d}...")
            for t in range(T_max):
                x_kt[k, t] = worksheet.cell_value(colx=1 + k, rowx=3 + t)
        print(f"Read {K:d} trajectories of {T_max:d} samples each.")

        # Write trajectories.
        filename = PROCESSED_DATA / f"{dataset}.trajectories"
        print(f"Writing trajectories to '{filename}'...")
        with open(filename, "w") as outfile:
            for t in range(T_max):
                for k in range(K):
                    if k > 0:
                        outfile.write(" ")
                    outfile.write(f"{x_kt[k, t]:f}")
                outfile.write("\n")


if __name__ == "__main__":
    main()
