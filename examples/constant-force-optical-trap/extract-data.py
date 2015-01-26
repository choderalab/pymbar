# Extract forces and extension timeseries data from Excel spreadsheet files (.xls) provided by Michael T. Woodside
#
# REQUIREMENTS
#
# xlrd (pure Python Excel spreadsheet (.xls) file reader)
# http://www.lexicon.net/sjmachin/xlrd.htm

#=============================================================================================
# IMPORTS
#=============================================================================================

from numpy import * # array routines
import numpy
import commands
import os
import os.path
import sys

try:
    import xlrd # pure Python Excel spreadsheet (.xls) file reader
except ImportError:
    print "Can't run, requires the xlrd .xls file reader module, at http://www.lexicon.net/sjmachin/xlrd.htm"
    sys.exit()

#=============================================================================================
# PARAMETERS
#=============================================================================================

original_data_directory = 'original-data' # location of original data (.xls files)
processed_data_directory = 'processed-data' # location for where processed data is to be stored
datasets = ['20R55_4T'] # prefixes of datasets to process (filenames are '%(prefix)_data.xls')

#=============================================================================================
# SUBROUTINES
#=============================================================================================

#=============================================================================================
# MAIN
#=============================================================================================

# process all datasets
for dataset in datasets:
    print "Extracting data from dataset '%s'..." % dataset

    # Open Excel spreadsheet file.
    workbook_filename = os.path.join(original_data_directory, dataset + '_data.xls')
    workbook = xlrd.open_workbook(workbook_filename)

    # DEBUG
    print "Workbook contains %d worksheets:" % workbook.nsheets
    print workbook.sheet_names()

    # Get the first worksheet.
    worksheet = workbook.sheet_by_index(0)

    # DEBUG
    print "First worksheet (named '%s') has %d columns and %d rows." % (worksheet.name, worksheet.ncols, worksheet.nrows)

    # Extract biasing forces.
    K = worksheet.ncols - 1
    biasing_force_k = zeros([K], float64)
    for k in range(K):
        biasing_force_k[k] = worksheet.cell_value(rowx = 1, colx = 1 + k)
    print "%d biasing forces (in pN):" % K
    print biasing_force_k

    # Write biasing forces.
    filename = os.path.join(processed_data_directory, dataset + '.forces')
    print "Writing biasing forces to '%s'..." % filename
    outfile = open(filename, 'w')
    for k in range(K):
        if (k > 0): outfile.write(' ')
        outfile.write('%.2f' % biasing_force_k[k])
    outfile.close()

    # Read trajectories.
    T_max = worksheet.nrows - 3
    x_kt = zeros([K, T_max])
    for k in range(K):
        print "Reading trajectory %d / %d..." % (k+1, K+1)
        for t in range(T_max):
            x_kt[k,t] = worksheet.cell_value(colx = 1 + k, rowx = 3 + t)
    print "Read %d trajectories of %d samples each." % (K, T_max)
    
    # Write trajectories.
    filename = os.path.join(processed_data_directory, dataset + '.trajectories')
    print "Writing trajectories to '%s'..." % filename
    outfile = open(filename, 'w')
    for t in range(T_max):
        for k in range(K):
            if (k > 0): outfile.write(' ')
            outfile.write('%f' % x_kt[k,t])
        outfile.write('\n')
    outfile.close()


    

