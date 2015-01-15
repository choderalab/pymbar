# Example illustrating the application of MBAR to compute a 1D PMF from a series of force-clamp single-molecule experiments.
#
# REFERENCE
#
# Woodside MT, Behnke-Parks WL, Larizadeh K, Travers K, Herschlag D, and Block SM. Nanomechanical measurements of the sequence-dependent folding landscapes of single nucleic acid hairpins. PNAS 103:6190, 2006.

#=============================================================================================
# IMPORTS
#=============================================================================================

from numpy import * # array routines
import numpy
from math import * # additional mathematical functions
import pymbar # multistate Bennett acceptance ratio analysis (provided by pymbar)
from pymbar import timeseries # timeseries analysis (provided by pymbar)
import commands
import os
import os.path
import sys
import time

#=============================================================================================
# PARAMETERS
#=============================================================================================

prefix = '20R55_4T' # for paper
# prefix = '10R50_4T'
# prefix = '25R50_4T'
# prefix = '30R50_4T' 
directory = 'processed-data/'
temperature = 296.15 # temperature (in K)
nbins = 50 # number of bins for 1D PMF
output_directory = 'output/'
plot_directory = 'plots/'

#=============================================================================================
# CONSTANTS
#=============================================================================================

kB = 1.381e-23 # Boltzmann constant (in J/K)
pN_nm_to_kT = (1.0e-9) * (1.0e-12) / (kB * temperature) # conversion from nM pN to units of kT

#=============================================================================================
# SUBROUTINES
#=============================================================================================

def construct_nonuniform_bins(x_n, nbins):
    """Construct histogram using bins of unequal size to ensure approximately equal population in each bin.

    ARGUMENTS
      x_n (1D numpy array of floats) - x_n[n] is data point n

    RETURN VALUES
      bin_left_boundary_i (1D numpy array of floats) - data in bin i will satisfy bin_left_boundary_i[i] <= x < bin_left_boundary_i[i+1]
      bin_center_i (1D numpy array of floats) - bin_center_i[i] is the center of bin i
      bin_width_i (1D numpy array of floats) - bin_width_i[i] is the width of bin i
      bin_n (1D numpy array of int32) - bin_n[n] is the bin index (in range(nbins)) of x_n[n]
    """

    # Determine number of samples.
    N = x_n.size

    # Get indices of elements of x_n sorted in order.
    sorted_indices = x_n.argsort()

    # Allocate storage for results.
    bin_left_boundary_i = zeros([nbins+1], float64) 
    bin_right_boundary_i = zeros([nbins+1], float64)
    bin_center_i = zeros([nbins], float64)
    bin_width_i = zeros([nbins], float64)
    bin_n = zeros([N], int32)
    
    # Determine sampled range, adding a little bit to the rightmost range to ensure no samples escape the range.
    x_min = x_n.min()
    x_max = x_n.max()
    x_max += (x_max - x_min) * 1.0e-5

    # Determine bin boundaries and bin assignments.
    for bin_index in range(nbins):
        # indices of first and last data points in this span
        first_index = int(float(N) / float(nbins) * float(bin_index))
        last_index = int(float(N) / float(nbins) * float(bin_index+1))

        # store left bin boundary
        bin_left_boundary_i[bin_index] = x_n[sorted_indices[first_index]]

        # store assignments
        bin_n[sorted_indices[first_index:last_index]] = bin_index                   

    # set rightmost boundary
    bin_left_boundary_i[nbins] = x_max
    
    # Determine bin centers and widths
    for bin_index in range(nbins):
        bin_center_i[bin_index] = (bin_left_boundary_i[bin_index] + bin_left_boundary_i[bin_index+1]) / 2.0
        bin_width_i[bin_index] = (bin_left_boundary_i[bin_index+1] - bin_left_boundary_i[bin_index])

    # DEBUG
#    outfile = open('states.out', 'w')
#    for n in range(N):
#        outfile.write('%8f %8d\n' % (x_n[n], bin_n[n]))
#    outfile.close()
        
    return (bin_left_boundary_i, bin_center_i, bin_width_i, bin_n)

#=============================================================================================
# MAIN
#=============================================================================================

# read biasing forces for different trajectories
filename = os.path.join(directory, prefix + '.forces')
infile = open(filename, 'r')
elements = infile.readline().split()
K = len(elements) # number of biasing forces
biasing_force_k = zeros([K], float64) # biasing_force_k[k] is the constant external biasing force used to collect trajectory k (in pN)
for k in range(K):
    biasing_force_k[k] = float(elements[k])
infile.close()
print "biasing forces (in pN) = "
print biasing_force_k

# Determine maximum number of snapshots in all trajectories.
filename = os.path.join(directory, prefix + '.trajectories')
T_max = int(commands.getoutput('wc -l %s' % filename).split()[0]) + 1

# Allocate storage for original (correlated) trajectories
T_k = zeros([K], int32) # T_k[k] is the number of snapshots from umbrella simulation k
x_kt = zeros([K,T_max], float64) # x_kt[k,t] is the position of snapshot t from trajectory k (in nm)

# Read the trajectories.
filename = os.path.join(directory, prefix + '.trajectories')
print "Reading %s..." % filename
infile = open(filename, 'r')
lines = infile.readlines()
infile.close()
# Parse data.
for line in lines:
    elements = line.split()
    for k in range(K):
        t = T_k[k]
        x_kt[k,t] = float(elements[k])
        T_k[k] += 1        

# Create a list of indices of all configurations in kt-indexing.
mask_kt = zeros([K,T_max], dtype=bool_)
for k in range(0,K):
    mask_kt[k,0:T_k[k]] = True
# Create a list from this mask.
all_data_indices = where(mask_kt)

# Construct equal-frequency extension bins
print "binning data..."
bin_kt = zeros([K, T_max], int32)
(bin_left_boundary_i, bin_center_i, bin_width_i, bin_assignments) = construct_nonuniform_bins(x_kt[all_data_indices], nbins)
bin_kt[all_data_indices] = bin_assignments

# Compute correlation times.
N_max = 0
g_k = zeros([K], float64)
for k in range(K):
    # Compute statistical inefficiency for extension timeseries
    g = timeseries.statisticalInefficiency(x_kt[k,0:T_k[k]], x_kt[k,0:T_k[k]])
    # store statistical inefficiency
    g_k[k] = g
    print "timeseries %d : g = %.1f, %.0f uncorrelated samples (of %d total samples)" % (k+1, g, floor(T_k[k] / g), T_k[k])
    N_max = max(N_max, ceil(T_k[k] / g) + 1)

# Subsample trajectory position data.
x_kn = zeros([K, N_max], float64)
bin_kn = zeros([K, N_max], int32)
N_k = zeros([K], int32)
for k in range(K):
    # Compute correlation times for potential energy and chi timeseries.
    indices = timeseries.subsampleCorrelatedData(x_kt[k,0:T_k[k]])
    # Store subsampled positions.
    N_k[k] = len(indices)
    x_kn[k,0:N_k[k]] = x_kt[k,indices]
    bin_kn[k,0:N_k[k]] = bin_kt[k,indices]

# Set arbitrary zeros for external biasing potential.
x0_k = zeros([K], float64) # x position corresponding to zero of potential
for k in range(K):
    x0_k[k] = x_kn[k,0:N_k[k]].mean()
print "x0_k = "
print x0_k

# Compute bias energies in units of kT.
u_kln = zeros([K,K,N_max], float64) # u_kln[k,l,n] is the reduced (dimensionless) relative potential energy of snapshot n from umbrella simulation k evaluated at umbrella l
for k in range(K):
    for l in range(K):
        # compute relative energy difference from sampled state to each other state
        # U_k(x) = F_k x
        # where F_k is external biasing force
        # (F_k pN) (x nm) (pN / 
        # u_kln[k,l,0:N_k[k]] = - pN_nm_to_kT * (biasing_force_k[l] - biasing_force_k[k]) * x_kn[k,0:N_k[k]]
        u_kln[k,l,0:N_k[k]] = - pN_nm_to_kT * biasing_force_k[l] * (x_kn[k,0:N_k[k]] - x0_k[l]) + pN_nm_to_kT * biasing_force_k[k] * (x_kn[k,0:N_k[k]] - x0_k[k])

# DEBUG
start_time = time.time()

# Initialize MBAR.
print "Running MBAR..."
mbar = pymbar.MBAR(u_kln, N_k, verbose = True, method = 'adaptive', relative_tolerance = 1.0e-10)

# Compute unbiased energies (all biasing forces are zero).
u_kn = zeros([K,N_max], float64) # u_kn[k,n] is the reduced potential energy without umbrella restraints of snapshot n of umbrella simulation k
for k in range(K):
#    u_kn[k,0:N_k[k]] = - pN_nm_to_kT * (0.0 - biasing_force_k[k]) * x_kn[k,0:N_k[k]]    
    u_kn[k,0:N_k[k]] = 0.0 + pN_nm_to_kT * biasing_force_k[k] * (x_kn[k,0:N_k[k]] - x0_k[k])

# Compute PMF in unbiased potential (in units of kT).
print "Computing PMF..."
(f_i, df_i) = mbar.computePMF(u_kn, bin_kn, nbins)
# compute estimate of PMF including Jacobian term
pmf_i = f_i + numpy.log(bin_width_i)
# Write out unbiased estimate of PMF
print "Unbiased PMF (in units of kT)"
print "%8s %8s %8s %8s %8s" % ('bin', 'f', 'df', 'pmf', 'width')
for i in range(nbins):
    print "%8.3f %8.3f %8.3f %8.3f %8.3f" % (bin_center_i[i], f_i[i], df_i[i], pmf_i[i], bin_width_i[i])

filename = os.path.join(output_directory, 'pmf-unbiased.out')
outfile = open(filename, 'w')
for i in range(nbins):
    outfile.write("%8.3f %8.3f %8.3f\n" % (bin_center_i[i], pmf_i[i], df_i[i]))

outfile.close()

# DEBUG
stop_time = time.time()
elapsed_time = stop_time - start_time
print "analysis took %f seconds" % elapsed_time

# compute observed and expected histograms at each state
for l in range(0,K):
    # compute PMF at state l
    (f_i, df_i) = mbar.computePMF(u_kln[:,l,:], bin_kn, nbins)
    # compute estimate of PMF including Jacobian term
    pmf_i = f_i + numpy.log(bin_width_i)
    # center pmf
    pmf_i -= pmf_i.mean()
    # compute probability distribution
    p_i = numpy.exp(- f_i + f_i.min())
    p_i /= p_i.sum()
    # compute observed histograms, filtering to within [x_min,x_max] range
    N_i_observed = zeros([nbins], float64)
    dN_i_observed = zeros([nbins], float64)    
    for t in range(T_k[l]):
        bin_index = bin_kt[l,t]
        N_i_observed[bin_index] += 1.0
    N = N_i_observed.sum()
    # estimate uncertainties in observed counts
    for bin_index in range(nbins):
        dN_i_observed[bin_index] = sqrt(g_k[l] * N_i_observed[bin_index] * (1.0 - N_i_observed[bin_index] / float(N)))
    # compute expected histograms
    N_i_expected = float(N) * p_i
    dN_i_expected = numpy.sqrt(float(N) * p_i * (1.0 - p_i)) # only approximate, since correlations df_i df_j are neglected
    # plot
    print "state %d (%f pN)" % (l, biasing_force_k[l])
    for bin_index in range(nbins):
        print "%8.3f %10f %10f +- %10f" % (bin_center_i[bin_index], N_i_expected[bin_index], N_i_observed[bin_index], dN_i_observed[bin_index])        

    # Write out observed bin counts
    filename = os.path.join(output_directory, 'counts-observed-%d.out' % l)    
    outfile = open(filename, 'w')
    for i in range(nbins):
        outfile.write("%8.3f %16f %16f\n" % (bin_center_i[i], N_i_observed[i], dN_i_observed[i]))
        
    outfile.close()    
    # write out expected bin counts
    filename = os.path.join(output_directory, 'counts-expected-%d.out' % l)
    outfile = open(filename, 'w')
    for i in range(nbins):
        outfile.write("%8.3f %16f %16f\n" % (bin_center_i[i], N_i_expected[i], dN_i_expected[i]))
        
    outfile.close()    

    # compute PMF from observed counts
    indices = where(N_i_observed > 0)[0]
    pmf_i_observed = zeros([nbins], float64)
    dpmf_i_observed = zeros([nbins], float64)
    pmf_i_observed[indices] = - numpy.log(N_i_observed[indices]) + numpy.log(bin_width_i[indices])
    pmf_i_observed[indices] -= pmf_i_observed[indices].mean() # shift observed PMF
    dpmf_i_observed[indices] = dN_i_observed[indices] / N_i_observed[indices]
    # write out observed PMF
    filename = os.path.join(output_directory, 'pmf-observed-%d.out' % l)    
    outfile = open(filename, 'w')
    for i in indices:
        outfile.write("%8.3f %8.3f %8.3f\n" % (bin_center_i[i], pmf_i_observed[i], dpmf_i_observed[i]))
        
    outfile.close()    

    # Write out unbiased estimate of PMF
    pmf_i -= pmf_i[indices].mean() # shift to align with observed
    filename = os.path.join(output_directory, 'pmf-expected-%d.out' % l)
    outfile = open(filename, 'w')
    for i in range(nbins):
        outfile.write("%8.3f %8.3f %8.3f\n" % (bin_center_i[i], pmf_i[i], df_i[i]))
        
    outfile.close()    

    # make gnuplot plots
    biasing_force = biasing_force_k[l]
    filename = os.path.join(plot_directory, 'pmf-comparison-%d.eps' % l)
    gnuplot_input = """
set term postscript color solid
set output "%(filename)s"
set title "%(prefix)s - %(biasing_force).2f pN"
set xlabel "extension (nm)"
set ylabel "potential of mean force (kT)"
plot "%(output_directory)s/pmf-expected-%(l)d.out" u 1:2:3 with yerrorbars t "MBAR optimal estimate", "%(output_directory)s/pmf-observed-%(l)d.out" u 1:2:3 with yerrorbars t "observed from single experiment"
""" % vars()
    gnuplot_input_filename = os.path.join(plot_directory, 'gnuplot.in')
    outfile = open(gnuplot_input_filename, 'w')
    outfile.write(gnuplot_input)
    outfile.close()
    output = commands.getoutput('gnuplot < %(gnuplot_input_filename)s' % vars())
    output = commands.getoutput('epstopdf %(filename)s' % vars())    
    


