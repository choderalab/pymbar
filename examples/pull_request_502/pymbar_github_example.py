#===============================================================================
#                                   IMPORTS
#===============================================================================
import pymbar  # multistate Bennett acceptance ratio
from pymbar import timeseries  # timeseries analysis
print(pymbar.__file__)

import numpy as np
import pandas as pd
import scipy

import matplotlib.pyplot as plt
from matplotlib import colors
import os

import numba
#===============================================================================
#                                   IMPORTS
#===============================================================================
@numba.jit(nopython=True)
def harmonic_umbrella_bias(cv_val, cv_details):
    """
    """
    r1, k1 = cv_details
    
    U = 0.5 * k1 * np.power((cv_val - r1), 2)
    
    return U


@numba.jit(nopython=True)
def eval_reduced_pot_energies(N_k, u_kln, u_kn, beta_k, cv_x_kn, restraint_x_k, cv_y_kn, restraint_y_k, b_kln):

    for k in range(K):
        for n in range(N_k[k]):
            for l in range(K):
                # Compute energy of snapshot n from simulation k in umbrella potential l
                u_kln[k, l, n] = u_kn[k, n] + beta_k[k] * (harmonic_umbrella_bias(cv_x_kn[k,n], restraint_x_k[l]) + harmonic_umbrella_bias(cv_y_kn[k,n], restraint_y_k[l]) + b_kln[k,l,n] )
    

def copy_array_numpy_zereos_like_2d(array1, array2, grid=False):
    # both arrays should be same size (x and y have same # points),
    # but output should be just bias
    # grid=False bc needed to call the 2d spline
    return np.zeros_like(array1)
    

class umbrella_set_2d(object):
    
        """
        An Umbrella Set is defined as:
            any number of umbrella windows that share a static bias term
            they may differ in harmonic umbrella terms
        """
    
        def __init__(self, name):
            self.name = name
        
        
        def umbr_harm_xy(self, x_locs, y_locs, k_x, k_y):
            
            if x_locs.ndim != 1:
                raise Exception('x_locs must be 1d arrray!')
            if y_locs.ndim != 1:
                raise Exception('y_locs must be 1d arrray!')
            if len(x_locs) != len(y_locs):
                raise Exception('y_locs and y_locs must be same length!')
            
            self.num_im = len(x_locs)
            self.umbr_harm_x_locs = x_locs
            self.umbr_harm_y_locs = y_locs
            
            # Check both k_x and k_y
            if isinstance(k_x, list) or isinstance(k_x, np.ndarray):
                if len(k_x) != self.num_im:
                    self.umbr_harm_kxs = np.asarray(k_x)
                else:
                    raise Exception('k_x is the wrong length!')
            else:
                self.umbr_harm_kxs = np.full(self.num_im, k_x)
            
            if isinstance(k_y, list) or isinstance(k_y, np.ndarray):
                if len(k_y) != self.num_im:
                    self.umbr_harm_kys = np.asarray(k_y)
                else:
                    raise Exception('k_y is the wrong length!')
            else:
                self.umbr_harm_kys = np.full(self.num_im, k_y)
        
        
        def umbr_static(self, file=None, need_to_invert=False, deg_x=3, deg_y=3, s=0):
            """
            s = spline smoothing parameter (default is 0)
            no extrapolation scheme with 2d spline, so we use the already padded bias file so no need to extrapolate
            also a benefit is this plumed file must be in order and regular grid
            so we can just use unique x and unique y for creating spline
            """
            if file:
                col_names =  ["bin_x", "bin_y", "f", "df_dx", "df_dy"]
                df = pd.read_csv(file, sep='\s+', comment="#", names=col_names)
                df_pivot = df.pivot_table(index="bin_x", columns="bin_y", values="f")
                X = df_pivot.index.values
                Y = df_pivot.columns.values
                Z = df_pivot.T.values
                
                if need_to_invert:
                    # WRONG MAYBE, NOT TESTED
                    self.umbr_static_spline = scipy.interpolate.RectBivariateSpline(X, Y, -Z.T, bbox=[None, None, None, None], kx=deg_x, ky=deg_y, s=s)
                else:
                    self.umbr_static_spline = scipy.interpolate.RectBivariateSpline(X, Y, Z.T, bbox=[None, None, None, None], kx=deg_x, ky=deg_y, s=s)
            else:
                self.umbr_static_spline = copy_array_numpy_zereos_like_2d
        
        
        def sim_temp(self, temp):
            if isinstance(temp, list) or isinstance(temp, np.ndarray):
                if len(temp) != self.num_im:
                    self.temps = np.asarray(temp)
                else:
                    raise Exception('List/Array is the wrong length!')
            else:
                self.temps = np.full(self.num_im, temp)
        
        
        def cv_file(self, file_base):
            self.cv_files = [file_base.format(i=x) for x in range(self.num_im)]
                



#===============================================================================
#                             SETUP FILES & MBAR
#===============================================================================
#------ Constants --------------------------------------------------------------
kB = 1.3806503 * 6.0221415 / 4184.0     # Boltzmann constant in kcal/mol/K

#------ Parameters --------------------------------------------------------------
N_max   = 42000
# PLEASE CHANGE NBINS 
nbins_x     = 25       
nbins_y     = 25                       
temperature = 300.0                     # Desired Output temp for FES
kBT = kB * temperature                  # Inverse temp for the desired ouput temp

#------ File Details --------------------------------------------------------------
base_dir    = "/projectnb/cui-buchem/rosadche/tanmoy_fes/pymbar/combined_dfs/"

pdbfile  = "methyl_phosphate_2min"
sim_base = "nuc_after_pt_2d"
cv_col_x = "P_Olg"               # Which plumed CV to use
cv_col_y = "P_Onuc"               # this is the column second column


data = dict()

# unlike the 1d case, we should pass 2 lists of lenght total simualtions where 
# one is all x values and one is all y values, like output from np.meshgrid and using ravel (check order) if full grid
# if the simualtions are not on a regualr grid, this gives more flexibility of user to pass info
data["run0"] = umbrella_set_2d("fes_inv_run0")
x_locs, y_locs = np.meshgrid(np.linspace(1.8, 4.3, num=7, endpoint=True), np.linspace(3.4, 1.8, num=6, endpoint=True))
x_locs_flat = x_locs.ravel(order="F")
y_locs_flat = y_locs.ravel(order="F")
data["run0"].umbr_harm_xy(x_locs_flat, y_locs_flat, k_x=75.0, k_y=75.0)
data["run0"].sim_temp(temperature)
# should use the padded version used with simualtion so we don't repeat that effort
data["run0"].umbr_static(file="/projectnb/cui-buchem/rosadche/tanmoy_fes/dftb_fes_inv/fes_inv/nuc_after_pt_orig_external.dat", need_to_invert=True, deg_x=3, deg_y=3, s=0)
#data["run0"].umbr_static(file=None, need_to_invert=True, deg_x=3, deg_y=3, s=0)
data["run0"].cv_file(file_base=f"./csvs/{pdbfile}_{sim_base}_fes_inv_run0_image_" + "{i}_combined.csv")

data_keys_lst = list(data.keys())


# ----- Parse Key Data from the dictionary data of Umbrella Sets ---------
K_per_set   = np.asarray( [data[x].num_im for x in data_keys_lst] ) # many harmonic umbrellas are in each set for all sets
K           = np.sum( K_per_set )                                   # K is total number of images. 
K_digitize  = np.digitize(np.arange(0, K), np.cumsum(K_per_set))    # which set does every image belong to

#------ Allocate Storage --------------------------------------------------------------
N_k = np.zeros([K], dtype=int)      # N_k[k] is the number of snapshots from umbrella simulation k
restraint_x_k = np.zeros([K,2])       # Retraint_k[k] is the Umbrella spring constant and center vals for simualtion k: r1, k1
restraint_y_k = np.zeros([K,2])       # Retraint_k[k] is the Umbrella spring constant and center vals for simualtion k: r1, k1
cv_x_kn = np.zeros([K, N_max])    # cv_mat_kn[k,n] is the CV value for snapshot n from umbrella simulation k
cv_x_kn[:] = np.nan
cv_y_kn = np.zeros([K, N_max])  
cv_y_kn[:] = np.nan
u_kn = np.zeros([K, N_max])         # u_kn[k,n] is the reduced potential energy without umbrella restraints of snapshot n of umbrella simulation k (only needed if T is not constant)
u_kln = np.zeros([K, K, N_max])     # u_kln[k,l,n] is the reduced potential energy of snapshot n from umbrella simulation k evaluated at umbrella l
b_kln = np.zeros([K, K, N_max])     # b_kln[k,l,n] is the static bias of snapshot n (from umbrella simulation k) evaluated at umbrella l

T_k     = np.concatenate( [data[x].temps for x in data_keys_lst] )     # pull initial temperatures from umbrella sets
beta_k  = 1.0 / (kB * T_k)
#------ Load Data --------------------------------------------------------------
print("Loading Data...")
cv_min_x = np.inf
cv_max_x = -1.0 * np.inf
cv_min_y = np.inf
cv_max_y = -1.0 * np.inf

for k in range(K):
    
    set_id          = K_digitize[k]
    set_min_loc_in_K_digitize = np.where(K_digitize == set_id)[0][0]
    image_in_set    = k - set_min_loc_in_K_digitize
    which_set = data_keys_lst[set_id]
    print(f"set {which_set} image {image_in_set}")
    
    # not correct
    cv_details_x = np.asarray( [data[which_set].umbr_harm_x_locs[image_in_set], data[which_set].umbr_harm_kxs[image_in_set]] )
    restraint_x_k[k, :]  = cv_details_x
    cv_details_y = np.asarray( [data[which_set].umbr_harm_y_locs[image_in_set], data[which_set].umbr_harm_kys[image_in_set]] )
    restraint_y_k[k, :]  = cv_details_y

    file = data[which_set].cv_files[image_in_set]
    df = pd.read_csv(file)
    
    cv_vals_x = df[cv_col_x].to_numpy()
    cv_vals_y = df[cv_col_y].to_numpy()
    cv_x_kn[k, 0:len(cv_vals_x)] = cv_vals_x
    cv_y_kn[k, 0:len(cv_vals_y)] = cv_vals_y
    
    for i in range(len(data_keys_lst)):
        set_for_bias = data_keys_lst[i]
        locs = np.where(K_digitize == i)[0]
        b_kln[k, locs, 0:len(cv_vals_x)] = df["ext.bias"].to_numpy()
    
    # Compute correlation times for cv_val timeseries
    g_x = timeseries.statistical_inefficiency(cv_vals_x) # compute statistical inefficiency
    g_y = timeseries.statistical_inefficiency(cv_vals_y) # compute statistical inefficiency
    g = max(g_x, g_y)
    g = 5
    indices = timeseries.subsample_correlated_data(cv_vals_x, g=g) # compute indices of uncorrelated timeseries
    #print( f"{k}     | {g:.2f} | {len(indices)}" )
    
    # Subsample data.
    N_k[k] = len(indices)
    u_kn[k, 0 : N_k[k]]       = u_kn[k, indices]
    cv_x_kn[k, 0 : N_k[k]]    = cv_vals_x[indices]
    cv_y_kn[k, 0 : N_k[k]]    = cv_vals_y[indices]
    b_kln[k, : ,  0 : N_k[k]]   = b_kln[k, : , indices].T

    if np.nanmin(cv_x_kn[k, 0 : N_k[k]]) < cv_min_x:
        cv_min_x = np.nanmin(cv_x_kn[k, 0 : N_k[k]])
    
    if np.nanmax(cv_x_kn[k, 0 : N_k[k]]) > cv_max_x:
        cv_max_x = np.nanmax(cv_x_kn[k, 0 : N_k[k]])
        
    if np.nanmin(cv_y_kn[k, 0 : N_k[k]]) < cv_min_y:
        cv_min_y = np.nanmin(cv_y_kn[k, 0 : N_k[k]])
    
    if np.nanmax(cv_y_kn[k, 0 : N_k[k]]) > cv_max_y:
        cv_max_y = np.nanmax(cv_y_kn[k, 0 : N_k[k]])


# ===================================================================================================
# Compute free energy surface at the desired temperature.
# ===================================================================================================
#------ Histogramming -----------------------------            
print("Creating 2D Histogram Bins...")
hist_counts, xedges, yedges = np.histogram2d(cv_x_kn.ravel(), cv_y_kn.ravel(), bins=(nbins_x, nbins_y), range=[[cv_min_x, cv_max_x], [cv_min_y, cv_max_y]], density=False)
centers_x = 0.5 * (xedges[1:] + xedges[:-1])
centers_y = 0.5 * (yedges[1:] + yedges[:-1])

# restart here
bin_edges = []
bin_edges.append( xedges )
bin_edges.append( yedges )

non_zero_indices      = np.nonzero(hist_counts)
bins_nonzero          = len(non_zero_indices[0])
centers_x_nonzero     = centers_x[non_zero_indices[0]]
centers_y_nonzero     = centers_y[non_zero_indices[1]]
centers_nonzero       = np.vstack((centers_x_nonzero, centers_y_nonzero)).T

#"""
print(f"{bins_nonzero:d} bins were populated out of {len(hist_counts.ravel())} (for full rectilinear 2d grid)")
"""
for i in range(bins_nonzero):
    print(f"bin {i:>5} ({centers_nonzero[i][0]:6.1f}, {centers_nonzero[i][1]:6.1f}) {hist_counts[non_zero_indices[0][i], non_zero_indices[1][i]]:12f} conformations")
"""

x_n = np.zeros([np.sum(N_k), 2])  # the configurations

Ntot = 0
for k in range(K):
    for n in range(N_k[k]):
        x_n[Ntot, 0] = cv_x_kn[k, n]
        x_n[Ntot, 1] = cv_y_kn[k, n]
        Ntot += 1


# Compute energy of snapshot n from simulation k in umbrella potential l
#------ Evaluate reduced energies in all umbrellas -----------------------------
print("Evaluating reduced potential energies...")
u_kn -= u_kn.min() # arbitrary up to a constant
eval_reduced_pot_energies(N_k, u_kln, u_kn, beta_k, cv_x_kn, restraint_x_k, cv_y_kn, restraint_y_k, b_kln)

#------ Initialize free energy object with data collected ----------------------
print("Creating FES Object...")
mbar_options = dict()
#mbar_options["solver_protocol"] = "robust"
#mbar_options["relative_tolerance"] = 1.0e-12
mbar_options["verbose"] = True
f_k_str = """  0.           1.15473768   2.12407      1.51820162   2.47140244
   2.67326135  -9.80609257  -8.31462961  -6.14982543  -4.19582768
  -3.49876849  -2.9416488  -15.23423934 -13.67955287 -12.05266483
  -9.38333712  -5.83046512  -4.81443802 -18.16794981 -17.45041341
 -15.4195321  -12.06608162  -7.0951889   -4.73904161 -21.54704098
 -18.55496545 -15.98504957 -13.05527255  -7.3630131   -3.09273673
 -26.65281179 -22.20012933 -18.7905904  -14.54400519  -8.61141264
  -3.63752367 -31.69575533 -25.15863505 -20.53422694 -16.46094288
  -9.73840392  -3.38247384 """
f_k = np.asarray( [float(x) for x in f_k_str.split(" ") if x] )
mbar_options["initial_f_k"] = f_k 
fes = pymbar.FES(u_kln, N_k, mbar_options=mbar_options)

#------ Free Energy Surface -----------------------------    
print("Computing free energy surface...")
# f_i[i] is the dimensionless free energy of bin i (in kT) at the temperature of interest
# df_i[i,j] is an estimate of the covariance in the estimate of (f_i[i] - f_j[j], with reference the lowest free energy state.
# Compute FES in unbiased potential (in units of kT).
histogram_parameters = {}
# just to chek if the right most edge is incosistent, but it seems fine
#bin_edges[-1] += 1e-5
histogram_parameters["bin_edges"] = bin_edges
fes.generate_fes(u_kn, x_n, fes_type="histogram", histogram_parameters=histogram_parameters)
results = fes.get_fes(centers_nonzero, reference_point="from-lowest", uncertainty_method="analytical")
f_i  = kBT * results["f_i"]
df_i = kBT * results["df_i"]

# Show free energy and uncertainty of each occupied bin relative to lowest free energy
text = "# 2D free energy surface in kcal/mol from histogramming" + "\n"
text += f"{'bin':>8s} {cv_col_x:>10s} {cv_col_y:>10s} {'N':>8s} {'f':>10s} {'df':>10s}" + "\n"

for i in range(bins_nonzero):
    text += f"{i:>8d} {centers_nonzero[i][0]:>10.3f} {centers_nonzero[i][1]:>10.3f} {hist_counts[non_zero_indices[0][i], non_zero_indices[1][i]]:>8.0f} {f_i[i]:>10.3f} {df_i[i]:>10.3f}" + "\n"

with open(f"fes_2d_bias_x_{cv_col_x}_y_{cv_col_y}.dat", "w") as f:
    f.write(text)
    
    
    
    
    