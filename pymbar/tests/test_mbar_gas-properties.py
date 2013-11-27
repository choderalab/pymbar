"""Test MBAR by performing statistical tests on a set of of 1D harmonic oscillators.
for which the true free energy differences can be computed analytically.
"""

import numpy as np
from pymbar import MBAR
import pymbar.mbar
from pymbar.testsystems import harmonic_oscillators
from pymbar.utils import ensure_type, convert_ukn_to_uijn, convert_An_to_Akn, get_data_filename
from pymbar.utils_for_testing import eq
from pymbar.old.mbar import MBAR as MBAR1  # Import mbar 1.0 for some reference calculations


def test_MBAR2_vs_MBAR1():
    """Gas Properties Test: Compare free energies of pymbar2.0 vs pymbar1.0"""

    u_kln_filename = get_data_filename("testsystems/datasets/gas-properties/u_kln.npz")
    N_k_filename = get_data_filename("testsystems/datasets/gas-properties/N_k.dat")
    
    u_kln = np.load(u_kln_filename)["arr_0"]
    N_k = np.loadtxt(N_k_filename)

    mbar = MBAR(u_kln, N_k)
    
    mbar1 = MBAR1(u_kln, N_k)
       
    eq(mbar.f_k, mbar1.f_k)
