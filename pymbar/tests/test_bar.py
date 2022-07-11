"""Test bar by performing statistical tests on a set of model systems
for which the true free energy differences can be computed analytically.
"""

import numpy as np
import pytest
from pymbar import other_estimators as estimators
from pymbar import MBAR
from pymbar.testsystems import harmonic_oscillators, exponential_distributions
from pymbar.utils_for_testing import assert_equal, assert_almost_equal

precision = 8  # the precision for systems that do have analytical results that should be matched.
# Scales the z_scores so that we can reject things that differ at the ones decimal place.  TEMPORARY HACK
z_scale_factor = 12.0
# 0.5 is rounded to 1, so this says they must be within 3.0 sigma
N_k = np.array([500, 800])


def generate_ho(O_k=np.array([1.0, 2.0]), K_k=np.array([0.5, 2.0])):
    return "Harmonic Oscillators", harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, K_k)


def generate_exp(rates=np.array([1.0, 4.0])):  # Rates, e.g. Lambda
    return "Exponentials", exponential_distributions.ExponentialTestCase(rates)


system_generators = [generate_ho, generate_exp]


@pytest.fixture(scope="module", params=system_generators)
def bar_and_test(request):
    name, test = request.param()
    w_F, w_R, N_k_output = test.sample(N_k, mode="wFwR")
    assert_equal(N_k, N_k_output)
    bars = dict()
    # can't return method, because bar is just a function
    bars["sci"] = estimators.bar(w_F, w_R, method="self-consistent-iteration")
    bars["bis"] = estimators.bar(w_F, w_R, method="bisection")
    bars["fp"] = estimators.bar(w_F, w_R, method="false-position")
    bars["dBAR"] = estimators.bar(w_F, w_R, uncertainty_method="BAR")
    bars["dMBAR"] = estimators.bar(w_F, w_R, uncertainty_method="MBAR")

    yield_bundle = {"bars": bars, "test": test, "w_F": w_F, "w_R": w_R}
    yield yield_bundle


@pytest.mark.parametrize("system_generator", system_generators)
def test_sample(system_generator):
    """Draw samples via test object."""

    name, test = system_generator()
    print(name)

    w_F, w_R, N_k = test.sample([10, 8], mode="wFwR")
    w_F, w_R, N_k = test.sample([1, 1], mode="wFwR")
    w_F, w_R, N_k = test.sample([10, 0], mode="wFwR")
    w_F, w_R, N_k = test.sample([0, 5], mode="wFwR")


def test_bar_free_energies(bar_and_test):

    """Can bar calculate moderately correct free energy differences?"""

    bars, test = bar_and_test["bars"], bar_and_test["test"]

    fe0 = test.analytical_free_energies()
    fe0 = fe0[1:] - fe0[0]

    results_fp = bars["fp"]
    fe_fp = results_fp["Delta_f"]
    dfe_fp = results_fp["dDelta_f"]
    z = (fe_fp - fe0) / dfe_fp
    assert_almost_equal(z / z_scale_factor, np.zeros(len(z)), decimal=0)

    results_sci = bars["sci"]
    fe_sci = results_sci["Delta_f"]
    dfe_sci = results_sci["dDelta_f"]
    z = (fe_sci - fe0) / dfe_sci
    assert_almost_equal(z / z_scale_factor, np.zeros(len(z)), decimal=0)

    results_bis = bars["bis"]
    fe_bis = results_bis["Delta_f"]
    dfe_bis = results_bis["dDelta_f"]
    z = (fe_bis - fe0) / dfe_bis
    assert_almost_equal(z / z_scale_factor, np.zeros(len(z)), decimal=0)

    # make sure the different methods are nearly equal.
    assert_almost_equal(fe_bis, fe_fp, decimal=precision)
    assert_almost_equal(fe_sci, fe_bis, decimal=precision)
    assert_almost_equal(fe_fp, fe_bis, decimal=precision)

    # Test uncertainty methods
    results_dBAR = bars["dBAR"]
    dfe_bar = results_dBAR["dDelta_f"]
    results_dMBAR = bars["dMBAR"]
    dfe_mbar = results_dMBAR["dDelta_f"]

    # not sure exactly how close they need to be for sample problems?
    assert_almost_equal(dfe_bar, dfe_mbar, decimal=3)


def test_bar_overlap():

    for system_generator in system_generators:
        name, test = system_generator()
        x_n, u_kn, N_k_output, s_n = test.sample(N_k, mode="u_kn")
        assert_equal(N_k_output, N_k)

        i = 0
        j = 1
        i_indices = np.arange(0, N_k[0])
        j_indices = np.arange(N_k[0], N_k[0] + N_k[1])
        w_f = u_kn[j, i_indices] - u_kn[i, i_indices]
        w_r = u_kn[i, j_indices] - u_kn[j, j_indices]

        # Compute overlap
        bar_overlap = estimators.bar_overlap(w_f, w_r)

        # Compare with MBAR
        mbar = MBAR(u_kn, N_k)
        mbar_overlap = mbar.compute_overlap()["scalar"]

        assert_almost_equal(bar_overlap, mbar_overlap, decimal=precision)
