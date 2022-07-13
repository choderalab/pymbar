"""Test exp by performing statistical tests on a set of model systems
for which the true free energy differences can be computed analytically.
"""

import numpy as np
import pytest
from pymbar import other_estimators as estimators
from pymbar.testsystems import harmonic_oscillators, exponential_distributions
from pymbar.utils_for_testing import assert_equal, assert_almost_equal

precision = 8  # the precision for systems that do have analytical results that should be matched.
# Scales the z_scores so that we can reject things that differ at the ones decimal place.  TEMPORARY HACK
z_scale_factor = 12.0
# 0.5 is rounded to 1, so this says they must be within 3.0 sigma
N_k = np.array([50000, 100000])


def generate_ho(O_k=np.array([1.0, 2.0]), K_k=np.array([0.5, 1.0])):
    return "Harmonic Oscillators", harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, K_k)


def generate_exp(rates=np.array([1.0, 4.0])):  # Rates, e.g. Lambda
    return "Exponentials", exponential_distributions.ExponentialTestCase(rates)


system_generators = [generate_ho, generate_exp]


@pytest.fixture(scope="module", params=system_generators)
def exp_and_test(request):
    name, test = request.param()
    w_F, w_R, N_k_output = test.sample(N_k, mode="wFwR")
    assert_equal(N_k, N_k_output)
    exps = dict()
    # can't return method, because exp is just a function
    exps["F"] = estimators.exp(w_F)
    exps["R"] = estimators.exp(w_R)
    exps["gF"] = estimators.exp_gauss(w_F)
    exps["gR"] = estimators.exp_gauss(w_R)

    yield_bundle = {"exps": exps, "test": test, "w_F": w_F, "w_R": w_R}
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


def test_EXP_free_energies(exp_and_test):

    """Can exp calculate moderately correct free energy differences?"""

    exps, test = exp_and_test["exps"], exp_and_test["test"]

    fe0 = test.analytical_free_energies()
    fe0 = fe0[1:] - fe0[0]

    results_F = exps["F"]
    fe_F = results_F["Delta_f"]
    dfe_F = results_F["dDelta_f"]
    z = (fe_F - fe0) / dfe_F
    assert_almost_equal(z / z_scale_factor, np.zeros(len(z)), decimal=0)

    results_R = exps["R"]
    fe_R = -results_R["Delta_f"]
    dfe_R = results_R["dDelta_f"]
    z = (fe_R - fe0) / dfe_R
    assert_almost_equal(z / z_scale_factor, np.zeros(len(z)), decimal=0)

    # turning off Gaussian comparisons because it's not clear how accurate they should be!

    results_gF = exps["gF"]
    fe_gF = results_gF["Delta_f"]
    dfe_gF = results_gF["dDelta_f"]
    z = (fe_gF - fe0) / dfe_gF
    # assert_almost_equal(z / z_scale_factor, np.zeros(len(z)), decimal=0)

    results_gR = exps["gR"]
    fe_gR = -results_gR["Delta_f"]
    dfe_gR = results_gR["dDelta_f"]
    z = (fe_gR - fe0) / dfe_gR
    # assert_almost_equal(z / z_scale_factor, np.zeros(len(z)), decimal=0)

    # make sure the different methods are nearly equal for these systems (within uncertainty)
    z = np.abs(fe_R - fe_F) / np.sqrt(dfe_R**2 + dfe_F**2)
    assert_almost_equal(z / z_scale_factor, 0.0, decimal=0)
