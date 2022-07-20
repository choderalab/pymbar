import numpy as np
import pytest
import pymbar
from pymbar.utils_for_testing import (
    assert_array_almost_equal,
    oscillators,
    exponentials,
)
from pymbar.tests.test_mbar import z_scale_factor


@pytest.fixture(scope="module")
def base_oscillator():
    name, u_kn, N_k, s_n, test = oscillators(50, 100, provide_test=True)
    return {"name": name, "u_kn": u_kn, "N_k": N_k, "s_n": s_n, "test": test}


@pytest.fixture(scope="module")
def more_oscillators():
    name, u_kn, N_k, s_n, test = oscillators(50, 500, provide_test=True)
    return {"name": name, "u_kn": u_kn, "N_k": N_k, "s_n": s_n, "test": test}


@pytest.mark.flaky(max_runs=2)  # Uses flaky plugin for pytest
@pytest.mark.parametrize(
    "statesa, statesb, test_system",
    [(100, 100, oscillators), (200, 50, oscillators), (200, 50, exponentials)],
)
def test_solvers(statesa, statesb, test_system):
    name, U, N_k, s_n, _ = test_system(statesa, statesb, provide_test=True)
    print(name)
    mbar = pymbar.MBAR(U, N_k)
    assert_array_almost_equal(
        pymbar.mbar_solvers.mbar_gradient(U, N_k, mbar.f_k), np.zeros(N_k.shape), decimal=8
    )
    assert_array_almost_equal(np.exp(mbar.Log_W_nk).sum(0), np.ones(len(N_k)), decimal=10)
    assert_array_almost_equal(np.exp(mbar.Log_W_nk).dot(N_k), np.ones(U.shape[1]), decimal=10)
    assert_array_almost_equal(
        pymbar.mbar_solvers.self_consistent_update(U, N_k, mbar.f_k), mbar.f_k, decimal=10
    )


def run_mbar_protocol(oscillator_bundle, protocol):
    test = oscillator_bundle["test"]
    u_kn = oscillator_bundle["u_kn"]
    N_k = oscillator_bundle["N_k"]
    fa = test.analytical_free_energies()
    fa = fa[1:] - fa[0]
    # Solve MBAR with zeros for initial weights
    mbar = pymbar.MBAR(u_kn, N_k, solver_protocol=({"method": protocol},))
    # Solve MBAR with the correct f_k used for the initial weights
    mbar = pymbar.MBAR(u_kn, N_k, initial_f_k=mbar.f_k, solver_protocol=({"method": protocol},))
    return mbar, fa


@pytest.mark.parametrize(
    "protocol",
    [
        "adaptive",
        "hybr",
        "lm",
        "L-BFGS-B",
        "dogleg",
        "CG",
        "BFGS",
        "Newton-CG",
        "TNC",
        "trust-ncg",
        "trust-krylov",
        "trust-exact",
        "SLSQP",
    ],
)
def test_protocols(base_oscillator, more_oscillators, protocol):
    """
    Test that free energy is moderately equal to analytical solution, independent of solver protocols
    """

    # Importing the hacky fix to asert that free energies are moderately correct

    try:
        mbar, fa = run_mbar_protocol(base_oscillator, protocol)
    except Exception as e:  # pylint: disable=broad-except
        print(f"Caught error in initial oscillator test, trying with more samples. Error:\n\n{e}")
        mbar, fa = run_mbar_protocol(more_oscillators, protocol)
    results = mbar.compute_free_energy_differences()
    fe = results["Delta_f"][0, 1:]
    fe_sigma = results["dDelta_f"][0, 1:]
    z = (fe - fa) / fe_sigma
    assert_array_almost_equal(z / z_scale_factor, np.zeros(len(z)), decimal=0)
