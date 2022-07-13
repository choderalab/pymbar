import numpy as np
import pytest
import pymbar
from pymbar.utils_for_testing import (
    suppress_derivative_warnings_for_tests,
    suppress_matrix_warnings_for_tests,
    assert_array_almost_equal,
    oscillators,
    exponentials,
)
from pymbar.tests.test_mbar import z_scale_factor


@pytest.fixture(scope="module")
def base_oscillator():
    name, u_kn, N_k, s_n, test = oscillators(50, 100, provide_test=True)
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

    # Test against old MBAR code.
    with suppress_derivative_warnings_for_tests():
        with suppress_matrix_warnings_for_tests():
            mbar0 = pymbar.old_mbar.MBAR(U, N_k)
    assert_array_almost_equal(mbar.f_k, mbar0.f_k, decimal=8)
    assert_array_almost_equal(np.exp(mbar.Log_W_nk), np.exp(mbar0.Log_W_nk), decimal=5)


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
        "SLSQP",
    ],
)
def test_protocols(base_oscillator, protocol):
    """
    Test that free energy is moderately equal to analytical solution, independent of solver protocols
    """

    # Importing the hacky fix to asert that free energies are moderately correct

    test = base_oscillator["test"]
    u_kn = base_oscillator["u_kn"]
    N_k = base_oscillator["N_k"]
    fa = test.analytical_free_energies()
    fa = fa[1:] - fa[0]
    with suppress_derivative_warnings_for_tests():
        # scipy.optimize.minimize methods, same ones that are checked for in mbar_solvers.py
        # subsampling_protocols = ['adaptive', 'L-BFGS-B', 'dogleg', 'CG', 'BFGS', 'Newton-CG', 'TNC', 'trust-ncg', 'SLSQP']
        # scipy.optimize.root methods. Omitting methods which do not use the Jacobian. Adding the custom adaptive protocol.
        # Solve MBAR with zeros for initial weights
        mbar = pymbar.MBAR(u_kn, N_k, solver_protocol=({"method": protocol},))
        # Solve MBAR with the correct f_k used for the inital weights
        mbar = pymbar.MBAR(
            u_kn, N_k, initial_f_k=mbar.f_k, solver_protocol=({"method": protocol},)
        )
        results = mbar.compute_free_energy_differences()
        fe = results["Delta_f"][0, 1:]
        fe_sigma = results["dDelta_f"][0, 1:]
        z = (fe - fa) / fe_sigma
        assert_array_almost_equal(z / z_scale_factor, np.zeros(len(z)), decimal=0)
