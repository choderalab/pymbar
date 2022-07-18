import numpy as np
import pytest
import pymbar
from pymbar.utils_for_testing import (
    assert_almost_equal,
    oscillators,
    exponentials,
)


@pytest.mark.parametrize(
    "statesa, statesb, test_system",
    [(100, 100, oscillators), (200, 50, oscillators), (200, 50, exponentials)],
)
def _test(statesa, statesb, test_system):
    name, U, N_k, s_n = test_system(statesa, statesb)
    print(name)
    mbar = pymbar.MBAR(U, N_k)
    results1 = mbar.compute_free_energy_differences(uncertainty_method="svd")
    results2 = mbar.compute_free_energy_differences(uncertainty_method="svd-ew")
    fij1 = results1["Delta_f"]
    dfij1 = results1["dDelta_f"]
    fij2 = results2["Delta_f"]
    dfij2 = results2["dDelta_f"]

    assert_almost_equal(
        pymbar.mbar_solvers.mbar_gradient(U, N_k, mbar.f_k), np.zeros(N_k.shape), decimal=8
    )
    assert_almost_equal(np.exp(mbar.Log_W_nk).sum(0), np.ones(len(N_k)), decimal=10)
    assert_almost_equal(np.exp(mbar.Log_W_nk).dot(N_k), np.ones(U.shape[1]), decimal=10)
    assert_almost_equal(
        pymbar.mbar_solvers.self_consistent_update(U, N_k, mbar.f_k), mbar.f_k, decimal=10
    )

    assert_almost_equal(fij1, fij2, decimal=8)
    assert_almost_equal(dfij1, dfij2, decimal=8)
