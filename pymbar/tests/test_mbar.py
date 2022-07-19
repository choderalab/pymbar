"""Test MBAR by performing statistical tests on a set of model systems
for which the true free energy differences can be computed analytically.
"""

import numpy as np
import pytest
from pymbar import MBAR
from pymbar.testsystems import harmonic_oscillators, exponential_distributions
from pymbar.utils_for_testing import assert_equal, assert_almost_equal
from pymbar.utils import ParameterError

precision = 8  # the precision for systems that do have analytical results that should be matched.
# Scales the z_scores so that we can reject things that differ at the ones decimal place.  TEMPORARY HACK
z_scale_factor = 12.0
# 0.5 is rounded to 1, so this says they must be within 3.0 sigma
N_k = np.array([1000, 500, 0, 800])


def generate_ho(O_k=np.array([1.0, 2.0, 3.0, 4.0]), K_k=np.array([0.5, 1.0, 1.5, 2.0])):
    return "Harmonic Oscillators", harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, K_k)


def generate_exp(rates=np.array([1.0, 2.0, 3.0, 4.0])):  # Rates, e.g. Lambda
    return "Exponentials", exponential_distributions.ExponentialTestCase(rates)


def convert_to_differences(x_ij, dx_ij, xa):
    xa_ij = xa - np.vstack(xa)

    # add ones to the diagonal of the uncertainties, because they are zero
    for i in range(len(N_k)):
        dx_ij[i, i] += 1
    z = (x_ij - xa_ij) / dx_ij
    for i in range(len(N_k)):
        # these terms should be zero; so we only throw an error if they aren't
        z[i, i] = x_ij[i, i] - xa_ij[i, i]
    return z


system_generators = [generate_ho, generate_exp]
observables = ["position", "position^2", "RMS deviation", "potential energy"]


@pytest.fixture(scope="module", params=system_generators)
def mbar_and_test(request):
    name, test = request.param()
    x_n, u_kn, N_k_output, s_n = test.sample(N_k, mode="u_kn")
    assert_equal(N_k, N_k_output)
    mbar = MBAR(u_kn, N_k, verbose=True, n_bootstraps=200)  # Bootstrap needed for a few tests
    yield_bundle = {"mbar": mbar, "test": test, "x_n": x_n, "u_kn": u_kn}
    yield yield_bundle


@pytest.fixture(scope="module")
def mbar_and_test_harmonic():
    name, test = generate_ho()
    x_n, u_kn, N_k_output, s_n = test.sample(N_k, mode="u_kn")
    assert_equal(N_k, N_k_output)
    mbar = MBAR(u_kn, N_k, verbose=True)
    yield_bundle = {"mbar": mbar, "test": test, "x_n": x_n, "u_kn": u_kn}
    yield yield_bundle


@pytest.fixture(scope="module")
def mbar_and_test_kln():
    name, test = generate_ho()
    x_n, u_kn, N_k_output = test.sample(N_k, mode="u_kln")
    assert_equal(N_k, N_k_output)
    mbar = MBAR(u_kn, N_k, verbose=True)
    yield_bundle = {"mbar": mbar, "test": test, "x_n": x_n, "u_kn": u_kn}
    yield yield_bundle


@pytest.fixture()  # Function  level scope
def fixed_harmonic_sample():
    _, test = generate_ho()
    return test


@pytest.fixture()
def single_harmonic_u_kn(fixed_harmonic_sample):
    _, u_kn, _, _ = fixed_harmonic_sample.sample(N_k, mode="u_kn")
    return u_kn


def free_energies_almost_equal(mbar_fe, err_fe, analytical_fe):
    """Helper to test if MBAR vs Analytical free energies are almost equal"""
    mbar_fe, err_fe = mbar_fe[0, 1:], err_fe[0, 1:]
    analytical_fe = analytical_fe[1:] - analytical_fe[0]

    z = (mbar_fe - analytical_fe) / err_fe
    assert_almost_equal(z / z_scale_factor, np.zeros(len(z)), decimal=0)


def test_ukln(mbar_and_test_kln):
    """Test that MBAR's u_kln->u_kn works correctly"""
    mbar, test = mbar_and_test_kln["mbar"], mbar_and_test_kln["test"]
    results = mbar.compute_free_energy_differences()
    fe = results["Delta_f"]
    fe_sigma = results["dDelta_f"]
    free_energies_almost_equal(fe, fe_sigma, test.analytical_free_energies())


def test_duplicate_state(fixed_harmonic_sample, caplog):
    """Test that MBAR's duplicate state check is working"""
    _, u_kn, _, _ = fixed_harmonic_sample.sample(N_k, mode="u_kn")
    u_kn_dup = np.append(u_kn, u_kn[[0], :], axis=0)
    N_k_dup = np.append(N_k, [0])
    mbar = MBAR(u_kn_dup, N_k_dup, verbose=True)
    assert "likely to to be the same thermodynamic state" in caplog.text
    results = mbar.compute_free_energy_differences()
    fe = results["Delta_f"]
    assert np.allclose(fe[0], fe[-1])


def test_x_kindices(fixed_harmonic_sample):
    """Test that setting x_kindices results in the same calculation"""
    _, u_kn, _, _ = fixed_harmonic_sample.sample(N_k, mode="u_kn")
    flat_x_indices = np.concatenate([[k] * n for k, n in enumerate(N_k)]).astype(int)
    mbar = MBAR(u_kn, N_k)
    mbar_indices = MBAR(u_kn, N_k, x_kindices=flat_x_indices)
    fe = mbar.compute_free_energy_differences()["Delta_f"]
    fes = mbar_indices.compute_free_energy_differences()["Delta_f"]
    assert np.allclose(fe, fes)


@pytest.mark.xfail(raises=ParameterError)
def test_bad_inital_f_k(fixed_harmonic_sample):
    _, u_kn, _, _ = fixed_harmonic_sample.sample(N_k, mode="u_kn")
    MBAR(u_kn, N_k, initial_f_k=[0] * (N_k.size + 1))


def test_covariance_of_sums_runs(mbar_and_test_kln):
    """Test that CovarianceOfSums function runs"""
    # TODO: Is this function still needed? And what would be a better test?
    mbar = mbar_and_test_kln["mbar"]
    results = mbar.compute_free_energy_differences(return_theta=True)
    theta = results["Theta"]
    mbar.compute_covariance_of_sums(theta, 1, np.array([1, -1]))


@pytest.mark.parametrize("system_generator", system_generators)
def test_analytical(system_generator):
    """Generate test objects and calculate analytical results."""
    name, test = system_generator()
    mu = test.analytical_means()
    variance = test.analytical_variances()
    f_k = test.analytical_free_energies()
    for observable in observables:
        A_k = test.analytical_observable(observable=observable)
    s_k = test.analytical_entropies()


@pytest.mark.parametrize("system_generator", system_generators)
def test_sample(system_generator):
    """Draw samples via test object."""

    name, test = system_generator()
    print(name)

    x_n, u_kn, N_k, s_n = test.sample([5, 6, 7, 8], mode="u_kn")
    x_n, u_kn, N_k, s_n = test.sample([5, 5, 5, 5], mode="u_kn")
    x_n, u_kn, N_k, s_n = test.sample([1, 1, 1, 1], mode="u_kn")
    x_n, u_kn, N_k, s_n = test.sample([10, 0, 8, 0], mode="u_kn")

    x_kn, u_kln, N_k = test.sample([5, 6, 7, 8], mode="u_kln")
    x_kn, u_kln, N_k = test.sample([5, 5, 5, 5], mode="u_kln")
    x_kn, u_kln, N_k = test.sample([1, 1, 1, 1], mode="u_kln")
    x_kn, u_kln, N_k = test.sample([10, 0, 8, 0], mode="u_kln")


@pytest.mark.parametrize(
    "uncertainty_method",
    [
        None,
        "approximate",
        "svd",
        "svd-ew",
        "bootstrap",
        pytest.param("waffles", marks=pytest.mark.xfail),
    ],
)
def test_mbar_free_energies(mbar_and_test, uncertainty_method):

    """Can MBAR calculate moderately correct free energy differences?"""
    mbar, test = mbar_and_test["mbar"], mbar_and_test["test"]

    results = mbar.compute_free_energy_differences(
        return_theta=True, uncertainty_method=uncertainty_method
    )
    fe = results["Delta_f"]
    fe_sigma = results["dDelta_f"]
    free_energies_almost_equal(fe, fe_sigma, test.analytical_free_energies())


@pytest.mark.xfail(strict=True)  # This whole test should always fail and passes are problems
@pytest.mark.parametrize("n_bootstrap", [None, -4, 0, 100.3])
def test_mbar_bad_bootstrap(single_harmonic_u_kn, n_bootstrap):
    """Test that bad parameters for n_bootstraps makes bootstrap fail"""
    mbar = MBAR(single_harmonic_u_kn, N_k, verbose=True, n_bootstraps=n_bootstrap)
    mbar.compute_free_energy_differences(uncertainty_method="bootstrap")


@pytest.mark.parametrize(
    "method",
    ["zeros", "mean-reduced-potential", "BAR", pytest.param("waffles", marks=pytest.mark.xfail)],
)
def test_mbar_initialization(fixed_harmonic_sample, method):
    """Do the MBAR initialization methods work?"""
    _, u_kn, _, _ = fixed_harmonic_sample.sample(N_k, mode="u_kn")
    mbar = MBAR(u_kn, N_k, initialize=method, verbose=True)
    results = mbar.compute_free_energy_differences()
    fe = results["Delta_f"]
    fe_sigma = results["dDelta_f"]
    free_energies_almost_equal(fe, fe_sigma, fixed_harmonic_sample.analytical_free_energies())


def test_mbar_compute_expectations_position_averages(mbar_and_test):

    """Can MBAR calculate E(x_n)??"""

    mbar, test, x_n = mbar_and_test["mbar"], mbar_and_test["test"], mbar_and_test["x_n"]
    results = mbar.compute_expectations(x_n)
    mu = results["mu"]
    sigma = results["sigma"]

    mu0 = test.analytical_observable(observable="position")

    z = (mu0 - mu) / sigma
    assert_almost_equal(z / z_scale_factor, np.zeros(len(z)), decimal=0)


def test_mbar_compute_expectations_position_differences(mbar_and_test):

    """Can MBAR calculate E(x_n)??"""
    mbar, test, x_n = mbar_and_test["mbar"], mbar_and_test["test"], mbar_and_test["x_n"]
    results = mbar.compute_expectations(x_n, output="differences")
    mu_ij = results["mu"]
    sigma_ij = results["sigma"]

    mu0 = test.analytical_observable(observable="position")
    z = convert_to_differences(mu_ij, sigma_ij, mu0)
    assert_almost_equal(z / z_scale_factor, np.zeros(np.shape(z)), decimal=0)


def test_mbar_compute_expectations_position2(mbar_and_test):

    """Can MBAR calculate E(x_n^2)??"""

    mbar, test, x_n = mbar_and_test["mbar"], mbar_and_test["test"], mbar_and_test["x_n"]
    results = mbar.compute_expectations(x_n**2)
    mu = results["mu"]
    sigma = results["sigma"]
    mu0 = test.analytical_observable(observable="position^2")

    z = (mu0 - mu) / sigma
    assert_almost_equal(z / z_scale_factor, np.zeros(len(z)), decimal=0)


def test_mbar_compute_expectations_potential(mbar_and_test):

    """Can MBAR calculate E(u_kn)??"""

    mbar, test, u_kn = mbar_and_test["mbar"], mbar_and_test["test"], mbar_and_test["u_kn"]
    results = mbar.compute_expectations(u_kn, state_dependent=True)
    mu = results["mu"]
    sigma = results["sigma"]
    mu0 = test.analytical_observable(observable="potential energy")
    z = (mu0 - mu) / sigma
    assert_almost_equal(z / z_scale_factor, np.zeros(len(z)), decimal=0)


@pytest.mark.parametrize(
    "observable,state_dependent,sample_mode,single_dim,with_uxx",
    [
        ("position", False, "u_kln", False, False),
        ("position", False, "u_kln", False, True),
        ("position", False, "u_kn", False, False),
        ("position", False, "u_kn", False, True),
        ("position", False, "u_kn", True, False),
        ("potential energy", True, "u_kln", False, False),
        ("potential energy", True, "u_kln", False, True),
        ("potential energy", True, "u_kn", False, False),
        ("potential energy", True, "u_kn", False, True),
        ("potential energy", True, "u_kn", True, False),
    ],
)
def test_mbar_compute_expectations_edges(
    mbar_and_test_harmonic,
    mbar_and_test_kln,
    observable,
    state_dependent,
    sample_mode,
    single_dim,
    with_uxx,
):
    if sample_mode == "u_kln":
        payload = mbar_and_test_kln
    else:
        payload = mbar_and_test_harmonic
    mbar = payload["mbar"]
    test = payload["test"]
    u_xxx = payload["u_kn"]
    if state_dependent:
        obs = payload["u_kn"]
    else:
        obs = payload["x_n"]
    if single_dim:
        u_xxx = u_xxx[0]
    results = mbar.compute_expectations(
        obs, state_dependent=state_dependent, u_kn=u_xxx if with_uxx else None, return_theta=True
    )
    mu = results["mu"]
    sigma = results["sigma"]
    mu0 = test.analytical_observable(observable=observable)
    z = (mu0 - mu) / sigma
    assert_almost_equal(z / z_scale_factor, np.zeros(len(z)), decimal=0)


def multiExpectationAssertion(results, test, state=1):
    mu = results["mu"]
    sigma = results["sigma"]
    mu0 = test.analytical_observable(observable="position")[state]
    mu1 = test.analytical_observable(observable="position^2")[state]
    z = (mu0 - mu[0]) / sigma[0]
    assert_almost_equal(z / z_scale_factor, 0 * z, decimal=0)
    z = (mu1 - mu[1]) / sigma[1]
    assert_almost_equal(z / z_scale_factor, 0 * z, decimal=0)


def test_mbar_compute_multiple_expectations(mbar_and_test):

    """Can MBAR calculate E(u_kn)??"""

    mbar, test, x_n, u_kn = (
        mbar_and_test["mbar"],
        mbar_and_test["test"],
        mbar_and_test["x_n"],
        mbar_and_test["u_kn"],
    )
    A = np.zeros([2, len(x_n)])
    A[0, :] = x_n
    A[1, :] = x_n**2
    state = 1
    results = mbar.compute_multiple_expectations(A, u_kn[state, :])
    multiExpectationAssertion(results, test, state=state)


def test_mbar_compute_multiple_expectations_more_dims(mbar_and_test_kln):

    """Can MBAR calculate E(u_kn) with 3 dimensions??"""

    mbar, test, x_n, u_kn = (
        mbar_and_test_kln["mbar"],
        mbar_and_test_kln["test"],
        mbar_and_test_kln["x_n"],
        mbar_and_test_kln["u_kn"],
    )
    A = np.zeros([2, x_n.shape[0], x_n.shape[1]])
    A[0, :, :] = x_n
    A[1, :, :] = x_n**2
    state = 1
    results = mbar.compute_multiple_expectations(
        A, u_kn[:, state, :], compute_covariance=True, return_theta=True
    )
    multiExpectationAssertion(results, test, state=state)


def test_mbar_compute_entropy_and_enthalpy(mbar_and_test, with_uxx=True):

    """Can MBAR calculate f_k, <u_k> and s_k ??"""

    mbar, test, x_n, u_kn = (
        mbar_and_test["mbar"],
        mbar_and_test["test"],
        mbar_and_test["x_n"],
        mbar_and_test["u_kn"],
    )
    results = mbar.compute_entropy_and_enthalpy(u_kn if with_uxx else None, verbose=True)
    f_ij = results["Delta_f"]
    df_ij = results["dDelta_f"]
    u_ij = results["Delta_u"]
    du_ij = results["dDelta_u"]
    s_ij = results["Delta_s"]
    ds_ij = results["dDelta_s"]

    fa = test.analytical_free_energies()
    ua = test.analytical_observable("potential energy")
    sa = test.analytical_entropies()

    fa_ij = fa - fa.T
    ua_ij = ua - ua.T
    sa_ij = sa - sa.T

    z = convert_to_differences(f_ij, df_ij, fa)
    assert_almost_equal(z / z_scale_factor, np.zeros(np.shape(z)), decimal=0)
    z = convert_to_differences(u_ij, du_ij, ua)
    assert_almost_equal(z / z_scale_factor, np.zeros(np.shape(z)), decimal=0)
    z = convert_to_differences(s_ij, ds_ij, sa)
    assert_almost_equal(z / z_scale_factor, np.zeros(np.shape(z)), decimal=0)


@pytest.mark.parametrize(
    "as_kln,with_uxx",
    [
        (True, True),
        (True, False),
        (False, False)
        # False True handled by main test
    ],
)
def test_mbar_compute_entropy_and_enthalpy_edges(
    mbar_and_test_harmonic, mbar_and_test_kln, as_kln, with_uxx
):
    if as_kln:
        payload = mbar_and_test_kln
    else:
        payload = mbar_and_test_harmonic
    test_mbar_compute_entropy_and_enthalpy(payload, with_uxx=with_uxx)


def test_mbar_compute_effective_sample_number(mbar_and_test):
    """testing compute_effective_sample_number"""

    mbar = mbar_and_test["mbar"]
    # one mathematical effective sample numbers should be between N_k and sum_k N_k
    N_eff = mbar.compute_effective_sample_number()
    sumN = np.sum(N_k)
    assert all(N_eff > N_k)
    assert all(N_eff < sumN)


def test_mbar_compute_overlap_analytical():
    """Tests Overlap with identical states, which gives analytical results."""

    d = len(N_k)
    even_O_k = 2.0 * np.ones(d)
    even_K_k = 0.5 * np.ones(d)
    even_N_k = 100 * np.ones(d)
    name, test = generate_ho(O_k=even_O_k, K_k=even_K_k)
    x_n, u_kn, N_k_output, s_n = test.sample(even_N_k, mode="u_kn")
    mbar = MBAR(u_kn, even_N_k)

    results = mbar.compute_overlap()
    overlap_scalar = results["scalar"]
    eigenval = results["eigenvalues"]
    O = results["matrix"]

    reference_matrix = (1.0 / d) * np.ones([d, d])
    reference_eigenvalues = np.zeros(d)
    reference_eigenvalues[0] = 1.0
    reference_scalar = np.float64(1.0)

    assert_almost_equal(O, reference_matrix, decimal=precision)
    assert_almost_equal(eigenval, reference_eigenvalues, decimal=precision)
    assert_almost_equal(overlap_scalar, reference_scalar, decimal=precision)


def test_mbar_compute_overlap_nonanalytical(mbar_and_test):
    """Tests Overlap with stochastic tests"""
    mbar = mbar_and_test["mbar"]
    results = mbar.compute_overlap()
    overlap_scalar = results["scalar"]
    eigenval = results["eigenvalues"]
    O = results["matrix"]

    assert isinstance(overlap_scalar, (float, int))
    # rows of matrix should sum to one
    sumrows = np.array(np.sum(O, axis=1))
    assert_almost_equal(sumrows, np.ones(np.shape(sumrows)), decimal=precision)
    assert_almost_equal(eigenval[0], np.float64(1.0), decimal=precision)


def test_mbar_weights(mbar_and_test):

    """testing weights"""

    mbar = mbar_and_test["mbar"]
    W = mbar.weights()
    sumrows = np.sum(W, axis=0)
    assert_almost_equal(sumrows, np.ones(len(sumrows)), decimal=precision)


@pytest.mark.parametrize(
    "system_generator,mode,bad_n",
    [
        (generate_ho, "u_kn", False),
        (generate_exp, "u_kn", False),
        (generate_ho, "u_kln", False),
        pytest.param(generate_ho, "u_kn", True, marks=pytest.mark.xfail(strict=True)),
    ],
)
def test_mbar_computePerturbedFreeEnergeies(system_generator, mode, bad_n):

    """testing compute_perturbed_free_energies"""

    # only do MBAR with the first and last set

    name, test = system_generator()
    if mode == "u_kln":
        x_n, u_kn, N_k_output = test.sample(N_k, mode=mode)
        numN = max(N_k[:2])
        if bad_n:
            numN = numN - 1
        mslice = np.s_[:2, :2, :numN]
        pslice = np.s_[:2, 2:, :numN]
    else:
        x_n, u_kn, N_k_output, s_n = test.sample(N_k, mode=mode)
        numN = np.sum(N_k[:2])
        if bad_n:
            numN = numN - 1
        mslice = np.s_[:2, :numN]
        pslice = np.s_[2:, :numN]

    mbar = MBAR(u_kn[mslice], N_k[:2])
    results = mbar.compute_perturbed_free_energies(u_kn[pslice])
    fe = results["Delta_f"]
    fe_sigma = results["dDelta_f"]

    fe, fe_sigma = fe[0, 1:], fe_sigma[0, 1:]
    fe0 = test.analytical_free_energies()[2:]
    fe0 = fe0[1:] - fe0[0]

    z = (fe - fe0) / fe_sigma
    assert_almost_equal(z / z_scale_factor, np.zeros(len(z)), decimal=0)


def test_mbar_compute_expectations_inner(mbar_and_test):

    """Can MBAR calculate general expectations inner code (note: this just tests completion)"""

    mbar, test, x_n, u_kn = (
        mbar_and_test["mbar"],
        mbar_and_test["test"],
        mbar_and_test["x_n"],
        mbar_and_test["u_kn"],
    )
    A_in = np.array([x_n, x_n**2, x_n**3])
    u_n = u_kn[:2, :]
    state_map = np.array([[0, 0], [1, 0], [2, 0], [2, 1]], int)
    _ = mbar.compute_expectations_inner(A_in, u_n, state_map)
