from pymbar import timeseries
from pymbar import testsystems
import numpy as np
from scipy import stats
import pytest

from pymbar.utils_for_testing import assert_almost_equal

try:
    import statsmodels.api as sm  # pylint: disable=unused-import

    HAVE_STATSMODELS = True
except ImportError as err:
    HAVE_STATSMODELS = False

has_statmodels = pytest.mark.skipif(
    not HAVE_STATSMODELS, reason="Skipping FFT based tests because statsmodels not installed."
)


@pytest.fixture(scope="module")
def data(N=10000, K=10):
    var = np.ones(N)

    for replica in range(2, K + 1):
        var = np.concatenate((var, np.ones(N)))

    X = np.random.normal(np.zeros(K * N), var).reshape((K, N)) / 10.0
    Y = np.random.normal(np.zeros(K * N), var).reshape((K, N))

    energy = 10 * (X**2) / 2.0 + (Y**2) / 2.0

    return X, Y, energy


def test_statistical_inefficiency_single(data):
    X, Y, energy = data
    timeseries.statistical_inefficiency(X[0])
    timeseries.statistical_inefficiency(X[0], X[0])
    timeseries.statistical_inefficiency(X[0] ** 2)
    timeseries.statistical_inefficiency(X[0] ** 2, X[0] ** 2)
    timeseries.statistical_inefficiency(energy[0])
    timeseries.statistical_inefficiency(energy[0], energy[0])

    timeseries.statistical_inefficiency(X[0], X[0] ** 2)

    # TODO: Add some checks to test statistical inefficinecies are within normal range


def test_statistical_inefficiency_multiple(data):
    X, Y, energy = data
    timeseries.statistical_inefficiency_multiple(X)
    timeseries.statistical_inefficiency_multiple(X**2)
    timeseries.statistical_inefficiency_multiple(X[0, :] ** 2)
    timeseries.statistical_inefficiency_multiple(X[0:2, :] ** 2)
    timeseries.statistical_inefficiency_multiple(energy)
    # TODO: Add some checks to test statistical inefficinecies are within normal range


@has_statmodels
def test_statistical_inefficiency_fft(data):
    X, Y, energy = data
    timeseries.statistical_inefficiency_fft(X[0])
    timeseries.statistical_inefficiency_fft(X[0] ** 2)
    timeseries.statistical_inefficiency_fft(energy[0])

    g0 = timeseries.statistical_inefficiency_fft(X[0])
    g1 = timeseries.statistical_inefficiency(X[0])
    g2 = timeseries.statistical_inefficiency(X[0], X[0])
    g3 = timeseries.statistical_inefficiency(X[0], fft=True)
    assert_almost_equal(g0, g1, decimal=6)
    assert_almost_equal(g0, g2, decimal=6)
    assert_almost_equal(g0, g3, decimal=6)


@has_statmodels
def test_statistical_inefficiency_fft_gaussian():

    # Run multiple times to get things with and without negative "spikes" at C(1)
    for i in range(5):
        x = np.random.normal(size=100000)
        g0 = timeseries.statistical_inefficiency(x, fast=False)
        g1 = timeseries.statistical_inefficiency(x, x, fast=False)
        g2 = timeseries.statistical_inefficiency_fft(x)
        g3 = timeseries.statistical_inefficiency(x, fft=True)
        assert_almost_equal(g0, g1, decimal=5)
        assert_almost_equal(g0, g2, decimal=5)
        assert_almost_equal(g0, g3, decimal=5)

        assert_almost_equal(np.log(g0), np.log(1.0), decimal=1)

    for i in range(5):
        x = np.random.normal(size=100000)
        x = np.repeat(
            x, 3
        )  # e.g. Construct correlated gaussian e.g. [a, b, c] -> [a, a, a, b, b, b, c, c, c]
        g0 = timeseries.statistical_inefficiency(x, fast=False)
        g1 = timeseries.statistical_inefficiency(x, x, fast=False)
        g2 = timeseries.statistical_inefficiency_fft(x)
        g3 = timeseries.statistical_inefficiency(x, fft=True)
        assert_almost_equal(g0, g1, decimal=5)
        assert_almost_equal(g0, g2, decimal=5)
        assert_almost_equal(g0, g3, decimal=5)

        assert_almost_equal(np.log(g0), np.log(3.0), decimal=1)


def test_detectEquil():
    x = np.random.normal(size=10000)
    t, g, Neff_max = timeseries.detect_equilibration(x)


@has_statmodels
def test_detectEquil_binary():
    x = np.random.normal(size=10000)
    t, g, Neff_max = timeseries.detect_equilibration_binary_search(x)


@has_statmodels
def test_compare_detectEquil(show_hist=False):
    """
    compare detect_equilibration implementations (with and without binary search + fft)
    """
    t_res = []
    N = 100
    for _ in range(100):
        A_t = testsystems.correlated_timeseries_example(N=N, tau=5.0) + 2.0
        B_t = testsystems.correlated_timeseries_example(N=N, tau=5.0) + 1.0
        C_t = testsystems.correlated_timeseries_example(N=N * 2, tau=5.0)
        D_t = np.concatenate([A_t, B_t, C_t])
        bs_de = timeseries.detect_equilibration_binary_search(D_t, bs_nodes=10)
        std_de = timeseries.detect_equilibration(D_t, fast=False, nskip=1)
        t_res.append(bs_de[0] - std_de[0])
    t_res_mode = float(stats.mode(t_res)[0][0])
    assert_almost_equal(t_res_mode, 0.0, decimal=1)
    if show_hist:
        import matplotlib.pyplot as plt

        plt.hist(t_res)
        plt.show()


def test_detectEquil_constant_trailing():
    """
    This explicitly tests issue #122, see https://github.com/choderalab/pymbar/issues/122

    We only check that the code doesn't give an exception.  The exact value of Neff can either be
    ~50 if we try to include part of the equilibration samples, or it can be Neff=1 if we find that the
    whole first half is discarded.
    """
    x = np.random.normal(size=100) * 0.01
    x[50:] = 3.0
    # The input data is some MCMC chain where the trailing end of the chain is a constant sequence.
    t, g, Neff_max = timeseries.detect_equilibration(x)


def test_correlationFunctionMultiple():
    """
    tests the truncate and norm feature
    """
    A_t = [testsystems.correlated_timeseries_example(N=10000, tau=10.0) for _ in range(10)]
    corr_norm = timeseries.normalized_fluctuation_correlation_function_multiple(A_kn=A_t)
    corr = timeseries.normalized_fluctuation_correlation_function_multiple(A_kn=A_t, norm=False)
    corr_norm_trun = timeseries.normalized_fluctuation_correlation_function_multiple(
        A_kn=A_t, truncate=True
    )
    corr_trun = timeseries.normalized_fluctuation_correlation_function_multiple(
        A_kn=A_t, norm=False, truncate=True
    )
    assert corr_norm_trun[-1] >= 0
    assert corr_trun[-1] >= 0
    assert corr_norm[0] == 1.0
    assert corr_norm_trun[0] == 1.0
    assert len(corr_trun) == len(corr_norm_trun)
