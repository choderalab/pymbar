from pymbar import timeseries
from pymbar import testsystems
import numpy as np
from scipy import stats
from pymbar.utils_for_testing import eq, skipif
from six.moves import xrange

try:
    import statsmodels.api as sm
    HAVE_STATSMODELS = True
except ImportError as err:
    HAVE_STATSMODELS = False


def generate_data(N=10000, K=10):
    var = np.ones(N)

    for replica in xrange(2, K + 1):
        var = np.concatenate((var, np.ones(N)))

    X = np.random.normal(np.zeros(K * N), var).reshape((K, N)) / 10.0
    Y = np.random.normal(np.zeros(K * N), var).reshape((K, N))

    energy = 10 * (X ** 2) / 2.0 + (Y ** 2) / 2.0

    return X, Y, energy

def test_statistical_inefficiency_single():
    X, Y, energy = generate_data()
    timeseries.statisticalInefficiency(X[0])
    timeseries.statisticalInefficiency(X[0], X[0])
    timeseries.statisticalInefficiency(X[0] ** 2)
    timeseries.statisticalInefficiency(X[0] ** 2, X[0] ** 2)
    timeseries.statisticalInefficiency(energy[0])
    timeseries.statisticalInefficiency(energy[0], energy[0])

    timeseries.statisticalInefficiency(X[0], X[0] ** 2)

    # TODO: Add some checks to test statistical inefficinecies are within normal range



def test_statistical_inefficiency_multiple():
    X, Y, energy = generate_data()
    timeseries.statisticalInefficiencyMultiple(X)
    timeseries.statisticalInefficiencyMultiple(X ** 2)
    timeseries.statisticalInefficiencyMultiple(X[0, :] ** 2)
    timeseries.statisticalInefficiencyMultiple(X[0:2, :] ** 2)
    timeseries.statisticalInefficiencyMultiple(energy)
    # TODO: Add some checks to test statistical inefficinecies are within normal range


@skipif(not HAVE_STATSMODELS, "Skipping FFT based tests because statsmodels not installed.")
def test_statistical_inefficiency_fft():
    X, Y, energy = generate_data()
    timeseries.statisticalInefficiency_fft(X[0])
    timeseries.statisticalInefficiency_fft(X[0] ** 2)
    timeseries.statisticalInefficiency_fft(energy[0])

    g0 = timeseries.statisticalInefficiency_fft(X[0])
    g1 = timeseries.statisticalInefficiency(X[0])
    g2 = timeseries.statisticalInefficiency(X[0], X[0])
    g3 = timeseries.statisticalInefficiency(X[0], fft=True)
    eq(g0, g1)
    eq(g0, g2)
    eq(g0, g3)

@skipif(not HAVE_STATSMODELS, "Skipping FFT based tests because statsmodels not installed.")
def test_statistical_inefficiency_fft_gaussian():

    # Run multiple times to get things with and without negative "spikes" at C(1)
    for i in range(5):
        x = np.random.normal(size=100000)
        g0 = timeseries.statisticalInefficiency(x, fast=False)
        g1 = timeseries.statisticalInefficiency(x, x, fast=False)
        g2 = timeseries.statisticalInefficiency_fft(x)
        g3 = timeseries.statisticalInefficiency(x, fft=True)
        eq(g0, g1, decimal=5)
        eq(g0, g2, decimal=5)
        eq(g0, g3, decimal=5)

        eq(np.log(g0), np.log(1.0), decimal=1)

    for i in range(5):
        x = np.random.normal(size=100000)
        x = np.repeat(x, 3)  # e.g. Construct correlated gaussian e.g. [a, b, c] -> [a, a, a, b, b, b, c, c, c]
        g0 = timeseries.statisticalInefficiency(x, fast=False)
        g1 = timeseries.statisticalInefficiency(x, x, fast=False)
        g2 = timeseries.statisticalInefficiency_fft(x)
        g3 = timeseries.statisticalInefficiency(x, fft=True)
        eq(g0, g1, decimal=5)
        eq(g0, g2, decimal=5)
        eq(g0, g3, decimal=5)

        eq(np.log(g0), np.log(3.0), decimal=1)



def test_detectEquil():
    x = np.random.normal(size=10000)
    (t, g, Neff_max) = timeseries.detectEquilibration(x)

@skipif(not HAVE_STATSMODELS, "Skipping FFT based tests because statsmodels not installed.")
def test_detectEquil_binary():
    x = np.random.normal(size=10000)
    (t, g, Neff_max) = timeseries.detectEquilibration_binary_search(x)

@skipif(not HAVE_STATSMODELS, "Skipping FFT based tests because statsmodels not installed.")
def test_compare_detectEquil(show_hist=False):
    """
    compare detectEquilibration implementations (with and without binary search + fft)
    """
    t_res = []
    N=100
    for _ in xrange(100):
        A_t = testsystems.correlated_timeseries_example(N=N, tau=5.0) + 2.0
        B_t = testsystems.correlated_timeseries_example(N=N, tau=5.0) + 1.0
        C_t = testsystems.correlated_timeseries_example(N=N*2, tau=5.0)
        D_t = np.concatenate([A_t, B_t, C_t])
        bs_de = timeseries.detectEquilibration_binary_search(D_t, bs_nodes=10)
        std_de = timeseries.detectEquilibration(D_t, fast=False, nskip=1)
        t_res.append(bs_de[0]-std_de[0])
    t_res_mode = float(stats.mode(t_res)[0][0])
    eq(t_res_mode,0.,decimal=1)
    if show_hist:
        import matplotlib.pyplot as plt
        plt.hist(t_res)
        plt.show()



def test_detectEquil_constant_trailing():
    # This explicitly tests issue #122, see https://github.com/choderalab/pymbar/issues/122
    x = np.random.normal(size=100) * 0.01
    x[50:] = 3.0
    # The input data is some MCMC chain where the trailing end of the chain is a constant sequence.
    (t, g, Neff_max) = timeseries.detectEquilibration(x)
    """
    We only check that the code doesn't give an exception.  The exact value of Neff can either be
    ~50 if we try to include part of the equilibration samples, or it can be Neff=1 if we find that the
    whole first half is discarded.
    """

def test_correlationFunctionMultiple():
    """
    tests the truncate and norm feature
    """
    A_t = [testsystems.correlated_timeseries_example(N=10000, tau=10.0) for i in range(10)]
    corr_norm = timeseries.normalizedFluctuationCorrelationFunctionMultiple(A_kn=A_t)
    corr = timeseries.normalizedFluctuationCorrelationFunctionMultiple(A_kn=A_t, norm=False)
    corr_norm_trun = timeseries.normalizedFluctuationCorrelationFunctionMultiple(A_kn=A_t, truncate=True)
    corr_trun = timeseries.normalizedFluctuationCorrelationFunctionMultiple(A_kn=A_t, norm=False, truncate=True)
    assert (corr_norm_trun[-1] >= 0)
    assert (corr_trun[-1] >= 0)
    assert (corr_norm[0] == 1.)
    assert (corr_norm_trun[0] == 1.)
    assert (len(corr_trun) == len(corr_norm_trun))
