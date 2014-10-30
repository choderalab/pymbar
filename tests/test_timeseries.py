from pymbar import timeseries
import numpy as np
from pymbar.utils_for_testing import eq, skipif

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


def test_detectEquil_constant_trailing():
    # This explicitly tests issue #122, see https://github.com/choderalab/pymbar/issues/122
    x = np.random.normal(size=100) * 0.01
    x[50:] = 3.0
    # The input data is some MCMC chain where the trailing end of the chain is a constant sequence.
    # The desired 
    (t, g, Neff_max) = timeseries.detectEquilibration(x)
    assert Neff_max < 60, "Should have approximately Neff = 50, found %d" % (N_eff_max)
