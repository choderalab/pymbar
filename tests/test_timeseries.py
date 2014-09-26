from pymbar import timeseries
import numpy as np
from pymbar.utils_for_testing import eq


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
    timeseries.statisticalInefficiency(X[0] ** 2)
    timeseries.statisticalInefficiency(energy[0])
    # TODO: Add some checks to test statistical inefficinecies are within normal range



def test_statistical_inefficiency_multiple():
    X, Y, energy = generate_data()
    timeseries.statisticalInefficiencyMultiple(X)
    timeseries.statisticalInefficiencyMultiple(X ** 2)
    timeseries.statisticalInefficiencyMultiple(X[0, :] ** 2)
    timeseries.statisticalInefficiencyMultiple(X[0:2, :] ** 2)
    timeseries.statisticalInefficiencyMultiple(energy)
    # TODO: Add some checks to test statistical inefficinecies are within normal range


def test_statistical_inefficiency_fft():
    X, Y, energy = generate_data()
    timeseries.statisticalInefficiency_fft(X[0])
    timeseries.statisticalInefficiency_fft(X[0] ** 2)
    timeseries.statisticalInefficiency_fft(energy[0])
    
    g0 = timeseries.statisticalInefficiency(X[0])
    g = timeseries.statisticalInefficiency(X[0])
    eq(g0, g)


def test_statistical_inefficiency_fft_gaussian():
    
    # Run multiple times to get things with and without negative "spikes" at C(1)
    for i in range(5):
        x = np.random.normal(size=100000)
        g0 = timeseries.statisticalInefficiency(x, fast=False)
        g = timeseries.statisticalInefficiency_fft(x)
        eq(g0, g, decimal=5)
        
        eq(np.log(g0), np.log(1.0), decimal=1)

    for i in range(5):
        x = np.random.normal(size=100000)
        x = np.repeat(x, 3)  # e.g. Construct correlated gaussian e.g. [a, b, c] -> [a, a, a, b, b, b, c, c, c]
        g0 = timeseries.statisticalInefficiency(x, fast=False)
        g = timeseries.statisticalInefficiency_fft(x)
        eq(g0, g, decimal=5)
        eq(np.log(g0), np.log(3.0), decimal=1)
