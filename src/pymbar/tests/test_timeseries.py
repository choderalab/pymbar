from pymbar import timeseries
import numpy as np


def generate_data(N=10000, K=10):
    var = np.ones(N)

    for replica in xrange(2, K + 1):
        var = np.concatenate((var, np.ones(N)))

    X = np.random.normal(np.zeros(K * N), var).reshape((K, N)) / 10.0
    Y = np.random.normal(np.zeros(K * N), var).reshape((K, N))

    energy = 10 * (X ** 2) / 2.0 + (Y ** 2) / 2.0

    return X, Y, energy


def test_statistical_inefficiency_multiple():
    X, Y, energy = generate_data()
    timeseries.statisticalInefficiencyMultiple(X)
    timeseries.statisticalInefficiencyMultiple(X ** 2)
    timeseries.statisticalInefficiencyMultiple(X[0, :] ** 2)
    timeseries.statisticalInefficiencyMultiple(X[0:2, :] ** 2)
    timeseries.statisticalInefficiencyMultiple(energy)
    # TODO: Add some checks to test statistical inefficinecies are within normal range
