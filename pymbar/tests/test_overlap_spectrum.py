"""Real overlap spectrum, including states without samples."""
import numpy as np
import pytest
from pymbar import MBAR


@pytest.mark.parametrize("counts", [[20, 30, 40], [20, 0, 40], [20, 0, 0]])
def test_overlap_real_spectrum_with_unsampled_states(counts):
    rng = np.random.default_rng(315)
    centers = np.array([-0.5, 0.0, 0.75])
    x = np.concatenate([rng.normal(center, 1.0, n) for center, n in zip(centers, counts)])
    u_kn = 0.5 * (x[None, :] - centers[:, None]) ** 2
    mbar = MBAR(u_kn, np.array(counts), relative_tolerance=1e-12)
    result = mbar.compute_overlap()
    weights = mbar.weights()
    expected = np.array(counts) * (weights.T @ weights)
    reference = np.linalg.eigvals(expected)
    np.testing.assert_allclose(reference.imag, 0, atol=1e-12)
    assert np.isrealobj(result["scalar"])
    assert np.isrealobj(result["eigenvalues"])
    np.testing.assert_array_equal(result["matrix"], expected)
    np.testing.assert_allclose(result["eigenvalues"], np.sort(reference.real)[::-1], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(expected.sum(axis=1), 1, atol=1e-10)
    np.testing.assert_allclose(result["scalar"], 1 - result["eigenvalues"][1], atol=1e-12)
