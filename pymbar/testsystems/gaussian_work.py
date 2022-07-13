import numpy as np


def gaussian_work_example(N_F=200, N_R=200, mu_F=2.0, DeltaF=None, sigma_F=1.0, seed=None):
    """Generate samples from forward and reverse Gaussian work distributions.

    Parameters
    ----------
    N_F : int, optional
        number of forward measurements (default: 200)
    N_R : float, optional
        number of reverse measurements (default: 200)
    mu_F : float, optional
        mean of forward work distribution (default: 2.0)
    DeltaF : float, optional
        the free energy difference, which can be specified instead of mu_F (default: None)
    sigma_F : float, optional
        variance of the forward work distribution (default: 1.0)
    seed : int, optional
        If not None, specify the numpy random number seed. Old state is restored after completion.

    Returns
    -------
    w_F : np.ndarray, dtype=float
        forward work values
    w_R : np.ndarray, dtype=float
        reverse work values

    Notes
    -----
    By the Crooks fluctuation theorem (CFT), the forward and backward work distributions are related by

    P_R(-w) = P_F(w) \\exp[DeltaF - w]

    If the forward distribution is Gaussian with mean \\mu_F and std dev \\sigma_F, then

    P_F(w) = (2 \\pi)^{-1/2} \\sigma_F^{-1} \\exp[-(w - \\mu_F)^2 / (2 \\sigma_F^2)]

    With some algebra, we then find the corresponding mean and std dev of the reverse distribution are

    \\mu_R = - \\mu_F + \\sigma_F^2
    \\sigma_R = \\sigma_F \\exp[\\mu_F - \\sigma_F^2 / 2 + \\Delta F]

    where all quantities are in reduced units (e.g. divided by kT).

    Note that \\mu_F and \\Delta F are not independent!  By the Zwanzig relation,

    E_F[exp(-w)] = \\int dw \\exp(-w) P_F(w) = \\exp[-\\Delta F]

    which, with some integration, gives

    \\Delta F = \\mu_F + \\sigma_F^2/2

    which can be used to determine either \\mu_F or \\DeltaF.

    Examples
    --------

    Generate work values with default parameters.

    >>> [w_F, w_R] = gaussian_work_example()

    Generate 50 forward work values and 70 reverse work values.

    >>> [w_F, w_R] = gaussian_work_example(N_F=50, N_R=70)

    Generate work values specifying the work distribution parameters.

    >>> [w_F, w_R] = gaussian_work_example(mu_F=3.0, sigma_F=2.0)

    Generate work values specifying the work distribution parameters, specifying free energy difference instead of mu_F.

    >>> [w_F, w_R] = gaussian_work_example(mu_F=None, DeltaF=3.0, sigma_F=2.0)

    Generate work values with known seed to ensure reproducibility for testing.

    >>> [w_F, w_R] = gaussian_work_example(seed=0)

    """

    # Make sure either mu_F or DeltaF, but not both, are specified.
    if (mu_F is not None) and (DeltaF is not None):
        raise ValueError(
            "mu_F and DeltaF are not independent, and cannot both be specified; one must be set to None."
        )
    if (mu_F is None) and (DeltaF is None):
        raise ValueError("Either mu_F or DeltaF must be specified.")
    if mu_F is None:
        mu_F = DeltaF + sigma_F**2 / 2.0
    if DeltaF is None:
        DeltaF = mu_F - sigma_F**2 / 2.0

    # Set random number generator into a known state for reproducibility.
    random = np.random.RandomState(seed)

    # Determine mean and variance of reverse work distribution by Crooks
    # fluctuation theorem (CFT).
    mu_R = -mu_F + sigma_F**2
    sigma_R = sigma_F * np.exp(mu_F - sigma_F**2 / 2.0 - DeltaF)

    # Draw samples from forward and reverse distributions.
    w_F = random.randn(N_F) * sigma_F + mu_F
    w_R = random.randn(N_R) * sigma_R + mu_R

    return [w_F, w_R]
