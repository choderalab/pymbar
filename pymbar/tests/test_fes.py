import numpy as np
import pytest
from pymbar import FES
from pymbar.utils import ParameterError
from pymbar.utils_for_testing import assert_almost_equal

try:
    import sklearn  # pylint: disable=unused-import

    has_sklearn = True
except ImportError:
    has_sklearn = False


beta = 1.0
z_scale_factor = 12.0


def generate_fes_data(ndim=1, nsamples=1000, K0=20.0, Ku=100.0, gridscale=0.2, xrange=None):

    x0 = np.zeros([ndim])  # center of base potential
    numbrellas = 1
    nperdim = np.zeros([ndim], int)
    if xrange is None:
        xrange = [[-3, 3]] * ndim
    for d in range(ndim):
        nperdim[d] = xrange[d][1] - xrange[d][0] + 1
        numbrellas *= nperdim[d]

    # print("There are a total of {:d} umbrellas.".format(numbrellas))

    # Enumerate umbrella centers, and compute the analytical free energy of that umbrella
    # print("Constructing umbrellas...")

    ksum = (Ku + K0) / beta
    kprod = (Ku * K0) / (beta * beta)
    f_k_analytical = np.zeros(numbrellas)
    xu_i = np.zeros([numbrellas, ndim])  # xu_i[i,:] is the center of umbrella i

    dp = np.zeros(ndim, int)
    dp[0] = 1
    for d in range(1, ndim):
        dp[d] = nperdim[d] * dp[d - 1]

    umbrella_zero = 0
    for i in range(numbrellas):
        center = []
        for d in range(ndim):
            val = gridscale * ((int(i // dp[d])) % nperdim[d] + xrange[d][0])
            center.append(val)
        center = np.array(center)
        xu_i[i, :] = center
        mu2 = np.dot(center, center)
        f_k_analytical[i] = np.log(
            (ndim * np.pi / ksum) ** (3.0 / 2.0) * np.exp(-kprod * mu2 / (2.0 * ksum))
        )
        if np.all(center == 0.0):  # assumes that we have one state that is at the zero.
            umbrella_zero = i
        i += 1
        f_k_analytical -= f_k_analytical[umbrella_zero]

    # print("Generating {:d} samples for each of {:d} umbrellas...".format(nsamples, numbrellas))
    x_n = np.zeros([numbrellas * nsamples, ndim])

    for i in range(numbrellas):
        for dim in range(ndim):
            # Compute mu and sigma for this dimension for sampling from V0(x) + Vu(x).
            # Product of Gaussians: N(x ; a, A) N(x ; b, B) = N(a ; b , A+B) x N(x ; c, C) where
            # C = 1/(1/A + 1/B)
            # c = C(a/A+b/B)
            # A = 1/K0, B = 1/Ku
            sigma = 1.0 / (K0 + Ku)
            mu = sigma * (x0[dim] * K0 + xu_i[i, dim] * Ku)
            # Generate normal deviates for this dimension.
            x_n[i * nsamples : (i + 1) * nsamples, dim] = np.random.normal(
                mu, np.sqrt(sigma), [nsamples]
            )

    u_kn = np.zeros([numbrellas, nsamples * numbrellas])
    # Compute reduced potential due to V0.
    u_n = beta * (K0 / 2) * np.sum((x_n[:, :] - x0) ** 2, axis=1)
    for k in range(numbrellas):
        uu = (
            beta * (Ku / 2) * np.sum((x_n[:, :] - xu_i[k, :]) ** 2, axis=1)
        )  # reduced potential due to umbrella k
        u_kn[k, :] = u_n + uu

    fes_const = K0 / 2.0  # using a quadratic surface, so has same multpliciative value everywhere.

    def bias_potential(x, k_bias):
        dx = x - xu_i[k_bias, :]
        return beta * (Ku / 2.0) * np.dot(dx, dx)

    bias_potentials = [(lambda x, klocal=k: bias_potential(x, klocal)) for k in range(numbrellas)]

    return u_kn, u_n, x_n, f_k_analytical, fes_const, bias_potentials


@pytest.fixture(scope="module")
def fes_1d():

    gridscale = 0.2
    nbinsperdim = 15
    xrange = [[-3, 3]]
    ndim = 1
    nsamples = 1000
    K0 = 20.0
    Ku = 100

    payload = {
        "gridscale": gridscale,
        "nbinsperdim": nbinsperdim,
        "xrange": xrange,
        "ndim": ndim,
        "nsamples": nsamples,
        "K0": K0,
        "Ku": Ku,
    }

    u_kn, u_n, x_n, f_k_analytical, fes_const, bias_potentials = generate_fes_data(
        K0=K0, Ku=Ku, ndim=ndim, nsamples=nsamples, gridscale=gridscale, xrange=xrange
    )
    numbrellas = (np.shape(u_kn))[0]
    N_k = nsamples * np.ones([numbrellas], int)

    # Histogram bins are indexed using the scheme:
    xmin = gridscale * (np.min(xrange[0][0]) - 1 / 2.0)
    xmax = gridscale * (np.max(xrange[0][1]) + 1 / 2.0)
    dx = (xmax - xmin) / nbinsperdim
    nbins = nbinsperdim**ndim
    bin_edges = np.linspace(xmin, xmax, nbins + 1)  # list of bin edges.
    bin_centers = np.zeros([nbins, ndim])

    ibin = 0
    fes_analytical = np.zeros([nbins])
    minmu2 = 1000000
    zeroindex = 0
    # construct the bins and the fes
    for i in range(nbins):
        xbin = xmin + dx * (i + 0.5)
        bin_centers[ibin, 0] = xbin
        mu2 = xbin * xbin
        if mu2 < minmu2:
            minmu2 = mu2
            zeroindex = ibin
        fes_analytical[ibin] = fes_const * mu2
        ibin += 1
    fzero = fes_analytical[zeroindex]
    fes_analytical -= fzero
    bin_n = -1 * np.ones([numbrellas * nsamples], int)

    # Determine indices of those within bounds.
    within_bounds = (x_n[:, 0] >= xmin) & (x_n[:, 0] < xmax)
    # Determine states for these.
    bin_n[within_bounds] = np.floor((x_n[within_bounds, 0] - xmin) / dx)
    # Determine indices of bins that are not empty.
    bin_counts = np.zeros([nbins], int)
    for i in range(nbins):
        bin_counts[i] = (bin_n == i).sum()

    fes = FES(u_kn, N_k)

    # Make a quick calculation to get reference uncertainties
    fes.generate_fes(u_n, x_n, histogram_parameters={"bin_edges": bin_edges})
    results = fes.get_fes(
        bin_centers,
        reference_point="from-specified",
        fes_reference=0.0,
        uncertainty_method="analytical",
    )

    payload["fes"] = fes
    payload["u_kn"] = u_kn
    payload["N_k"] = N_k
    payload["u_n"] = u_n
    payload["x_n"] = x_n
    payload["dx"] = dx
    payload["nbins"] = nbins
    payload["bin_edges"] = bin_edges
    payload["bin_centers"] = bin_centers
    payload["fes_const"] = fes_const
    payload["fes_analytical"] = fes_analytical
    payload["f_k_analytical"] = f_k_analytical
    payload["bias_potentials"] = bias_potentials
    payload["reference_df_i"] = results["df_i"]
    del results

    return payload


@pytest.fixture(scope="module")
def fes_2d():

    xrange = [[-3, 3], [-3, 3]]
    ndim = 2
    nsamples = 300
    nbinsperdim = 10
    gridscale = 0.2
    K0 = 20.0
    Ku = 100
    delta = 0.0001  # to break ties in things being too close.

    payload = {
        "gridscale": gridscale,
        "nbinsperdim": nbinsperdim,
        "xrange": xrange,
        "ndim": ndim,
        "nsamples": nsamples,
        "K0": K0,
        "Ku": Ku,
    }

    u_kn, u_n, x_n, f_k_analytical, fes_const, bias_potentials = generate_fes_data(
        K0=K0, Ku=Ku, ndim=ndim, nsamples=nsamples, gridscale=gridscale, xrange=xrange
    )
    numbrellas = (np.shape(u_kn))[0]
    N_k = nsamples * np.ones([numbrellas], int)
    # print("Solving for free energies of state ...")

    # The dimensionless free energy is the integral of this, and can be computed as:
    #   f(beta,K)           = - ln [ (2*np.pi/(Ko+Ku))^(d/2) exp[ -Ku*Ko mu' mu / 2(Ko +Ku)]
    #   f(beta,K) - fzero   = -Ku*Ko / 2(Ko+Ku)  = 1/(1/(Ku/2) + 1/(K0/2))
    # for computing harmonic samples

    # Can compare the free energies computed with MBAR if desired with f_k_analytical

    xmin = gridscale * (np.min(xrange[0][0]) - 1 / 2.0)
    xmax = gridscale * (np.max(xrange[0][1]) + 1 / 2.0)
    ymin = gridscale * (np.min(xrange[1][0]) - 1 / 2.0)
    ymax = gridscale * (np.max(xrange[1][1]) + 1 / 2.0)
    dx = (xmax - xmin) / nbinsperdim
    dy = (ymax - ymin) / nbinsperdim
    nbins = nbinsperdim**ndim
    bin_centers = np.zeros([nbins, ndim])

    ibin = 0
    fes_analytical = np.zeros([nbins])
    minmu2 = 1000000
    zeroindex = 0
    # construct the bins and the fes
    for i in range(nbinsperdim):
        xbin = xmin + dx * (i + 0.5)
        for j in range(nbinsperdim):
            # Determine (x,y) of bin center.
            ybin = ymin + dy * (j + 0.5)
            bin_centers[ibin, 0] = xbin
            bin_centers[ibin, 1] = ybin
            mu2 = xbin * xbin + ybin * ybin
            if mu2 < minmu2:
                minmu2 = mu2
                zeroindex = ibin
            fes_analytical[ibin] = fes_const * mu2
            ibin += 1
    fzero = fes_analytical[zeroindex]
    fes_analytical -= fzero

    Ntot = int(np.sum(N_k))
    bin_n = -1 * np.ones([Ntot, 2], int)
    # Determine indices of those within bounds.  Outside bounds stays as -1 in that direction.
    within_boundsx = (x_n[:, 0] >= xmin) & (x_n[:, 0] < xmax)
    within_boundsy = (x_n[:, 1] >= ymin) & (x_n[:, 1] < ymax)
    # Determine states for these.
    xgrid = (x_n[within_boundsx, 0] - xmin) / dx
    ygrid = (x_n[within_boundsy, 1] - ymin) / dy
    bin_n[within_boundsx, 0] = xgrid
    bin_n[within_boundsy, 1] = ygrid

    # Determine 2Dindices of bins that are not empty.
    bin_counts = np.zeros(nbins, int)

    for n in range(Ntot):
        b = bin_n[n]
        bin_label = b[0] + nbinsperdim * b[1]
        bin_counts[bin_label] += 1

    bin_edges = [
        np.linspace(xmin, xmax, nbinsperdim + 1),
        np.linspace(ymin, ymax, nbinsperdim + 1),
    ]

    # initialize FES
    fes = FES(u_kn, N_k)

    # Make a quick calculation to get reference uncertainties
    fes.generate_fes(u_n, x_n, histogram_parameters={"bin_edges": bin_edges})
    results = fes.get_fes(
        bin_centers + delta,
        reference_point="from-specified",
        fes_reference=[0, 0],
        uncertainty_method="analytical",
    )

    payload["fes"] = fes
    payload["u_kn"] = u_kn
    payload["N_k"] = N_k
    payload["u_n"] = u_n
    payload["x_n"] = x_n
    payload["dx"] = dx
    payload["nbins"] = nbins
    payload["bin_edges"] = bin_edges
    payload["bin_centers"] = bin_centers
    payload["delta"] = delta
    payload["fes_const"] = fes_const
    payload["fes_analytical"] = fes_analytical
    payload["f_k_analytical"] = f_k_analytical
    payload["bias_potentials"] = bias_potentials
    payload["reference_df_i"] = results["df_i"]
    del results

    return payload


@pytest.mark.parametrize(
    "reference_point",
    [
        "from-lowest",
        "from-specified",
        pytest.param("from-normalization", marks=pytest.mark.xfail(raises=ParameterError)),
        pytest.param("all-differences", marks=pytest.mark.xfail(raises=ParameterError)),
    ],
)
def test_1d_fes_histogram(fes_1d, reference_point):

    fes = fes_1d["fes"]

    histogram_parameters = dict()
    histogram_parameters["bin_edges"] = fes_1d["bin_edges"]
    fes.generate_fes(fes_1d["u_n"], fes_1d["x_n"], histogram_parameters=histogram_parameters)
    results = fes.get_fes(
        fes_1d["bin_centers"],
        reference_point=reference_point,
        fes_reference=0.0,
        uncertainty_method="analytical",
    )
    f_ih = results["f_i"]
    df_ih = results["df_i"]

    z = np.zeros(fes_1d["nbins"])
    for i in range(0, fes_1d["nbins"]):
        if df_ih[i] != 0:
            z[i] = np.abs(fes_1d["fes_analytical"][i] - f_ih[i]) / df_ih[i]
        else:
            z[i] = 0
    assert_almost_equal(z / z_scale_factor, np.zeros(len(z)), decimal=0)


def base_1d_fes_kde(fes_1d, gen_kwargs, reference_point):

    fes = fes_1d["fes"]

    kde_parameters = dict()
    kde_parameters["bandwidth"] = 0.5 * fes_1d["dx"]
    # no analytical uncertainty for kde yet, have to use bootstraps
    fes.generate_fes(
        fes_1d["u_n"], fes_1d["x_n"], fes_type="kde", kde_parameters=kde_parameters, **gen_kwargs
    )
    results_kde = fes.get_fes(
        fes_1d["bin_centers"],
        reference_point=reference_point,
        fes_reference=0.0,
        uncertainty_method=None,
    )
    f_ik = results_kde["f_i"]
    # df_ih = results_kde['df_i']
    # Just use the reference for now
    df_ih = fes_1d["reference_df_i"]
    if df_ih is None:
        df_ih = fes_1d["reference_df_i"]

    z = np.zeros(fes_1d["nbins"])
    for i in range(0, fes_1d["nbins"]):
        if df_ih[i] != 0:
            z[i] = np.abs(fes_1d["fes_analytical"][i] - f_ik[i]) / df_ih[i]
        else:
            z[i] = 0
    assert_almost_equal(z / z_scale_factor, np.zeros(len(z)), decimal=0)


@pytest.mark.skipif(
    not has_sklearn,
    reason="Must have sklearn (package scikit-learn) installed to use KDE type FES",
)
@pytest.mark.parametrize("gen_kwargs", [{}, {"seed": 10}])
@pytest.mark.parametrize(
    "reference_point",
    [
        "from-lowest",
        "from-specified",
        pytest.param("from-normalization", marks=pytest.mark.xfail(raises=ParameterError)),
        pytest.param("all-differences", marks=pytest.mark.xfail(raises=ParameterError)),
    ],
)
def test_1d_fes_kde(fes_1d, gen_kwargs, reference_point):
    base_1d_fes_kde(fes_1d, gen_kwargs, reference_point)


@pytest.mark.skipif(
    not has_sklearn,
    reason="Must have sklearn (package scikit-learn) installed to use KDE type FES",
)
def test_1d_fes_kde_bootstraped(fes_1d):
    # Make tests faster overall by only testing bootstraps once.
    # Once more paths are fixed, this can be folded into the gen_kwargs of the more general test
    base_1d_fes_kde(fes_1d, {"n_bootstraps": 2}, "from-lowest")


def base_1d_fes_spline(fes_1d, gen_kwargs, reference_point):

    fes = fes_1d["fes"]
    bin_centers = fes_1d["bin_centers"]
    fes_analytical = fes_1d["fes_analytical"]

    # now spline
    spline_parameters = dict()
    spline_parameters["spline_weights"] = "unbiasedstate"  # fastest spline method
    spline_parameters["nspline"] = 4
    spline_parameters["spline_initialize"] = "explicit"

    spline_parameters["xinit"] = bin_centers[:, 0]
    spline_parameters["yinit"] = fes_analytical  # cheat by starting with "true" answer - for speed

    spline_parameters["xrange"] = fes_1d["xrange"][0]
    spline_parameters["fkbias"] = fes_1d["bias_potentials"]

    spline_parameters["kdegree"] = 3
    spline_parameters["optimization_algorithm"] = "Newton-CG"
    spline_parameters["optimize_options"] = {"disp": True, "tol": 10 ** (-6)}
    spline_parameters["objective"] = "ml"
    spline_parameters["map_data"] = None

    # no analytical uncertainty for kde yet, have to use bootstraps
    fes.generate_fes(
        fes_1d["u_n"],
        fes_1d["x_n"],
        fes_type="spline",
        spline_parameters=spline_parameters,
        **gen_kwargs
    )
    # Something wrong with unbiased state?
    results_spline = fes.get_fes(
        bin_centers, reference_point=reference_point, fes_reference=0.0, uncertainty_method=None
    )
    f_is = results_spline["f_i"]
    # df_ih = results_spline['df_i']
    # Just use the reference for now
    df_ih = fes_1d["reference_df_i"]
    if df_ih is None:
        df_ih = fes_1d["reference_df_i"]

    z = np.zeros(fes_1d["nbins"])
    for i in range(0, fes_1d["nbins"]):
        if df_ih[i] != 0:
            z[i] = np.abs(fes_analytical[i] - f_is[i]) / df_ih[i]
        else:
            z[i] = 0
    assert_almost_equal(z / z_scale_factor, np.zeros(len(z)), decimal=0)


@pytest.mark.parametrize("gen_kwargs", [{}, {"seed": 10}])
@pytest.mark.parametrize(
    "reference_point",
    [
        "from-lowest",
        # Lots of things are wrong with this, not going to debug it at this time.
        # pytest.param('from-specified', marks=pytest.mark.xfail(reason="Not sure why this fails")),
        # pytest.param('from-normalization', marks=pytest.mark.xfail),
        # pytest.param('all-differences', marks=pytest.mark.xfail)
    ],
)
def test_1d_fes_spline(fes_1d, gen_kwargs, reference_point):
    base_1d_fes_spline(fes_1d, gen_kwargs, reference_point)


def test_1d_fes_spline_bootstraped(fes_1d):
    # Make tests faster overall by only testing bootstraps once.
    # Once more paths are fixed, this can be folded into the gen_kwargs of the more general test
    base_1d_fes_spline(fes_1d, {"n_bootstraps": 2}, "from-lowest")


@pytest.mark.parametrize(
    "reference_point",
    [
        "from-lowest",
        "from-specified",
        pytest.param("from-normalization", marks=pytest.mark.xfail(raises=ParameterError)),
        pytest.param("all-differences", marks=pytest.mark.xfail(raises=ParameterError)),
    ],
)
def test_2d_fes_histogram(fes_2d, reference_point):

    """testing fes_generate_fes and fes_get_fes in 2D"""

    fes = fes_2d["fes"]
    fes_analytical = fes_2d["fes_analytical"]

    # set histogram parameters.
    histogram_parameters = dict()
    histogram_parameters["bin_edges"] = fes_2d["bin_edges"]
    fes.generate_fes(
        fes_2d["u_n"],
        fes_2d["x_n"],
        fes_type="histogram",
        histogram_parameters=histogram_parameters,
    )

    results = fes.get_fes(
        fes_2d["bin_centers"] + fes_2d["delta"],
        reference_point=reference_point,
        fes_reference=[0, 0],
    )
    f_ih = results["f_i"]
    df_ih = fes_2d["reference_df_i"]

    nbins = fes_2d["nbins"]
    z = np.zeros(nbins)
    for i in range(0, nbins):
        if df_ih[i] != 0:
            z[i] = np.abs(fes_analytical[i] - f_ih[i]) / df_ih[i]
        else:
            z[i] = 0
    assert_almost_equal(z / z_scale_factor, np.zeros(len(z)), decimal=0)


@pytest.mark.skipif(
    not has_sklearn,
    reason="Must have sklearn (package scikit-learn) installed to use KDE type FES",
)
@pytest.mark.parametrize("gen_kwargs", [{}, {"seed": 10}])
@pytest.mark.parametrize(
    "reference_point",
    [
        "from-lowest",
        "from-specified",
        pytest.param("from-normalization", marks=pytest.mark.xfail(raises=ParameterError)),
        pytest.param("all-differences", marks=pytest.mark.xfail(raises=ParameterError)),
    ],
)
def test_2d_fes_kde(fes_2d, gen_kwargs, reference_point):

    fes = fes_2d["fes"]
    fes_analytical = fes_2d["fes_analytical"]

    # set kde parameters
    kde_parameters = dict()
    kde_parameters["bandwidth"] = 0.5 * fes_2d["dx"]
    fes.generate_fes(
        fes_2d["u_n"], fes_2d["x_n"], fes_type="kde", kde_parameters=kde_parameters, **gen_kwargs
    )
    # I don't know if this needs the +delta
    results_kde = fes.get_fes(
        fes_2d["bin_centers"] + fes_2d["delta"],
        reference_point=reference_point,
        fes_reference=[0, 0],
    )

    f_ik = results_kde["f_i"]
    df_ih = fes_2d["reference_df_i"]

    nbins = fes_2d["nbins"]
    z = np.zeros(nbins)
    for i in range(0, nbins):
        if df_ih[i] != 0:
            z[i] = np.abs(fes_analytical[i] - f_ik[i]) / df_ih[i]
        else:
            z[i] = 0
    assert_almost_equal(z / z_scale_factor, np.zeros(len(z)), decimal=0)
