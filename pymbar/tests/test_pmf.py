import numpy as np
import pytest
from pymbar import MBAR
from pymbar import PMF
from pymbar.testsystems import harmonic_oscillators, exponential_distributions
from pymbar.utils_for_testing import assert_equal, assert_almost_equal
try:
    import sklearn
    has_sklearn = True
except ImportError:
    has_sklearn = False


beta = 1.0
z_scale_factor = 12.0


def generate_pmf_data(ndim=1, nsamples=1000, K0=20.0, Ku=100.0, gridscale=0.2, xrange=None):

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

    ksum = (Ku+K0)/beta
    kprod = (Ku*K0)/(beta*beta)
    f_k_analytical = np.zeros(numbrellas)
    xu_i = np.zeros([numbrellas, ndim])  # xu_i[i,:] is the center of umbrella i
    
    dp = np.zeros(ndim, int)
    dp[0] = 1
    for d in range(1, ndim):
        dp[d] = nperdim[d]*dp[d-1]

    umbrella_zero = 0
    for i in range(numbrellas):
        center = []
        for d in range(ndim):
            val = gridscale*((int(i//dp[d])) % nperdim[d] + xrange[d][0])
            center.append(val)
        center = np.array(center)
        xu_i[i, :] = center
        mu2 = np.dot(center, center)
        f_k_analytical[i] = np.log((ndim*np.pi/ksum)**(3.0/2.0) * np.exp(-kprod*mu2/(2.0*ksum)))
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
            mu = sigma * (x0[dim]*K0 + xu_i[i, dim]*Ku)
            # Generate normal deviates for this dimension.
            x_n[i*nsamples:(i+1)*nsamples, dim] = np.random.normal(mu, np.sqrt(sigma), [nsamples])

    u_kn = np.zeros([numbrellas, nsamples*numbrellas])
    # Compute reduced potential due to V0.
    u_n = beta*(K0/2)*np.sum((x_n[:, :] - x0)**2, axis=1)
    for k in range(numbrellas):
        uu = beta*(Ku/2)*np.sum((x_n[:, :] - xu_i[k, :])**2, axis=1)  # reduced potential due to umbrella k
        u_kn[k, :] = u_n + uu

    pmf_const = K0/2.0  # using a quadratic surface, so has same multpliciative value everywhere.

    def bias_potential(x, k_bias):
        dx = x - xu_i[k_bias, :]
        return beta*(Ku/2.0) * np.dot(dx, dx)

    bias_potentials = [(lambda x, klocal=k: bias_potential(x, klocal)) for k in range(numbrellas)]

    return u_kn, u_n, x_n, f_k_analytical, pmf_const, bias_potentials


@pytest.fixture(scope='module')
def pmf_1d():

    gridscale = 0.2
    nbinsperdim = 15
    xrange = [[-3, 3]]
    ndim = 1
    nsamples = 1000
    K0 = 20.0
    Ku = 100

    payload = {
        'gridscale': gridscale,
        'nbinsperdim': nbinsperdim,
        'xrange': xrange,
        'ndim': ndim,
        'nsamples': nsamples,
        'K0': K0,
        'Ku': Ku
    }

    u_kn, u_n, x_n, f_k_analytical, pmf_const, bias_potentials = generate_pmf_data(
        K0=K0,
        Ku=Ku,
        ndim=ndim,
        nsamples=nsamples,
        gridscale=gridscale,
        xrange=xrange)
    numbrellas = (np.shape(u_kn))[0]
    N_k = nsamples * np.ones([numbrellas], int)

    # Histogram bins are indexed using the scheme:
    xmin = gridscale * (np.min(xrange[0][0]) - 1 / 2.0)
    xmax = gridscale * (np.max(xrange[0][1]) + 1 / 2.0)
    dx = (xmax - xmin) / nbinsperdim
    nbins = nbinsperdim ** ndim
    bin_edges = [np.linspace(xmin, xmax, nbins + 1)]  # list of bin edges.
    bin_centers = np.zeros([nbins, ndim])

    ibin = 0
    pmf_analytical = np.zeros([nbins])
    minmu2 = 1000000
    zeroindex = 0
    # construct the bins and the pmf
    for i in range(nbins):
        xbin = xmin + dx * (i + 0.5)
        bin_centers[ibin, 0] = xbin
        mu2 = xbin * xbin
        if (mu2 < minmu2):
            minmu2 = mu2
            zeroindex = ibin
        pmf_analytical[ibin] = pmf_const * mu2
        ibin += 1
    fzero = pmf_analytical[zeroindex]
    pmf_analytical -= fzero
    bin_n = -1 * np.ones([numbrellas * nsamples], int)

    # Determine indices of those within bounds.
    within_bounds = (x_n[:, 0] >= xmin) & (x_n[:, 0] < xmax)
    # Determine states for these.
    bin_n[within_bounds] = np.floor((x_n[within_bounds, 0] - xmin) / dx)
    # Determine indices of bins that are not empty.
    bin_counts = np.zeros([nbins], int)
    for i in range(nbins):
        bin_counts[i] = (bin_n == i).sum()

    pmf = PMF(u_kn, N_k)

    payload['pmf'] = pmf
    payload['u_kn'] = u_kn
    payload['N_k'] = N_k
    payload['u_n'] = u_n
    payload['x_n'] = x_n
    payload['dx'] = dx
    payload['nbins'] = nbins
    payload['bin_edges'] = bin_edges
    payload['bin_centers'] = bin_centers
    payload['pmf_const'] = pmf_const
    payload['pmf_analytical'] = pmf_analytical
    payload['f_k_analytical'] = f_k_analytical
    payload['bias_potentials'] = bias_potentials

    return payload


def test_1d_pmf_histogram(pmf_1d):

    pmf = pmf_1d['pmf']

    histogram_parameters = dict()
    histogram_parameters['bin_edges'] = pmf_1d['bin_edges']
    pmf.generatePMF(pmf_1d['u_n'], pmf_1d['x_n'], histogram_parameters=histogram_parameters)
    results = pmf.getPMF(pmf_1d['bin_centers'], uncertainties='from-specified', pmf_reference=0.0)
    f_ih = results['f_i']
    df_ih = results['df_i']

    z = np.zeros(pmf_1d['nbins'])
    for i in range(0, pmf_1d['nbins']):
        if df_ih[i] != 0:
            z[i] = np.abs(pmf_1d['pmf_analytical'][i]-f_ih[i])/df_ih[i]
        else:
            z[i] = 0
    assert_almost_equal(z / z_scale_factor, np.zeros(len(z)), decimal=0)


@pytest.mark.skipif(not has_sklearn, reason="Must have sklearn installed to use KDE type PMF")
def test_1d_pmf_kde(pmf_1d):

    pmf = pmf_1d['pmf']

    kde_parameters = dict()
    kde_parameters['bandwidth'] = 0.5*pmf_1d['dx']
    # no analytical uncertainty for kde yet, have to use bootstraps
    pmf.generatePMF(pmf_1d['u_n'], pmf_1d['x_n'], pmf_type='kde', kde_parameters=kde_parameters, nbootstraps=10)
    results_kde = pmf.getPMF(pmf_1d['bin_centers'], uncertainties='from-specified', pmf_reference=0.0)
    f_ik = results_kde['f_i']
    df_ih = results_kde['df_i']  # Bootstrapped

    z = np.zeros(pmf_1d['nbins'])
    for i in range(0, pmf_1d['nbins']):
       if df_ih[i] != 0:
           z[i] = np.abs(pmf_1d['pmf_analytical'][i]-f_ik[i])/df_ih[i]
       else:
           z[i] = 0
    assert_almost_equal(z / z_scale_factor, np.zeros(len(z)), decimal=0)


def test_1d_pmf_spline(pmf_1d):

    pmf = pmf_1d['pmf']
    bin_centers = pmf_1d['bin_centers']
    pmf_analytical = pmf_1d['pmf_analytical']

    # now spline
    spline_parameters = dict()
    spline_parameters['spline_weights'] = 'simplesum'  # fastest spline method
    spline_parameters['nspline'] = 4
    spline_parameters['spline_initialize'] = 'explicit'

    spline_parameters['xinit'] = bin_centers[:, 0]
    spline_parameters['yinit'] = pmf_analytical  # cheat by starting with "true" answer - for speed

    spline_parameters['xrange'] = pmf_1d['xrange'][0]
    spline_parameters['fkbias'] = pmf_1d['bias_potentials']

    spline_parameters['kdegree'] = 3
    spline_parameters['optimization_algorithm'] = 'Newton-CG'
    spline_parameters['optimize_options'] = {'disp': True, 'tol': 10**(-6)}
    spline_parameters['objective'] = 'ml'
    spline_parameters['map_data'] = None

    # TODO: Is this u_kn for spline or u_n like all the others? Right now I have it as u_kn as thats what it was
    # no analytical uncertainty for kde yet, have to use bootstraps
    pmf.generatePMF(pmf_1d['u_kn'], pmf_1d['x_n'], pmf_type='spline', spline_parameters=spline_parameters,
                    nbootstraps=1)
    results_spline = pmf.getPMF(bin_centers, uncertainties='from-lowest')  # something wrong with unbiased state?
    f_is = results_spline['f_i']
    df_ih = results_spline['df_i']  # Bootstrapped

    z = np.zeros(pmf_1d['nbins'])
    for i in range(0, pmf_1d['nbins']):
        if df_ih[i] != 0:
            z[i] = np.abs(pmf_analytical[i]-f_is[i])/df_ih[i]
        else:
            z[i] = 0
    assert_almost_equal(z / z_scale_factor, np.zeros(len(z)), decimal=0)


def test_pmf_getPMF_2d():

    """ testing pmf_generatePMF and pmf_getPMF in 2D """

    xrange = [[-3,3],[-3,3]]
    ndim = 2
    nsamples = 300
    nbinsperdim = 10
    gridscale = 0.2

    u_kn, u_n, x_n, f_k_analytical, pmf_const, bias_potentials = generate_pmf_data(
        K0=20.0, Ku=100.0, ndim=ndim, nsamples = nsamples, gridscale = gridscale, xrange=xrange)
    numbrellas = (np.shape(u_kn))[0]
    N_k = nsamples*np.ones([numbrellas], int)
    # print("Solving for free energies of state ...")

    # The dimensionless free energy is the integral of this, and can be computed as:
    #   f(beta,K)           = - ln [ (2*np.pi/(Ko+Ku))^(d/2) exp[ -Ku*Ko mu' mu / 2(Ko +Ku)]
    #   f(beta,K) - fzero   = -Ku*Ko / 2(Ko+Ku)  = 1/(1/(Ku/2) + 1/(K0/2))
    # for computing harmonic samples

    # Can compare the free energies computed with MBAR if desired with f_k_analytical

    xmin = gridscale*(np.min(xrange[0][0])-1/2.0)
    xmax = gridscale*(np.max(xrange[0][1])+1/2.0)
    ymin = gridscale*(np.min(xrange[1][0])-1/2.0)
    ymax = gridscale*(np.max(xrange[1][1])+1/2.0)
    dx = (xmax-xmin)/nbinsperdim
    dy = (ymax-ymin)/nbinsperdim
    nbins = nbinsperdim**ndim
    bin_centers = np.zeros([nbins,ndim])

    ibin = 0
    pmf_analytical = np.zeros([nbins])
    minmu2 = 1000000
    zeroindex = 0
    # construct the bins and the pmf
    for i in range(nbinsperdim):
        xbin = xmin + dx * (i + 0.5)
        for j in range(nbinsperdim):
            # Determine (x,y) of bin center.
            ybin = ymin + dy * (j + 0.5)
            bin_centers[ibin,0] = xbin
            bin_centers[ibin,1] = ybin
            mu2 = xbin*xbin+ybin*ybin
            if (mu2 < minmu2):
                minmu2 = mu2
                zeroindex = ibin
            pmf_analytical[ibin] = pmf_const*mu2
            ibin += 1
    fzero = pmf_analytical[zeroindex]
    pmf_analytical -= fzero

    Ntot = np.sum(N_k)
    bin_n = -1*np.ones([Ntot,2],int)
    # Determine indices of those within bounds.  Outside bounds stays as -1 in that direction.
    within_boundsx = (x_n[:,0] >= xmin) & (x_n[:,0] < xmax)
    within_boundsy = (x_n[:,1] >= ymin) & (x_n[:,1] < ymax)
    # Determine states for these.
    xgrid = (x_n[within_boundsx,0]-xmin)/dx
    ygrid = (x_n[within_boundsy,1]-ymin)/dy
    bin_n[within_boundsx,0] = xgrid
    bin_n[within_boundsy,1] = ygrid

    # Determine 2Dindices of bins that are not empty.
    bin_counts = np.zeros(nbins, int)

    for n in range(Ntot):
        b = bin_n[n]
        bin_label = b[0] + nbinsperdim*b[1]
        bin_counts[bin_label] += 1

    # initialize PMF
    pmf = PMF(u_kn, N_k)

    # set histogram parameters.
    histogram_parameters = dict()
    histogram_parameters['bin_edges'] = [np.linspace(xmin,xmax,nbinsperdim+1),np.linspace(ymin,ymax,nbinsperdim+1)] # list of histogram edges.
    pmf.generatePMF(u_n, x_n, pmf_type = 'histogram', histogram_parameters = histogram_parameters)
    delta = 0.0001  # to break ties in things being too close.

    results = pmf.getPMF(bin_centers+delta, uncertainties = 'from-specified', pmf_reference = [0,0])
    f_ih = results['f_i']
    df_ih = results['df_i']

    z = np.zeros(nbins)

    for i in range(0,nbins):
        if df_ih[i] != 0:
            z[i] = np.abs(pmf_analytical[i]-f_ih[i])/df_ih[i]
        else:
            z[i] = 0
    assert_almost_equal(z / z_scale_factor, np.zeros(len(z)), decimal=0)

    # set kde parameters
    #kde_parameters = dict()
    #kde_parameters['bandwidth'] = 0.5*dx
    #pmf.generatePMF(u_n, x_n, pmf_type = 'kde', kde_parameters = kde_parameters)
    #results_kde = pmf.getPMF(bin_centers, uncertainties='from-specified',pmf_reference = [0,0])

    #f_ik = results_kde['f_i']
    ## no analytical result for kde

    #for i in range(0,nbins):
    #    if df_ih[i] != 0:
    #        z[i] = np.abs(pmf_analytical[i]-f_ik[i])/df_ih[i]
    #    else:
    #        z[i] = 0
    #assert_almost_equal(z / z_scale_factor, np.zeros(len(z)), decimal=0)
