import numpy as np
import pytest
from pymbar import MBAR
from pmf import PMF
from pymbar.testsystems import harmonic_oscillators, exponential_distributions
from pymbar.utils_for_testing import assert_equal, assert_almost_equal

def generate_pmf_data(ndim=1, nbinsperdim=15, nsamples=1000, K0=20.0, Ku=100.0, gridscale=0.2, xrange=[[-3,3]]):

    beta = 1.0
    x0 = numpy.zeros([ndim], numpy.float64) # center of base potential
    numbrellas = 1
    nperdim = numpy.zeros([ndim],int)
    for d in range(ndim):
        nperdim[d] = xrange[d][1] - xrange[d][0] + 1
        numbrellas *= nperdim[d]

    # print("There are a total of {:d} umbrellas.".format(numbrellas))

    # Enumerate umbrella centers, and compute the analytical free energy of that umbrella
    # print("Constructing umbrellas...")

    ksum = (Ku+K0)/beta
    kprod = (Ku*K0)/(beta*beta)
    f_k_analytical = numpy.zeros(numbrellas, numpy.float64);
    xu_i = numpy.zeros([numbrellas, ndim], numpy.float64) # xu_i[i,:] is the center of umbrella i
    
    dp = numpy.zeros(ndim,int)
    dp[0] = 1
    for d in range(1,ndim):
        dp[d] = nperdim[d]*dp[d-1]

    umbrella_zero = 0
    for i in range(numbrellas):
        center = []
        for d in range(ndim):
            val = gridscale*((int(i//dp[d])) % nperdim[d] + xrange[d][0])
            center.append(val)
        center = numpy.array(center)
    xu_i[i,:] = center
    mu2 = numpy.dot(center,center)
    f_k_analytical[i] = numpy.log((ndim*numpy.pi/ksum)**(3.0/2.0) *numpy.exp(-kprod*mu2/(2.0*ksum)))
    if numpy.all(center==0.0):  # assumes that we have one state that is at the zero.
        umbrella_zero = i
    i += 1
    f_k_analytical -= f_k_analytical[umbrella_zero]

    # print("Generating {:d} samples for each of {:d} umbrellas...".format(nsamples, numbrellas))
    x_n = numpy.zeros([numbrellas * nsamples, ndim], numpy.float64)

    for i in range(numbrellas):
        for dim in range(ndim):
            # Compute mu and sigma for this dimension for sampling from V0(x) + Vu(x).
            # Product of Gaussians: N(x ; a, A) N(x ; b, B) = N(a ; b , A+B) x N(x ; c, C) where
            # C = 1/(1/A + 1/B)
            # c = C(a/A+b/B)
            # A = 1/K0, B = 1/Ku
            sigma = 1.0 / (K0 + Ku)
            mu = sigma * (x0[dim]*K0 + xu_i[i,dim]*Ku)
            # Generate normal deviates for this dimension.
            x_n[i*nsamples:(i+1)*nsamples,dim] = numpy.random.normal(mu, numpy.sqrt(sigma), [nsamples])

    u_kn = numpy.zeros([numbrellas, nsamples*numbrellas], numpy.float64)
    # Compute reduced potential due to V0.
    u_n = beta*(K0/2)*numpy.sum((x_n[:,:] - x0)**2, axis=1)
    for k in range(numbrellas):
        uu = beta*(Ku/2)*numpy.sum((x_n[:,:] - xu_i[k,:])**2, axis=1) # reduced potential due to umbrella k
        u_kn[k,:] = u_n + uu

    return u_kn, u_n, x_n, f_k_analytical


def test_pmf_getPMF():

    """ testing pmf_generatePMF and pmf_getPMF """

    gridscale = 0.2 
    nbinsperdim = 15
    xrange = [[-3,3]]
    ndim = 1
    u_kn, u_n, x_n, f_k_analytical = generate_pmf_data(
        K0=20.0, Ku = 100.0, ndim=ndim, nbinsperdim=nbinsperdim, nsamples=1000, gridscale=gridscale, xrange=xrange)
    numbrellas = (numpy.shape(u_kn))[0]
    N_k = nsamples*numpy.ones([numbrellas], int)
    
    # Histogram bins are indexed using the scheme:
    xmin = gridscale*(numpy.min(xrange[0][0])-1/2.0)
    xmax = gridscale*(numpy.max(xrange[0][1])+1/2.0)
    dx = (xmax-xmin)/nbinsperdim
    nbins = nbinsperdim**ndim
    bin_edges = [numpy.linspace(xmin,xmax,nbins+1)] # list of bin edges.
    bin_centers = numpy.zeros([nbins,ndim],numpy.float64)
    
    ibin = 0
    pmf_analytical = numpy.zeros([nbins],numpy.float64)
    minmu2 = 1000000
    zeroindex = 0
    # construct the bins and the pmf
    for i in range(nbins):
        xbin = xmin + dx * (i + 0.5)
        bin_centers[ibin,0] = xbin
        mu2 = xbin*xbin
        if (mu2 < minmu2):
            minmu2 = mu2
            zeroindex = ibin
            pmf_analytical[ibin] = K0*mu2/2.0
            ibin += 1
fzero = pmf_analytical[zeroindex]
pmf_analytical -= fzero
bin_n = -1*numpy.ones([numbrellas*nsamples], int)
# Determine indices of those within bounds.
within_bounds = (x_n[:,0] >= xmin) & (x_n[:,0] < xmax)
# Determine states for these.
bin_n[within_bounds] = numpy.floor((x_n[within_bounds,0]-xmin)/dx)
# Determine indices of bins that are not empty.
bin_counts = numpy.zeros([nbins], int)
for i in range(nbins):
  bin_counts[i] = (bin_n == i).sum()
# Compute PMF.
print("Solving for free energies of state to initialize PMF...")
mbar_options = dict()
mbar_options['verbose'] = True
pmf = PMF(u_kn,N_k,mbar_options=mbar_options)
print("Computing PMF ...")
histogram_parameters = dict()
histogram_parameters['bin_edges'] = bin_edges
pmf.generatePMF(u_n, x_n, histogram_parameters = histogram_parameters)
results = pmf.getPMF(bin_centers[:,0], uncertainties = 'from-specified', pmf_reference = 0.0)
f_i = results['f_i']
df_i = results['df_i']

# now KDE
kde_parameters = dict()
kde_parameters['bandwidth'] = 0.5*dx
pmf.generatePMF(u_n, x_n, pmf_type = 'kde', kde_parameters = kde_parameters)
results_kde = pmf.getPMF(bin_centers, uncertainties='from-specified', pmf_reference = 0.0)
f_ik = results_kde['f_i']

# Show free energy and uncertainty of each occupied bin relative to lowest free energy

print("1D PMF:")
print("%d counts out of %d counts not in any bin" % (numpy.sum(bin_n==-1),numbrellas*nsamples))
print("%8s %6s %8s %10s %10s %10s %10s %10s %10s %8s" % ('bin', 'x', 'N', 'f_hist', 'f_kde', 'true','err_h','err_kde','df','sigmas'))
for i in range(0,nbins):
  error = pmf_analytical[i]-f_i[i]
  if df_i[i] != 0:
    stdevs = numpy.abs(error)/df_i[i]
  else:
    stdevs = 0
  print('{:8d} {:6.2f} {:8d} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:8.2f}'.format(i, bin_centers[i,0], bin_counts[i], f_i[i], f_ik[i], pmf_analytical[i], error, pmf_analytical[i]-f_ik[i], df_i[i], stdevs))

print("============================================")
print("      Test 2: 2D PMF   ")
print("============================================")

xrange = [[-3,3],[-3,3]]
ndim = 2
nsamples = 300
u_kn, u_n, x_n, f_k_analytical = generate_pmf_data(K0 = K0, Ku = Ku, ndim=ndim, nbinsperdim = nbinsperdim, nsamples = nsamples, gridscale = gridscale, xrange=xrange)
numbrellas = (numpy.shape(u_kn))[0]
N_k = nsamples*numpy.ones([numbrellas], int)
print("Solving for free energies of state ...")
mbar = MBAR(u_kn, N_k)

# The dimensionless free energy is the integral of this, and can be computed as:
#   f(beta,K)           = - ln [ (2*numpy.pi/(Ko+Ku))^(d/2) exp[ -Ku*Ko mu' mu / 2(Ko +Ku)]
#   f(beta,K) - fzero   = -Ku*Ko / 2(Ko+Ku)  = 1/(1/(Ku/2) + 1/(K0/2))
# for computing harmonic samples

#Can compare the free energies computed with MBAR if desired with f_k_analytical

xmin = gridscale*(numpy.min(xrange[0][0])-1/2.0)
xmax = gridscale*(numpy.max(xrange[0][1])+1/2.0)
ymin = gridscale*(numpy.min(xrange[1][0])-1/2.0)
ymax = gridscale*(numpy.max(xrange[1][1])+1/2.0)
dx = (xmax-xmin)/nbinsperdim
dy = (ymax-ymin)/nbinsperdim
nbins = nbinsperdim**ndim
bin_centers = numpy.zeros([nbins,ndim],numpy.float64)

ibin = 0 
pmf_analytical = numpy.zeros([nbins],numpy.float64)
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
    pmf_analytical[ibin] = K0*mu2/2.0
    ibin += 1
fzero = pmf_analytical[zeroindex]
pmf_analytical -= fzero

Ntot = numpy.sum(N_k)
bin_n = -1*numpy.ones([Ntot,2],int)
# Determine indices of those within bounds.  Outside bounds stays as -1 in that direction.
within_boundsx = (x_n[:,0] >= xmin) & (x_n[:,0] < xmax)
within_boundsy = (x_n[:,1] >= ymin) & (x_n[:,1] < ymax)
# Determine states for these.
xgrid = (x_n[within_boundsx,0]-xmin)/dx
ygrid = (x_n[within_boundsy,1]-ymin)/dy
bin_n[within_boundsx,0] = xgrid
bin_n[within_boundsy,1] = ygrid

# Determine 2Dindices of bins that are not empty.
bin_counts = numpy.zeros(nbins, int)

for n in range(Ntot):
  b = bin_n[n]
  bin_label = b[0] + nbinsperdim*b[1]
  bin_counts[bin_label] += 1

# initialize PMF
print("Computing PMF ...")
pmf = PMF(u_kn, N_k)
# Compute PMF.          

#histogram_parameters['bin_n'] = bin_n # Indicates which state each sample comes from.  Each bin has 2D
histogram_parameters['bin_edges'] = [numpy.linspace(xmin,xmax,nbinsperdim+1),numpy.linspace(ymin,ymax,nbinsperdim+1)] # list of histogram edges.
pmf.generatePMF(u_n, x_n, pmf_type = 'histogram', histogram_parameters = histogram_parameters)
delta = 0.0001  # to break ties in things being too close.

results = pmf.getPMF(bin_centers+delta, uncertainties = 'from-specified', pmf_reference = [0,0])
f_i = results['f_i']
df_i = results['df_i']

kde_parameters['bandwidth'] = 0.5*dx
pmf.generatePMF(u_n, x_n, pmf_type = 'kde', kde_parameters = kde_parameters)
results_kde = pmf.getPMF(bin_centers, uncertainties='from-specified',pmf_reference = [0,0])

f_ik = results_kde['f_i']

# Show free energy and uncertainty of each occupied bin relative to lowest free energy
print("2D PMF:")
print("{:d} counts out of {:d} counts not in any bin".format(numpy.sum(numpy.any(bin_n==-1,axis=1)),Ntot))
print("{:8s} {:6s} {:6s} {:8s} {:10s} {:10s} {:10s} {:10s} {:8s}".format('bin', 'x', 'y', 'N', 'f', 'true','error','df','sigmas'))
for i in range(0,nbins):
  if df_i[i] == 0:
    stdevs = 0
  else:
    error = pmf_analytical[i]-f_i[i]
    stdevs = numpy.abs(error)/df_i[i]
  print('{:8d} {:6.2f} {:6.2f} {:8d} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:8.2f}'.format(i, bin_centers[i,0], bin_centers[i,1], bin_counts[i], f_i[i], f_ik[i], pmf_analytical[i], error, pmf_analytical[i]-f_ik[i], df_i[i], stdevs))





    xmin = test.O_k[refstate] - 1
    xmax = test.O_k[refstate] + 1
    within_bounds = (x_n >= xmin) & (x_n < xmax)
    bin_centers = dx*np.arange(np.int(xmin/dx), np.int(xmax/dx)) + dx/2
    bin_n = np.zeros(len(x_n), int)
    bin_n[within_bounds] = 1 + np.floor((x_n[within_bounds]-xmin)/dx)
    # 0 is reserved for samples outside the domain.  We will ignore this state
    range = np.max(bin_n)+1
    results = mbar.computePMF(u_kn[refstate, :], bin_n, range, uncertainties = 'from-specified', pmf_reference=1)
    results = pmf.getPMF()

    f_i = results['f_i']
    df_i = results['df_i']

    f0_i = 0.5*test.K_k[refstate]*(bin_centers-test.O_k[refstate])**2
    f_i, df_i = f_i[2:], df_i[2:]  # first state is ignored, second is zero, with zero uncertainty
    normf0_i = f0_i[1:] - f0_i[0]  # normalize to first state
    z = (f_i - normf0_i) / df_i
    assert_almost_equal(z / z_scale_factor, np.zeros(len(z)), decimal=0)
