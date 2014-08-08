##############################################################################
# pymbar: A Python Library for MBAR
#
# Copyright 2010-2014 University of Virginia, Memorial Sloan-Kettering Cancer Center
#
# Authors: Michael Shirts, John Chodera
# Contributors: Kyle Beauchamp
#
# pymbar is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with pymbar. If not, see <http://www.gnu.org/licenses/>.
##############################################################################

##############################################################################
# imports
##############################################################################

import itertools
import warnings

import numpy as np

##############################################################################
# functions / classes
##############################################################################


class TypeCastPerformanceWarning(RuntimeWarning):
    pass

def kln_to_kn(kln, N_k = None, cleanup = False):

    """ Convert KxKxN_max array to KxN max array

    if self.N is not initialized, it will be here.

    Parameters
    ----------
    u_kln : np.ndarray, float, shape=(KxLxN_max)
    N_k (optional) : np.array
        the N_k matrix from the previous formatting form
    cleanup (optional) : bool
        optional command to clean up, since u_kln can get very large

    Outputs
    -------
    u_kn: np.ndarray, float, shape=(LxN)
    """

    #print "warning: KxLxN_max arrays deprecated; convering into new preferred KxN shape"

    # rewrite into kn shape
    [K, L, N_max] = np.shape(kln)

    if N_k == None:
        # We assume that all N_k are N_max.
        # Not really an easier way to do this without being given the answer.
        N_k = N_max * np.ones([L], dtype=np.int32)
    N = np.sum(N_k)

    kn = np.zeros([L, N], dtype=np.float64)
    i = 0
    for k in range(K):  # loop through the old K; some might be zero
        for ik in range(N_k[k]):
            kn[:, i] = kln[k, :, ik]
            i += 1
    if cleanup:
        del(kln)  # very big, let's explicitly delete

    return kn

def kn_to_n(kn, N_k = None, cleanup = False):

    """ Convert KxN_max array to N array

    Parameters
    ----------
    u_kn: np.ndarray, float, shape=(KxN_max)
    N_k (optional) : np.array
        the N_k matrix from the previous formatting form
    cleanup (optional) : bool
        optional command to clean up, since u_kln can get very large

    Outputs
    -------
    u_n: np.ndarray, float, shape=(N)
    """

    #print "warning: KxN arrays deprecated; convering into new preferred N shape"
    # rewrite into kn shape

    # rewrite into kn shape
    [K, N_max] = np.shape(kn)

    if N_k == None:
        # We assume that all N_k are N_max.
        # Not really an easier way to do this without being given the answer.
        N_k = N_max*np.ones([K], dtype=np.int32)
    N = np.sum(N_k)

    n = np.zeros([N], dtype=np.float64)
    i = 0
    for k in range(K):  # loop through the old K; some might be zero
        for ik in range(N_k[k]):
            n[i] = kn[k, ik]
            i += 1
    if cleanup:
        del(kn)  # very big, let's explicitly delete
    return n

def ensure_type(val, dtype, ndim, name, length=None, can_be_none=False, shape=None,
                warn_on_cast=True, add_newaxis_on_deficient_ndim=False):
    """Typecheck the size, shape and dtype of a numpy array, with optional
    casting.

    Parameters
    ----------
    val : {np.ndaraay, None}
        The array to check
    dtype : {nd.dtype, str}
        The dtype you'd like the array to have
    ndim : int
        The number of dimensions you'd like the array to have
    name : str
        name of the array. This is used when throwing exceptions, so that
        we can describe to the user which array is messed up.
    length : int, optional
        How long should the array be?
    can_be_none : bool
        Is ``val == None`` acceptable?
    shape : tuple, optional
        What should be shape of the array be? If the provided tuple has
        Nones in it, those will be semantically interpreted as matching
        any length in that dimension. So, for example, using the shape
        spec ``(None, None, 3)`` will ensure that the last dimension is of
        length three without constraining the first two dimensions
    warn_on_cast : bool, default=True
        Raise a warning when the dtypes don't match and a cast is done.
    add_newaxis_on_deficient_ndim : bool, default=True
        Add a new axis to the beginining of the array if the number of
        dimensions is deficient by one compared to your specification. For
        instance, if you're trying to get out an array of ``ndim == 3``,
        but the user provides an array of ``shape == (10, 10)``, a new axis will
        be created with length 1 in front, so that the return value is of
        shape ``(1, 10, 10)``.

    Notes
    -----
    The returned value will always be C-contiguous.

    Returns
    -------
    typechecked_val : np.ndarray, None
        If `val=None` and `can_be_none=True`, then this will return None.
        Otherwise, it will return val (or a copy of val). If the dtype wasn't right,
        it'll be casted to the right shape. If the array was not C-contiguous, it'll
        be copied as well.

    """
    if can_be_none and val is None:
        return None

    if not isinstance(val, np.ndarray):
        # special case: if the user is looking for a 1d array, and
        # they request newaxis upconversion, and provided a scalar
        # then we should reshape the scalar to be a 1d length-1 array
        if add_newaxis_on_deficient_ndim and ndim == 1 and np.isscalar(val):
            val = np.array([val])
        else:
            raise TypeError(("%s must be numpy array. "
                             " You supplied type %s" % (name, type(val))))

    if warn_on_cast and val.dtype != dtype:
        warnings.warn("Casting %s dtype=%s to %s " % (name, val.dtype, dtype),
                      TypeCastPerformanceWarning)

    if not val.ndim == ndim:
        if add_newaxis_on_deficient_ndim and val.ndim + 1 == ndim:
            val = val[np.newaxis, ...]
        else:
            raise ValueError(("%s must be ndim %s. "
                              "You supplied %s" % (name, ndim, val.ndim)))

    val = np.ascontiguousarray(val, dtype=dtype)

    if length is not None and len(val) != length:
        raise ValueError(("%s must be length %s. "
                          "You supplied %s" % (name, length, len(val))))

    if shape is not None:
        # the shape specified given by the user can look like (None, None 3)
        # which indicates that ANY length is accepted in dimension 0 or
        # dimension 1
        sentenel = object()
        error = ValueError(("%s must be shape %s. You supplied  "
                            "%s" % (name, str(shape).replace('None', 'Any'), val.shape)))
        for a, b in itertools.izip_longest(val.shape, shape, fillvalue=sentenel):
            if a is sentenel or b is sentenel:
                # if the sentenel was reached, it means that the ndim didn't
                # match or something. this really shouldn't happen
                raise error
            if b is None:
                # if the user's shape spec has a None in it, it matches anything
                continue
            if a != b:
                # check for equality
                raise error

    return val


def _logsum(a_n):
    """Compute the log of a sum of exponentiated terms exp(a_n) in a numerically-stable manner.

    Parameters
    ----------
    a_n : np.ndarray, shape=(n_samples)
        a_n[n] is the nth exponential argument

    Returns
    -------
    a_n : np.ndarray, shape=(n_samples)
        a_n[n] is the nth exponential argument

    Notes
    -----

    _logsum a_n = max_arg + \log \sum_{n=1}^N \exp[a_n - max_arg]

    where max_arg = max_n a_n.  This is mathematically (but not numerically) equivalent to

    _logsum a_n = \log \sum_{n=1}^N \exp[a_n]



    Example
    -------
    >>> a_n = np.array([0.0, 1.0, 1.2], np.float64)
    >>> print '%.3e' % _logsum(a_n)
    1.951e+00
    """

    # Compute the maximum argument.
    max_log_term = np.max(a_n)

    # Compute the reduced terms.
    terms = np.exp(a_n - max_log_term)

    # Compute the log sum.
    log_sum = np.log(np.sum(terms)) + max_log_term

    return log_sum

#=============================================================================================
# Exception classes
#=============================================================================================


class ParameterError(Exception):

    """
    An error in the input parameters has been detected.

    """
    pass


class ConvergenceError(Exception):

    """
    Convergence could not be achieved.

    """
    pass


class BoundsError(Exception):

    """
    Could not determine bounds on free energy

    """
    pass
