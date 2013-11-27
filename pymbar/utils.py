# Copyright 2013 mdtraj developers
#
# This file is part of mdtraj
#
# mdtraj is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# mdtraj is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# mdtraj. If not, see http://www.gnu.org/licenses/.

##############################################################################
# imports
##############################################################################

import itertools
import warnings
from pkg_resources import resource_filename
import os

import numpy as np
import pandas as pd

import logging
logger = logging.getLogger(__name__)

##############################################################################
# functions / classes
##############################################################################


class TypeCastPerformanceWarning(RuntimeWarning):
    pass


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


def convert_ukn_to_uijn(u_kn, N_k):
    """Convert from 2D representation to 3D representation.

    Parameters
    ----------
    u_kn : np.ndarray, shape=(n_states, n_samples)
        Reduced potentials evaluated in each state for each sample x_n

    Returns
    -------
    u_ijn : np.ndarray, shape=(n_states, n_states, n_max)
        The reduced potential evaluated in state i, for a sample taken 
        from state j, with sample index n.

    Notes
    -----
    This is useful for converting data from pymbar 1.0 to pymbar 2.0
    formats.  NOTE: CURRENTLY SLOW.
    """
    N_k = ensure_type(N_k, dtype='int', ndim=1, name="N_k")
    
    n_states = len(N_k)
    N_max = N_k.max()
    n_samples = N_k.sum()
    
    u_kn = ensure_type(u_kn, dtype='float', ndim=2, name="u_kn", shape=(n_states, n_samples))
    
    u_ijn = np.zeros((n_states, n_states, N_max))
    
    #Create mapping from jk to j, k
    states = np.repeat(np.arange(n_states), N_k)
    frames = np.concatenate([np.arange(N_k[i]) for i in range(n_states)])
    
    for j in range(n_states):
        for ik in range(n_samples):
            i, k = states[ik], frames[ik]
            u_ijn[i, j, k] = u_kn[j, ik]
    
    return u_ijn
    
def convert_uijn_to_ukn(u_ijn, N_k):
    """Convert from 2D representation to 3D representation.

    Parameters
    ----------
    u_ijn : np.ndarray, shape=(n_states, n_states, n_max)
        The reduced potential evaluated in state i, for a sample taken 
        from state j, with sample index n.    
    N_k : np.ndarray, shape=(n_states)
        Number of samples from each state

    Returns
    -------
    u_kn : np.ndarray, shape=(n_states, n_samples)
        Reduced potentials evaluated in each state for each sample x_n


    Notes
    -----
    This is useful for converting data from pymbar 1.0 to pymbar 2.0
    formats.  NOTE: CURRENTLY SLOW.
    """
    N_k = ensure_type(N_k, dtype='int', ndim=1, name="N_k")
    
    n_states = len(N_k)
    N_max = N_k.max()
    n_samples = N_k.sum()
    
    u_ijn = ensure_type(u_ijn, dtype='float', ndim=3, name="u_ijn", shape=(n_states, n_states, N_max))
    
    u_kn = np.zeros((n_states, n_samples))
    
    #Create mapping from jk to j, k
    states = np.repeat(np.arange(n_states), N_k)
    frames = np.concatenate([np.arange(N_k[i]) for i in range(n_states)])
    
    for j in range(n_states):
        for ik in range(n_samples):
            i, k = states[ik], frames[ik]
            u_kn[j, ik] = u_ijn[i, j, k]
    
    return u_kn
    
def convert_Akn_to_An(A_kn, N_k):
    """Convert from 2D representation to 1D representation of a single observable.

    Parameters
    ----------
    A_kn : np.ndarray, shape=(n_states, n_max)
        The reduced potential evaluated in state i, for a sample taken 
        from state j, with sample index n.    
    N_k : np.ndarray, shape=(n_states)
        Number of samples from each state

    Returns
    -------
    A_n : np.ndarray, shape=(n_samples)
        Observable reshaped


    Notes
    -----
    This is useful for converting data from pymbar 1.0 to pymbar 2.0
    formats.  NOTE: CURRENTLY SLOW.
    """
    N_k = ensure_type(N_k, dtype='int', ndim=1, name="N_k")
    
    n_states = len(N_k)
    N_max = N_k.max()
    n_samples = N_k.sum()
    
    A_kn = ensure_type(A_kn, dtype='float', ndim=2, name="A_kn", shape=(n_states, N_max))
    
    A_k = np.zeros((n_samples))
    
    #Create mapping from jk to j, k
    states = np.repeat(np.arange(n_states), N_k)
    frames = np.concatenate([np.arange(N_k[i]) for i in range(n_states)])
    
    for ik in range(n_samples):
        i, k = states[ik], frames[ik]
        A_k[ik] = A_kn[i, k]
    
    return A_k
    
def convert_An_to_Akn(A_n, N_k):
    """Convert from 1D representation to 2D representation of a single observable.

    Parameters
    ----------
    A_n : np.ndarray, shape=(n_samples)
        Observable reshaped    
    N_k : np.ndarray, shape=(n_states)
        Number of samples from each state

    Returns
    -------
    A_kn : np.ndarray, shape=(n_states, n_max)
        The reduced potential evaluated in state i, for a sample taken 
        from state j, with sample index n.    


    Notes
    -----
    This is useful for converting data from pymbar 1.0 to pymbar 2.0
    formats.  NOTE: CURRENTLY SLOW.
    """
    N_k = ensure_type(N_k, dtype='int', ndim=1, name="N_k")
    
    n_states = len(N_k)
    N_max = N_k.max()
    n_samples = N_k.sum()
    
    A_n = ensure_type(A_n, dtype='float', ndim=1, name="A_n", shape=(n_samples,))
    
    A_kn = np.zeros((n_states, N_max))
    
    #Create mapping from jk to j, k
    states = np.repeat(np.arange(n_states), N_k)
    frames = np.concatenate([np.arange(N_k[i]) for i in range(n_states)])
    
    for ik in range(n_samples):
        i, k = states[ik], frames[ik]
        A_kn[i, k] = A_n[ik]
    
    return A_kn       

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

def logsumexp(arr, axis=0):
    """Computes the sum of arr assuming arr is in the log domain.

    Returns log(sum(exp(arr))) while minimizing the possibility of
    over/underflow.
    
    Notes
    -----
    Backported from sklearn.utils.extmath.

    Examples
    --------

    >>> import numpy as np
    >>> from pymbar.utils import logsumexp
    >>> a = np.arange(10)
    >>> np.log(np.sum(np.exp(a)))
    9.4586297444267107
    >>> logsumexp(a)
    9.4586297444267107
    """
    arr = np.rollaxis(arr, axis)
    # Use the max to normalize, as with the log this is what accumulates
    # the less errors
    vmax = arr.max(axis=0)
    out = np.log(np.sum(np.exp(arr - vmax), axis=0))
    out += vmax
    return out

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

def validate_weight_matrix(W, N_k, tolerance=1E-4):
    """Check that row and column sums are 1; else raise ParameterError."""
    N, K = W.shape
    column_sums = np.sum(W, axis=0)
    badcolumns = (np.abs(column_sums - 1) > tolerance)
    if np.any(badcolumns):
        which_badcolumns = np.arange(K)[badcolumns]
        firstbad = which_badcolumns[0]
        raise ParameterError(
            'Warning: Should have \sum_n W_nk = 1.  Actual column sum for state %d was %f. %d other columns have similar problems' %
                             (firstbad, column_sums[firstbad], np.sum(badcolumns)))

    row_sums = np.sum(W * N_k, axis=1)
    badrows = (np.abs(row_sums - 1) > tolerance)
    if np.any(badrows):
        which_badrows = np.arange(N)[badrows]
        firstbad = which_badrows[0]
        raise ParameterError(
            'Warning: Should have \sum_k N_k W_nk = 1.  Actual row sum for sample %d was %f. %d other rows have similar problems' %
                             (firstbad, row_sums[firstbad], np.sum(badrows)))


def deprecated(replacement=None):
    """A decorator which can be used to mark functions as deprecated.
    replacement is a callable that will be called with the same args
    as the decorated function.
    
    Notes
    -----
    
    Adapted from http://code.activestate.com/recipes/577819-deprecated-decorator/

    >>> @deprecated()
    ... def foo(x):
    ...     return x
    ...
    >>> ret = foo(1)
    DeprecationWarning: foo is deprecated
    >>> ret
    1
    >>>
    >>>
    >>> def newfun(x):
    ...     return 0
    ...
    >>> @deprecated(newfun)
    ... def foo(x):
    ...     return x
    ...
    >>> ret = foo(1)
    DeprecationWarning: foo is deprecated; use newfun instead
    >>> ret
    0
    >>>
    """
    def outer(oldfun):
        def inner(*args, **kwargs):
            msg = "%s is deprecated" % oldfun.__name__
            if replacement is not None:
                msg += "; use %s instead" % (replacement.__name__)
            logger.warn(msg)
            if replacement is not None:
                return replacement(*args, **kwargs)
            else:
                return oldfun(*args, **kwargs)
        return inner
    return outer


def get_data_filename(relative_path):
    """Get the full path to one of the reference files shipped for testing

    Parameters
    ----------
    name : str
        Name of the file to load (with respect to the pymbar folder).

    """

    fn = resource_filename('pymbar', relative_path)

    if not os.path.exists(fn):
        raise ValueError("Sorry! %s does not exist. If you just added it, you'll have to re-install" % fn)

    return fn
