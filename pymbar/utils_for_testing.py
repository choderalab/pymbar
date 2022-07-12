##############################################################################
# pymbar: A Python Library for MBAR
#
# Copyright 2016-2017 University of Colorado Boulder
# Copyright 2010-2017 Memorial Sloan-Kettering Cancer Center
# Portions of this software are Copyright 2010-2016 University of Virginia
#
# Authors: Michael Shirts, John Chodera
# Contributors: Kyle Beauchamp, Levi Naden
#
# pymbar is free software: you can redistribute it and/or modify
# it under the terms of the MIT License
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# MIT License for more details.
#
# You should have received a copy of the MIT License along with pymbar.
##############################################################################

import os
import functools
import numpy as np
from numpy.testing import (assert_allclose, assert_almost_equal,
                           assert_approx_equal, assert_array_almost_equal, assert_array_almost_equal_nulp,
                           assert_array_equal, assert_array_less, assert_array_max_ulp, assert_equal,
                           assert_raises, assert_string_equal, assert_warns)
from pkg_resources import resource_filename
import warnings
import contextlib


# if the system doesn't have scipy, we'd like
# this package to still work:
# we'll just redefine isspmatrix as a function that always returns
# false
try:
    from scipy.sparse import isspmatrix
except ImportError:
    isspmatrix = lambda x: False


__all__ = ['assert_allclose', 'assert_almost_equal', 'assert_approx_equal',
           'assert_array_almost_equal', 'assert_array_almost_equal_nulp',
           'assert_array_equal', 'assert_array_less', 'assert_array_max_ulp',
           'assert_equal', 'assert_raises', 'assert_string_equal', 'assert_warns',
           'get_fn', 'eq', 'assert_dict_equal', 'assert_sparse_matrix_equal',
           'expected_failure', 'skip',
           'suppress_derivative_warnings_for_tests']

##############################################################################
# functions
##############################################################################


@contextlib.contextmanager
def suppress_derivative_warnings_for_tests():
    """
    Suppress specific warnings and then reset when done, used as a with suppress_warnings():
    """
    # Supress the warnings when Jacobian and Hessian information is not used in a specific solver
    warnings.filterwarnings('ignore', '.*does not use the Jacobian.*')
    warnings.filterwarnings('ignore', '.*does not use Hessian.*')
    warnings.filterwarnings('ignore', '.*parameter will change to the default of machine precision.*')
    yield
    # Clear warning filters
    warnings.resetwarnings()


def get_fn(name):
    """Get the full path to one of the reference files shipped for testing

    In the source distribution, these files are in ``MDTraj/testing/reference``,
    but on installation, they're moved to somewhere in the user's python
    site-packages directory.

    Parameters
    ----------
    name : str
        Name of the file to load (with respect to the reference/ folder).

    """

    fn = resource_filename('mdtraj', os.path.join('testing/reference', name))

    if not os.path.exists(fn):
        raise ValueError('Sorry! %s does not exists. If you just '
                         'added it, you\'ll have to re install' % fn)

    return fn


def eq(o1, o2, decimal=6, err_msg=''):
    """Convenience function for asserting that two objects are equal to one another

    If the objects are both arrays or sparse matrices, this method will
    dispatch to an appropriate handler, which makes it a little bit more
    useful than just calling ``assert o1 == o2`` (which wont work for numpy
    arrays -- it returns an array of bools, not a single True or False)

    Parameters
    ----------
    o1 : object
        The first object
    o2 : object
        The second object
    decimal : int
        If the two objects are floats or arrays of floats, they'll be checked for
        equality up to this decimal place.
    err_msg : str
        Custom error message

    Returns
    -------
    passed : bool
        True if the tests pass. If the tests doesn't pass, since the AssertionError will be raised

    Raises
    ------
    AssertionError
        If the tests fail
    """
    assert (type(o1) is type(o2)), 'o1 and o2 not the same type: %s %s' % (type(o1), type(o2))

    if isinstance(o1, dict):
        assert_dict_equal(o1, o1, decimal)
    elif isinstance(o1, float):
        np.testing.assert_almost_equal(o1, o2, decimal)
    elif isspmatrix(o1):
        assert_sparse_matrix_equal(o1, o1, decimal)
    elif isinstance(o1, np.ndarray):
        if o1.dtype.kind == 'f' or o2.dtype.kind == 'f':
            # compare floats for almost equality
            assert_array_almost_equal(o1, o2, decimal, err_msg=err_msg)
        elif o1.dtype.type == np.core.records.record:
            # if its a record array, we need to comparse each term
            assert o1.dtype.names == o2.dtype.names
            for name in o1.dtype.names:
                eq(o1[name], o2[name], decimal=decimal, err_msg=err_msg)
        else:
            # compare everything else (ints, bools) for absolute equality
            assert_array_equal(o1, o2, err_msg=err_msg)
    # probably these are other specialized types
    # that need a special check?
    else:
        assert o1==o2

    return True


def assert_dict_equal(t1, t2, decimal=6):
    """
    Assert two dicts are equal.
    This method should actually
    work for any dict of numpy arrays/objects
    """

    # make sure the keys are the same
    assert t1.keys()==t2.keys()

    for key, val in t1.iteritems():
        # compare numpy arrays using numpy.testing
        if isinstance(val, np.ndarray):
            if val.dtype.kind == 'f':
                # compare floats for almost equality
                assert_array_almost_equal(val, t2[key], decimal)
            else:
                # compare everything else (ints, bools) for absolute equality
                assert_array_equal(val, t2[key])
        else:
            assert val==t2[key]


def assert_sparse_matrix_equal(m1, m2, decimal=6):
    """Assert two scipy.sparse matrices are equal."""
    # both are sparse matricies
    assert isspmatrix(m1)
    assert isspmatrix(m1)

    # make sure they have the same format
    assert m1.format==m2.format

    # even though its called assert_array_almost_equal, it will
    # work for scalars
    assert_array_almost_equal((m1 - m2).sum(), 0, decimal=decimal)
