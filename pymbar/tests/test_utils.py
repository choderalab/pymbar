import numpy as np
import pymbar
from pymbar.utils_for_testing import eq
import scipy.misc
from nose import SkipTest

def test_logsumexp():
    a = np.random.normal(size=(200, 500, 5))

    for axis in range(a.ndim):
        ans_ne = pymbar.utils.logsumexp(a, axis=axis)
        ans_no_ne = pymbar.utils.logsumexp(a, axis=axis, use_numexpr=False)
        ans_scipy = scipy.misc.logsumexp(a, axis=axis)
        eq(ans_ne, ans_no_ne)
        eq(ans_ne, ans_scipy)

def test_logsumexp_b():
    a = np.random.normal(size=(200, 500, 5))
    b = np.random.normal(size=(200, 500, 5)) ** 2.

    for axis in range(a.ndim):
        ans_ne = pymbar.utils.logsumexp(a, b=b, axis=axis)
        ans_no_ne = pymbar.utils.logsumexp(a, b=b, axis=axis, use_numexpr=False)
        ans_scipy = scipy.misc.logsumexp(a, b=b, axis=axis)
        eq(ans_ne, ans_no_ne)
        eq(ans_ne, ans_scipy)

def test_logsum():
    u = np.random.normal(size=(200))
    y1 = pymbar.utils.logsumexp(u)
    y2 = pymbar.utils._logsum(u)
    eq(y1, y2, decimal=12)
