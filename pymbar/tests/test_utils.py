import numpy as np
import pytest
import pymbar
from pymbar.utils_for_testing import assert_equal, assert_almost_equal
from pymbar.utils import ParameterError, ensure_type, TypeCastPerformanceWarning

try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp


def test_logsumexp():
    a = np.random.normal(size=(200, 500, 5))

    for axis in range(a.ndim):
        ans_ne = pymbar.utils.logsumexp(a, axis=axis)
        ans_no_ne = pymbar.utils.logsumexp(a, axis=axis, use_numexpr=False)
        ans_scipy = logsumexp(a, axis=axis)
        assert_almost_equal(ans_ne, ans_no_ne)
        assert_almost_equal(ans_ne, ans_scipy)


def test_logsumexp_single_infinite():
    a = np.inf
    ans = pymbar.utils.logsumexp(a)
    ans_scipy = logsumexp(a)
    assert_equal(ans, ans_scipy)


def test_logsumexp_b():
    a = np.random.normal(size=(200, 500, 5))
    b = np.random.normal(size=(200, 500, 5)) ** 2.0

    for axis in range(a.ndim):
        ans_ne = pymbar.utils.logsumexp(a, b=b, axis=axis)
        ans_no_ne = pymbar.utils.logsumexp(a, b=b, axis=axis, use_numexpr=False)
        ans_scipy = logsumexp(a, b=b, axis=axis)
        assert_almost_equal(ans_ne, ans_no_ne)
        assert_almost_equal(ans_ne, ans_scipy)


def test_logsum():
    u = np.random.normal(size=(200))
    y1 = pymbar.utils.logsumexp(u)
    y2 = pymbar.utils._logsum(u)
    assert_almost_equal(y1, y2, decimal=12)


@pytest.mark.xfail(raises=ParameterError)
def test_non_normalized_w_badrow():
    w = np.array([[0.5, 0.5, 0.75, 0.25]])
    n_k = np.array([1, 1])
    pymbar.utils.check_w_normalized(w, n_k)


@pytest.mark.xfail(raises=ParameterError)
def test_non_normalized_w_badcol():
    w = np.array([[0.5, 0.5], [0.5, 0.5]])
    n_k = np.array([1, 0])
    pymbar.utils.check_w_normalized(w, n_k)


@pytest.mark.parametrize(
    "fn_args_and_kwargs,expected,warn",
    [
        (
            {"val": None, "dtype": int, "ndim": 1, "name": "thetest", "can_be_none": True},
            None,
            None,
        ),
        (
            {
                "val": 0,
                "dtype": int,
                "ndim": 1,
                "name": "thetest",
                "add_newaxis_on_deficient_ndim": True,
            },
            np.array([0]),
            None,
        ),
        pytest.param(
            {
                "val": 0,
                "dtype": int,
                "ndim": 1,
                "name": "thetest",
                "add_newaxis_on_deficient_ndim": False,
            },
            "Should Fail",
            None,
            marks=pytest.mark.xfail,
        ),
        pytest.param(
            {
                "val": [],
                "dtype": int,
                "ndim": 1,
                "name": "thetest",
                "add_newaxis_on_deficient_ndim": True,
            },
            "Should Fail",
            None,
            marks=pytest.mark.xfail,
        ),
        (
            {
                "val": np.array([1.0]),
                "dtype": int,
                "ndim": 1,
                "name": "thetest",
                "warn_on_cast": True,
            },
            np.array([1]),
            TypeCastPerformanceWarning,
        ),
        (
            {
                "val": np.array([1]),
                "dtype": int,
                "ndim": 2,
                "name": "thetest",
                "add_newaxis_on_deficient_ndim": True,
            },
            np.array([[1]]),
            None,
        ),
        pytest.param(
            {
                "val": np.array([1]),
                "dtype": int,
                "ndim": 3,
                "name": "thetest",
                "add_newaxis_on_deficient_ndim": True,
            },
            "Should Fail",
            None,
            marks=pytest.mark.xfail,
        ),
        pytest.param(
            {"val": np.array([1, 2, 3]), "dtype": int, "ndim": 1, "name": "thetest", "length": 4},
            "Should Fail",
            None,
            marks=pytest.mark.xfail,
        ),
        (
            {
                "val": np.array([[1, 2, 3], [4, 5, 6]]),
                "dtype": int,
                "ndim": 2,
                "name": "thetest",
                "shape": (2, 3),
            },
            np.array([[1, 2, 3], [4, 5, 6]]),
            None,
        ),
        (
            {
                "val": np.array([[1, 2, 3], [4, 5, 6]]),
                "dtype": int,
                "ndim": 2,
                "name": "thetest",
                "shape": (None, 3),
            },
            np.array([[1, 2, 3], [4, 5, 6]]),
            None,
        ),
        pytest.param(
            {
                "val": np.array([[1, 2, 3], [4, 5, 6]]),
                "dtype": int,
                "ndim": 2,
                "name": "thetest",
                "shape": (2,),
            },
            "Should Fail",
            None,
            marks=pytest.mark.xfail,
        ),
        pytest.param(
            {
                "val": np.array([[1, 2, 3], [4, 5, 6]]),
                "dtype": int,
                "ndim": 2,
                "name": "thetest",
                "shape": (3, 1),
            },
            "Should Fail",
            None,
            marks=pytest.mark.xfail,
        ),
    ],
)
def test_type_ensure(fn_args_and_kwargs, expected, warn):
    if warn is not None:
        with pytest.warns(warn):
            ret = ensure_type(**fn_args_and_kwargs)
    else:
        ret = ensure_type(**fn_args_and_kwargs)
    if isinstance(ret, np.ndarray):
        assert np.allclose(ret, expected)
        assert ret.shape == expected.shape
    else:
        assert ret == expected


@pytest.mark.parametrize("n_k", [None, np.array([3] * 3)])
def test_kln_to_nk_to_k(n_k):
    # 3 samples per state
    # 3 states
    # Samples energies are (state index + 1)*(relative state index)
    u_kln = np.array(
        [
            # k=0
            [[0, 0, 0], [1, 1, 1], [2, 2, 2]],  # l=0: n=1*0  # l=1: n=1*1  # l=2: n=1*2
            # k=1
            [[-2, -2, -2], [0, 0, 0], [2, 2, 2]],  # l=0: n=2*-1  # l=1: n=2*0  # l=2: n=2*1
            # k=2
            [[-6, -6, -6], [-3, -3, -3], [0, 0, 0]],  # l=0: n=3*-2  # l=1: n=3*-1  # l=2: n=3*0
        ]
    )
    u_kn = np.array(
        [  # n 1 2 3...
            [0, 0, 0, -2, -2, -2, -6, -6, -6],  # k = 0
            [1, 1, 1, 0, 0, 0, -3, -3, -3],  # k = 1
            [2, 2, 2, 2, 2, 2, 0, 0, 0],  # k = 3
        ]
    )
    u_n = np.array(
        [0, 0, 0, -2, -2, -2, -6, -6, -6, 1, 1, 1, 0, 0, 0, -3, -3, -3, 2, 2, 2, 2, 2, 2, 0, 0, 0]
    )
    u_kn_out = pymbar.utils.kln_to_kn(u_kln, N_k=n_k, cleanup=True)
    assert np.allclose(u_kn, u_kn_out)
    if n_k is not None:
        # Because we're testing both None _and_ set N_k, we have to assume the u_kn we fed in was reset for
        # this second test. If we didn't, then the behavior for N_k=None is ambiguous or assumes things about
        # a state (programming/object) it has no knowledge of.
        n_k = np.array([9] * 3)
    u_n_out = pymbar.utils.kn_to_n(u_kn, N_k=n_k, cleanup=True)
    assert np.allclose(u_n, u_n_out)
