from warnings import warn
from functools import wraps

_docstring_warning = """
Warning
-------
This function/method name is deprecated, and provided for backward-compatibility only.
It may be removed in future versions. Please update your code to use `{}` instead.
"""

_deprecated_names = {}


def _deprecate(fn, old_name):
    """
    Helper to deprecate old names. It will add a deprecation warning and update the
    docstring with more information.

    Parameters
    ----------
    fn : callable
        The new function that will replace the old one
    old_name : str
        Original name to be deprecated

    >>> OldName = _deprecate(new_name, "OldName")
    """
    _deprecated_names[old_name] = fn.__name__

    @wraps(fn)
    def deprecated(*args, **kwargs):
        warn(
            f"`{old_name}` will be deprecated in pymbar 4+. Please update your code to use `{fn.__name__}` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return fn(*args, **kwargs)

    fn_doc = fn.__doc__ or ""
    deprecated.__doc__ = fn.__doc__ + _docstring_warning.format(fn.__name__)
    return deprecated
