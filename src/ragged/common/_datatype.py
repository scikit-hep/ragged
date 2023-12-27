# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/data_type_functions.html
"""

from __future__ import annotations

from ._obj import array
from ._typing import (
    Dtype,
)


def astype(x: array, dtype: Dtype, /, *, copy: bool = True) -> array:
    """
    Copies an array to a specified data type irrespective of type promotion rules.

    Args:
        x: Array to cast.
        dtype: Desired data type.
        copy: Specifies whether to copy an array when the specified `dtype`
            matches the data type of the input array `x`. If `True`, a newly
            allocated array is always returned. If `False` and the specified
            `dtype` matches the data type of the input array, the input array
            is returned; otherwise, a newly allocated array is returned.

    Returns:
        An array having the specified data type. The returned array has the
        same `shape` as `x`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.astype.html
    """

    assert x, "TODO"
    assert dtype, "TODO"
    assert copy, "TODO"
    assert False, "TODO"
