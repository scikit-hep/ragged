# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/creation_functions.html
"""

from __future__ import annotations

import awkward as ak

from ._obj import array
from ._typing import (
    Device,
    Dtype,
    NestedSequence,
    SupportsBufferProtocol,
    SupportsDLPack,
)


def arange(
    start: int | float,
    /,
    stop: None | int | float = None,
    step: int | float = 1,
    *,
    dtype: None | Dtype = None,
    device: None | Device = None,
) -> array:
    """
    Returns evenly spaced values within the half-open interval `[start, stop)`
    as a one-dimensional array.

    Args:
        start: If `stop` is specified, the start of interval (inclusive);
            otherwise, the end of the interval (exclusive). If `stop` is not
            specified, the default starting value is 0.
        stop: The end of the interval.
        step: The distance between two adjacent elements `(out[i+1] - out[i])`.
            Must not be 0; may be negative, this results in an empty array if
            `stop >= start`.
        dtype: Output array data type. If dtype is `None`, the output array
            data type is inferred from `start`, `stop` and `step`. If those are
            all integers, the output array dtype is `np.int64`; if one or more
            have type `float`, then the output array dtype is `np.float64`.
        device: Device on which to place the created array.

    Returns:
        A one-dimensional array containing evenly spaced values. The length of
        the output array is `ceil((stop-start)/step)` if `stop - start` and
        `step` have the same sign, and length 0 otherwise.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.arange.html
    """

    assert start, "TODO"
    assert stop, "TODO"
    assert step, "TODO"
    assert dtype, "TODO"
    assert device, "TODO"
    assert False, "TODO"


def asarray(
    obj: (
        array
        | ak.Array
        | bool
        | int
        | float
        | complex
        | NestedSequence[bool | int | float | complex]
        | SupportsBufferProtocol
        | SupportsDLPack
    ),
    dtype: None | Dtype | type | str = None,
    device: None | Device = None,
    copy: None | bool = None,
) -> array:
    """
    Convert the input to an array.

    Args:
        obj: Object to be converted to an array. May be a Python scalar, a
            (possibly nested) sequence of Python scalars, or an object
            supporting the Python buffer protocol or DLPack.
        dtype: Output array data type. If `dtype` is `None`, the output array
            data type is inferred from the data type(s) in `obj`. If all input
            values are Python scalars, then, in order of precedence,
                - if all values are of type `bool`, the output data type is
                  `bool`.
                - if all values are of type `int` or are a mixture of `bool`
                  and `int`, the output data type is `np.int64`.
                - if one or more values are `complex` numbers, the output data
                  type is `np.complex128`.
                - if one or more values are `float`s, the output data type is
                  `np.float64`.
        device: Device on which to place the created array. If device is `None`
            and `obj` is an array, the output array device is inferred from
            `obj`. If `"cpu"`, the array is backed by NumPy and resides in main
            memory; if `"cuda"`, the array is backed by CuPy and resides in
            CUDA global memory.
        copy: Boolean indicating whether or not to copy the input. If `True`,
            this function always copies. If `False`, the function never copies
            for input which supports the buffer protocol and raises a
            ValueError in case a copy would be necessary. If `None`, the
            function reuses the existing memory buffer if possible and copies
            otherwise.
    """
    return array(obj, dtype=dtype, device=device, copy=copy)
