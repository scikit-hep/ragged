# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/creation_functions.html
"""

from __future__ import annotations

import awkward as ak

from ._spec_array_object import array
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

    Returns:
        An array containing the data from `obj`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.asarray.html
    """
    return array(obj, dtype=dtype, device=device, copy=copy)


def empty(
    shape: int | tuple[int, ...],
    *,
    dtype: None | Dtype = None,
    device: None | Device = None,
) -> array:
    """
    Returns an uninitialized array having a specified shape.

    Args:
        shape: Output array shape.
        dtype: Output array data type. If `dtype` is `None`, the output array
            data type is `np.float64`.
        device: Device on which to place the created array.

    Returns:
        An array containing uninitialized data.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.empty.html
    """

    assert shape, "TODO"
    assert dtype, "TODO"
    assert device, "TODO"
    assert False, "TODO"


def empty_like(
    x: array, /, *, dtype: None | Dtype = None, device: None | Device = None
) -> array:
    """
    Returns an uninitialized array with the same shape as an input array x.

    Args:
        x: Input array from which to derive the output array shape.
        dtype: Output array data type. If `dtype` is `None`, the output array
            data type is inferred from `x`.
        device: Device on which to place the created array. If `device` is
            `None`, output array device is inferred from `x`.

    Returns:
        An array having the same shape as `x` and containing uninitialized data.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.empty_like.html
    """

    assert x, "TODO"
    assert dtype, "TODO"
    assert device, "TODO"
    assert False, "TODO"


def eye(
    n_rows: int,
    n_cols: None | int = None,
    /,
    *,
    k: int = 0,
    dtype: None | Dtype = None,
    device: None | Device = None,
) -> array:
    """
    Returns a two-dimensional array with ones on the kth diagonal and zeros elsewhere.

    Args:
        n_rows: Number of rows in the output array.
        n_cols: Number of columns in the output array. If `None`, the default
            number of columns in the output array is equal to `n_rows`.
        k: Index of the diagonal. A positive value refers to an upper diagonal,
            a negative value to a lower diagonal, and 0 to the main diagonal.
        dtype: Output array data type. If `dtype` is `None`, the output array
            data type is `np.float64`.
        device: Device on which to place the created array.

    Returns:
        An array where all elements are equal to zero, except for the kth
        diagonal, whose values are equal to one.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.eye.html
    """

    assert n_rows, "TODO"
    assert n_cols, "TODO"
    assert k, "TODO"
    assert dtype, "TODO"
    assert device, "TODO"
    assert False, "TODO"


def from_dlpack(x: object, /) -> array:
    """
    Returns a new array containing the data from another (array) object with a `__dlpack__` method.

    Args:
        x: Input (array) object.

    Returns:
        An array containing the data in `x`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.from_dlpack.html
    """

    assert x, "TODO"
    assert False, "TODO"


def full(
    shape: int | tuple[int, ...],
    fill_value: bool | int | float | complex,
    *,
    dtype: None | Dtype = None,
    device: None | Device = None,
) -> array:
    """
    Returns a new array having a specified shape and filled with fill_value.

    Args:
        shape: Output array shape.
        fill_value: Fill value.
        dtype: Output array data type. If `dtype` is `None`, the output array
            data type is inferred from `fill_value` according to the following
            rules:
                - if the fill value is an `int`, the output array data type is
                  `np.int64`.
                - if the fill value is a `float`, the output array data type
                  is `np.float64`.
                - if the fill value is a `complex` number, the output array
                  data type is `np.complex128`.
                - if the fill value is a `bool`, the output array is
                  `np.bool_`.
        device: Device on which to place the created array.

    Returns:
        An array where every element is equal to fill_value.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.full.html
    """

    assert shape, "TODO"
    assert fill_value, "TODO"
    assert dtype, "TODO"
    assert device, "TODO"
    assert False, "TODO"


def full_like(
    x: array,
    /,
    fill_value: bool | int | float | complex,
    *,
    dtype: None | Dtype = None,
    device: None | Device = None,
) -> array:
    """
    Returns a new array filled with fill_value and having the same shape as an input array x.

    Args:
        x: Input array from which to derive the output array shape.
        fill_value: Fill value.
        dtype: Output array data type. If `dtype` is `None`, the output array
            data type is inferred from `x`.
        device: Device on which to place the created array. If `device` is
            `None`, the output array device is inferred from `x`.

    Returns:
        An array having the same shape as `x` and where every element is equal
        to `fill_value`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.full_like.html
    """

    assert x, "TODO"
    assert fill_value, "TODO"
    assert dtype, "TODO"
    assert device, "TODO"
    assert False, "TODO"


def linspace(
    start: int | float | complex,
    stop: int | float | complex,
    /,
    num: int,
    *,
    dtype: None | Dtype = None,
    device: None | Device = None,
    endpoint: bool = True,
) -> array:
    r"""
    Returns evenly spaced numbers over a specified interval.

    Let `N` be the number of generated values (which is either `num` or `num+1`
    depending on whether `endpoint` is `True` or `False`, respectively). For
    real-valued output arrays, the spacing between values is given by

    $$\Delta_{\textrm{real}} = \frac{\textrm{stop} - \textrm{start}}{N - 1}$$

    For complex output arrays, let `a = real(start)`, `b = imag(start)`,
    `c = real(stop)`, and `d = imag(stop)`. The spacing between complex values
    is given by

    $$\Delta_{\textrm{complex}} = \frac{c-a}{N-1} + \frac{d-b}{N-1} j$$

    Args:
        start: The start of the interval.
        stop: The end of the interval. If `endpoint` is `False`, the function
            generates a sequence of `num+1` evenly spaced numbers starting with
            `start` and ending with `stop` and exclude the `stop` from the
            returned array such that the returned array consists of evenly
            spaced numbers over the half-open interval `[start, stop)`. If
            endpoint is `True`, the output array consists of evenly spaced
            numbers over the closed interval `[start, stop]`.
        num: Number of samples. Must be a nonnegative integer value.
        dtype: Output array data type. Should be a floating-point data type.
            If `dtype` is `None`,
                - if either `start` or `stop` is a `complex` number, the
                  output data type is `np.complex128`.
                - if both `start` and `stop` are real-valued, the output data
                  type is `np.float64`.
        device: Device on which to place the created array.
        endpoint: Boolean indicating whether to include `stop` in the interval.

    Returns:
        A one-dimensional array containing evenly spaced values.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.linspace.html
    """

    assert start, "TODO"
    assert stop, "TODO"
    assert num, "TODO"
    assert dtype, "TODO"
    assert device, "TODO"
    assert endpoint, "TODO"
    assert False, "TODO"


def meshgrid(*arrays: array, indexing: str = "xy") -> list[array]:
    """
    Returns coordinate matrices from coordinate vectors.

    Args:
        arrays: An arbitrary number of one-dimensional arrays representing
            grid coordinates. Each array should have the same numeric data type.
        indexing: Cartesian `"xy"` or matrix `"ij"` indexing of output. If
            provided zero or one one-dimensional vector(s) (i.e., the zero- and
            one-dimensional cases, respectively), the `indexing` keyword has no
            effect and should be ignored.

    Returns:
        List of `N` arrays, where `N` is the number of provided one-dimensional
        input arrays. Each returned array must have rank `N`. For `N`
        one-dimensional arrays having lengths `Ni = len(xi)`,
            - if matrix indexing `"ij"`, then each returned array must have the
              shape `(N1, N2, N3, ..., Nn)`.
            - if Cartesian indexing `"xy"`, then each returned array must have
              shape `(N2, N1, N3, ..., Nn)`.

        Accordingly, for the two-dimensional case with input one-dimensional
        arrays of length `M` and `N`, if matrix indexing `"ij"`, then each
        returned array must have shape `(M, N)`, and, if Cartesian indexing
        `"xy"`, then each returned array must have shape `(N, M)`.

        Similarly, for the three-dimensional case with input one-dimensional
        arrays of length `M`, `N`, and `P`, if matrix indexing `"ij"`, then
        each returned array must have shape `(M, N, P)`, and, if Cartesian
        indexing `"xy"`, then each returned array must have shape `(N, M, P)`.

        Each returned array should have the same data type as the input arrays.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.meshgrid.html
    """

    assert arrays, "TODO"
    assert indexing, "TODO"
    assert False, "TODO"


def ones(
    shape: int | tuple[int, ...],
    *,
    dtype: None | Dtype = None,
    device: None | Device = None,
) -> array:
    """
    Returns a new array having a specified `shape` and filled with ones.

    Args:
        shape: Output array shape.
        dtype: Output array data type. If `dtype` is `None`, the output array
            data type is `np.float64`.
        device: Device on which to place the created array.

    Returns:
        An array containing ones.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.ones.html
    """

    assert shape, "TODO"
    assert dtype, "TODO"
    assert device, "TODO"
    assert False, "TODO"


def ones_like(
    x: array, /, *, dtype: None | Dtype = None, device: None | Device = None
) -> array:
    """
    Returns a new array filled with ones and having the same `shape` as an
    input array `x`.

    Args:
        x: Input array from which to derive the output array shape.
        dtype: Output array data type. If `dtype` is `None`, the output array
            data type is inferred from `x`.
        device: Device on which to place the created array. If `device` is
            `None`, the output array device is inferred from `x`.

    Returns:
        An array having the same shape as x and filled with ones.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.ones_like.html
    """

    assert x, "TODO"
    assert dtype, "TODO"
    assert device, "TODO"
    assert False, "TODO"


def tril(x: array, /, *, k: int = 0) -> array:
    """
    Returns the lower triangular part of a matrix (or a stack of matrices) `x`.

    Args:
        x: Input array having shape `(..., M, N)` and whose innermost two
            dimensions form `M` by `N` matrices.
        `k`: Diagonal above which to zero elements. If `k = 0`, the diagonal is
            the main diagonal. If `k < 0`, the diagonal is below the main
            diagonal. If `k > 0`, the diagonal is above the main diagonal.

    Returns:
        An array containing the lower triangular part(s). The returned array
        has the same shape and data type as `x`. All elements above the
        specified diagonal `k` are zero. The returned array is allocated on the
        same device as `x`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.tril.html
    """

    assert x, "TODO"
    assert k, "TODO"
    assert False, "TODO"


def triu(x: array, /, *, k: int = 0) -> array:
    """
    Returns the upper triangular part of a matrix (or a stack of matrices) `x`.

    Args:
        x: Input array having shape `(..., M, N)` and whose innermost two
            dimensions form `M` by `N` matrices.
        k: Diagonal below which to zero elements. If `k = 0`, the diagonal is
            the main diagonal. If `k < 0`, the diagonal is below the main
            diagonal. If `k > 0`, the diagonal is above the main diagonal.

    Returns:
        An array containing the upper triangular part(s). The returned array
        has the same shape and data type as `x`. All elements below the
        specified diagonal `k` are zero. The returned array is allocated on the
        same device as `x`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.triu.html
    """

    assert x, "TODO"
    assert k, "TODO"
    assert False, "TODO"


def zeros(
    shape: int | tuple[int, ...],
    *,
    dtype: None | Dtype = None,
    device: None | Device = None,
) -> array:
    """
    Returns a new array having a specified shape and filled with zeros.

    Args:
        shape: Output array shape.
        dtype: Output array data type. If `dtype` is `None`, the output array
            data type is `np.float64`.
        device: Device on which to place the created array.

    Returns:
        An array containing zeros.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.zeros.html
    """

    assert shape, "TODO"
    assert dtype, "TODO"
    assert device, "TODO"
    assert False, "TODO"


def zeros_like(
    x: array, /, *, dtype: None | Dtype = None, device: None | Device = None
) -> array:
    """
    Returns a new array filled with zeros and having the same `shape` as an
    input array `x`.

    Args:
        x: Input array from which to derive the output array shape.
        dtype: Output array data type. If `dtype` is `None`, the output array
            data type is inferred from `x`.
        device: Device on which to place the created array. If `device` is
            `None`, the output array device is inferred from `x`.

    Returns:
        An array having the same shape as `x` and filled with zeros.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.zeros_like.html
    """

    assert x, "TODO"
    assert dtype, "TODO"
    assert device, "TODO"
    assert False, "TODO"
