# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/statistical_functions.html
"""

from __future__ import annotations

import numbers

import awkward as ak
import numpy as np

from ._spec_array_object import _box, _unbox, array
from ._typing import Dtype


def _regularize_axis(
    axis: None | int | tuple[int, ...], ndim: int
) -> None | tuple[int, ...]:
    if axis is None:
        return axis
    elif isinstance(axis, numbers.Integral):
        out = axis + ndim if axis < 0 else axis  # type: ignore[operator]
        if not 0 <= out < ndim:
            msg = f"axis {axis} is out of bounds for an array with {ndim} dimensions"
            raise ak.errors.AxisError(msg)
        return out  # type: ignore[no-any-return]
    else:
        out = []
        for x in axis:  # type: ignore[union-attr]
            out.append(x + ndim if x < 0 else x)
            if not 0 < out[-1] < ndim:
                msg = f"axis {x} is out of bounds for an array with {ndim} dimensions"
        if len(out) == 0:
            msg = "at least one axis must be specified"
            raise ak.errors.AxisError(msg)
        return tuple(sorted(out))


def _regularize_dtype(dtype: None | Dtype, array_dtype: Dtype) -> Dtype:
    if dtype is None:
        if array_dtype.kind in ("b", "i"):
            return np.dtype(np.int64)
        elif array_dtype.kind == "u":
            return np.dtype(np.uint64)
        elif array_dtype.kind == "f":
            return np.dtype(np.float64)
        elif array_dtype.kind == "c":
            return np.dtype(np.complex128)
        else:
            msg = f"unrecognized dtype.kind: {array_dtype.kind}"
            raise AssertionError(msg)
    else:
        return dtype


def _ensure_dtype(data: array, dtype: Dtype) -> array:
    if data.dtype == dtype:
        return data
    else:
        (tmp,) = _unbox(data)
        if isinstance(tmp, ak.Array):
            return _box(type(data), ak.values_astype(tmp, dtype))
        else:
            return _box(type(data), tmp.astype(dtype))  # type: ignore[union-attr]


def max(  # pylint: disable=W0622
    x: array, /, *, axis: None | int | tuple[int, ...] = None, keepdims: bool = False
) -> array:
    """
    Calculates the maximum value of the input array `x`.

    Args:
        x: Input array.
        axis: Axis or axes along which maximum values are computed. By default,
            the maximum value is computed over the entire array. If a tuple of
            integers, maximum values must be computed over multiple axes.
        keepdims: If `True`, the reduced axes (dimensions) are included in the
            result as singleton dimensions, and, accordingly, the result is
            broadcastable with the input array. Otherwise, if `False`, the
            reduced axes (dimensions) are not included in the result.

    Returns:
        If the maximum value was computed over the entire array, a
        zero-dimensional array containing the maximum value; otherwise, a
        non-zero-dimensional array containing the maximum values. The returned
        array has the same data type as `x`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.max.html
    """

    axis = _regularize_axis(axis, x.ndim)

    if isinstance(axis, tuple):
        (out,) = _unbox(x)
        for axis_item in axis[::-1]:
            if isinstance(out, ak.Array):
                out = ak.max(
                    out, axis=axis_item, keepdims=keepdims, mask_identity=False
                )
            else:
                out = np.max(out, axis=axis_item, keepdims=keepdims)
        return _box(type(x), out)
    else:
        (tmp,) = _unbox(x)
        if isinstance(tmp, ak.Array):
            out = ak.max(tmp, axis=axis, keepdims=keepdims, mask_identity=False)
        else:
            out = np.max(tmp, axis=axis, keepdims=keepdims)
        return _box(type(x), out)


def mean(
    x: array, /, *, axis: None | int | tuple[int, ...] = None, keepdims: bool = False
) -> array:
    """
    Calculates the arithmetic mean of the input array `x`.

    Args:
        x: Input array.
        axis: Axis or axes along which arithmetic means are computed. By
            default, the mean is computed over the entire array. If a tuple of
            integers, arithmetic means are computed over multiple axes.
        keepdims: If `True`, the reduced axes (dimensions) are included in the
            result as singleton dimensions, and, accordingly, the result is
            broadcastable with the input array. Otherwise, if `False`, the
            reduced axes (dimensions) are not included in the result.

    Returns:
        If the arithmetic mean was computed over the entire array, a
        zero-dimensional array containing the arithmetic mean; otherwise, a
        non-zero-dimensional array containing the arithmetic means. The
        returned array has the same data type as `x`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.mean.html
    """

    axis = _regularize_axis(axis, x.ndim)

    if isinstance(axis, tuple):
        sumwx = np.sum(*_unbox(x), axis=axis[-1], keepdims=keepdims)
        sumw = ak.count(*_unbox(x), axis=axis[-1], keepdims=keepdims)
        for axis_item in axis[-2::-1]:
            sumwx = np.sum(sumwx, axis=axis_item, keepdims=keepdims)
            sumw = np.sum(sumw, axis=axis_item, keepdims=keepdims)
    else:
        sumwx = np.sum(*_unbox(x), axis=axis, keepdims=keepdims)
        sumw = ak.count(*_unbox(x), axis=axis, keepdims=keepdims)

    with np.errstate(invalid="ignore", divide="ignore"):
        return _ensure_dtype(_box(type(x), sumwx / sumw), x.dtype)


def min(  # pylint: disable=W0622
    x: array, /, *, axis: None | int | tuple[int, ...] = None, keepdims: bool = False
) -> array:
    """
    Calculates the minimum value of the input array `x`.

    Args:
        x: Input array.
        axis: Axis or axes along which minimum values are computed. By default,
            the minimum value are computed over the entire array. If a tuple of
            integers, minimum values are computed over multiple axes.
        keepdims: If `True`, the reduced axes (dimensions) are included in the
            result as singleton dimensions, and, accordingly, the result is
            broadcastable with the input array. Otherwise, if `False`, the
            reduced axes (dimensions) are not included in the result.

    Returns:
        If the minimum value was computed over the entire array, a
        zero-dimensional array containing the minimum value; otherwise, a
        non-zero-dimensional array containing the minimum values. The returned
        array has the same data type as `x`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.min.html
    """

    axis = _regularize_axis(axis, x.ndim)

    if isinstance(axis, tuple):
        (out,) = _unbox(x)
        for axis_item in axis[::-1]:
            if isinstance(out, ak.Array):
                out = ak.min(
                    out, axis=axis_item, keepdims=keepdims, mask_identity=False
                )
            else:
                out = np.min(out, axis=axis_item, keepdims=keepdims)
        return _box(type(x), out)
    else:
        (tmp,) = _unbox(x)
        if isinstance(tmp, ak.Array):
            out = ak.min(tmp, axis=axis, keepdims=keepdims, mask_identity=False)
        else:
            out = np.min(tmp, axis=axis, keepdims=keepdims)
        return _box(type(x), out)


def prod(
    x: array,
    /,
    *,
    axis: None | int | tuple[int, ...] = None,
    dtype: None | Dtype = None,
    keepdims: bool = False,
) -> array:
    """
        Calculates the product of input array `x` elements.

        Args:
            x: Input array.
            axis: Axis or axes along which products are computed. By default, the
                product is computed over the entire array. If a tuple of integers,
                products are computed over multiple axes.
            dtype: Data type of the returned array. If `None`,

                - if the default data type corresponding to the data type "kind"
    a              (integer, real-valued floating-point, or complex floating-point)
                  of `x` has a smaller range of values than the data type of `x`
                  (e.g., `x` has data type `int64` and the default data type is
                  `int32`, or `x` has data type `uint64` and the default data type
                  is `int64`), the returned array has the same data type as `x`.
                - if `x` has a real-valued floating-point data type, the returned
                  array has the default real-valued floating-point data type.
                - if `x` has a complex floating-point data type, the returned array
                  has data type `np.complex128`.
                - if `x` has a signed integer data type (e.g., `int16`), the
                  returned array has data type `np.int64`.
                - if `x` has an unsigned integer data type (e.g., `uint16`), the
                  returned array has data type `np.uint64`.

                If the data type (either specified or resolved) differs from the
                data type of `x`, the input array will be cast to the specified
                data type before computing the product.

            keepdims: If `True`, the reduced axes (dimensions) are included in the
                result as singleton dimensions, and, accordingly, the result is
                broadcastable with the input array. Otherwise, if `False`, the
                reduced axes (dimensions) are not included in the result.

        Returns:
            If the product was computed over the entire array, a zero-dimensional
            array containing the product; otherwise, a non-zero-dimensional array
            containing the products. The returned array has a data type as
            described by the `dtype` parameter above.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.prod.html
    """

    axis = _regularize_axis(axis, x.ndim)
    dtype = _regularize_dtype(dtype, x.dtype)
    arr = _box(type(x), ak.values_astype(*_unbox(x), dtype)) if x.dtype == dtype else x

    if isinstance(axis, tuple):
        (out,) = _unbox(arr)
        for axis_item in axis[::-1]:
            out = np.prod(out, axis=axis_item, keepdims=keepdims)
        return _box(type(x), out)
    else:
        return _box(type(x), np.prod(*_unbox(arr), axis=axis, keepdims=keepdims))


def std(
    x: array,
    /,
    *,
    axis: None | int | tuple[int, ...] = None,
    correction: None | int | float = 0.0,
    keepdims: bool = False,
) -> array:
    """
    Calculates the standard deviation of the input array `x`.

    Args:
        x: Input array.
        axis: Axis or axes along which standard deviations are computed. By
            default, the standard deviation is computed over the entire array.
            If a tuple of integers, standard deviations is computed over
            multiple axes.
        correction: Degrees of freedom adjustment. Setting this parameter to a
            value other than 0 has the effect of adjusting the divisor during
            the calculation of the standard deviation according to `N - c`
            where `N` corresponds to the total number of elements over which
            the standard deviation is computed and `c` corresponds to the
            provided degrees of freedom adjustment. When computing the standard
            deviation of a population, setting this parameter to 0 is the
            standard choice (i.e., the provided array contains data
            constituting an entire population). When computing the corrected
            sample standard deviation, setting this parameter to 1 is the
            standard choice (i.e., the provided array contains data sampled
            from a larger population; this is commonly referred to as Bessel's
            correction).
        keepdims: If `True`, the reduced axes (dimensions) are included in the
            result as singleton dimensions, and, accordingly, the result is
            broadcastable with the input array. Otherwise, if `False`, the
            reduced axes (dimensions) are not included in the result.

    Returns:
        If the standard deviation was computed over the entire array, a
        zero-dimensional array containing the standard deviation; otherwise, a
        non-zero-dimensional array containing the standard deviations.
        The returned array has the same data type as `x`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.std.html
    """

    return _box(
        type(x),
        np.sqrt(*_unbox(var(x, axis=axis, correction=correction, keepdims=keepdims))),
    )


def sum(  # pylint: disable=W0622
    x: array,
    /,
    *,
    axis: None | int | tuple[int, ...] = None,
    dtype: None | Dtype = None,
    keepdims: bool = False,
) -> array:
    """
    Calculates the sum of the input array `x`.

    Args:
        x: Input array.
        axis: Axis or axes along which sums are computed. By default, the sum
            is computed over the entire array. If a tuple of integers, sums
            are computed over multiple axes.
        dtype: Data type of the returned array. If `None`,

            - if the default data type corresponding to the data type "kind"
              (integer, real-valued floating-point, or complex floating-point)
              of `x` has a smaller range of values than the data type of `x`
              (e.g., `x` has data type `int64` and the default data type is
              `int32`, or `x` has data type `uint64` and the default data type
              is `int64`), the returned array has the same data type as `x`.
            - if `x` has a real-valued floating-point data type, the returned
              array has the default real-valued floating-point data type.
            - if `x` has a complex floating-point data type, the returned array
              has data type `np.complex128`.
            - if `x` has a signed integer data type (e.g., `int16`), the
              returned array has data type `np.int64`.
            - if `x` has an unsigned integer data type (e.g., `uint16`), the
              returned array has data type `np.uint64`.

            If the data type (either specified or resolved) differs from the
            data type of `x`, the input array is cast to the specified data
            type before computing the sum.

        keepdims: If `True`, the reduced axes (dimensions) are included in the
            result as singleton dimensions, and, accordingly, the result is
            broadcastable with the input array. Otherwise, if `False`, the
            reduced axes (dimensions) are not included in the result.

    Returns:
        If the sum was computed over the entire array, a zero-dimensional array
        containing the sum; otherwise, an array containing the sums. The
        returned array must have a data type as described by the `dtype`
        parameter above.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.sum.html
    """

    axis = _regularize_axis(axis, x.ndim)
    dtype = _regularize_dtype(dtype, x.dtype)
    arr = _box(type(x), ak.values_astype(*_unbox(x), dtype)) if x.dtype == dtype else x

    if isinstance(axis, tuple):
        (out,) = _unbox(arr)
        for axis_item in axis[::-1]:
            out = np.sum(out, axis=axis_item, keepdims=keepdims)
        return _box(type(x), out)
    else:
        return _box(type(x), np.sum(*_unbox(arr), axis=axis, keepdims=keepdims))


def var(
    x: array,
    /,
    *,
    axis: None | int | tuple[int, ...] = None,
    correction: None | int | float = 0.0,
    keepdims: bool = False,
) -> array:
    """
    Calculates the variance of the input array `x`.

    Args:
        x: Input array.
        axis: Axis or axes along which variances are computed. By default, the
            variance is computed over the entire array. If a tuple of integers,
            variances are computed over multiple axes.
        correction: Degrees of freedom adjustment. Setting this parameter to a
            value other than 0 has the effect of adjusting the divisor during
            the calculation of the variance according to `N - c` where `N`
            corresponds to the total number of elements over which the variance
            is computed and `c` corresponds to the provided degrees of freedom
            adjustment. When computing the variance of a population, setting
            this parameter to 0 is the standard choice (i.e., the provided
            array contains data constituting an entire population). When
            computing the unbiased sample variance, setting this parameter to 1
            is the standard choice (i.e., the provided array contains data
            sampled from a larger population; this is commonly referred to as
            Bessel's correction).
        keepdims: If `True`, the reduced axes (dimensions) are included in the
            result as singleton dimensions, and, accordingly, the result is
            broadcastable with the input array. Otherwise, if `False`, the
            reduced axes (dimensions) are not included in the result.

    Returns:
        If the variance was computed over the entire array, a zero-dimensional
        array containing the variance; otherwise, a non-zero-dimensional array
        containing the variances. The returned array has the same data type as
        `x`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.var.html
    """

    axis = _regularize_axis(axis, x.ndim)

    if isinstance(axis, tuple):
        sumwxx = np.sum(np.square(*_unbox(x)), axis=axis[-1], keepdims=keepdims)
        sumwx = np.sum(*_unbox(x), axis=axis[-1], keepdims=keepdims)
        sumw = ak.count(*_unbox(x), axis=axis[-1], keepdims=keepdims)
        for axis_item in axis[-2::-1]:
            sumwxx = np.sum(sumwxx, axis=axis_item, keepdims=keepdims)
            sumwx = np.sum(sumwx, axis=axis_item, keepdims=keepdims)
            sumw = np.sum(sumw, axis=axis_item, keepdims=keepdims)
    else:
        sumwxx = np.sum(np.square(*_unbox(x)), axis=axis, keepdims=keepdims)
        sumwx = np.sum(*_unbox(x), axis=axis, keepdims=keepdims)
        sumw = ak.count(*_unbox(x), axis=axis, keepdims=keepdims)

    with np.errstate(invalid="ignore", divide="ignore"):
        out = sumwxx / sumw - np.square(sumwx / sumw)
        if correction is not None and correction != 0:
            out *= sumw / (sumw - correction)
        return _ensure_dtype(_box(type(x), out), x.dtype)
