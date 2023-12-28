# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/statistical_functions.html
"""

from __future__ import annotations

from ._spec_array_object import array
from ._typing import Dtype


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

    assert x, "TODO"
    assert axis, "TODO"
    assert keepdims, "TODO"
    assert False, "TODO"


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

    assert x, "TODO"
    assert axis, "TODO"
    assert keepdims, "TODO"
    assert False, "TODO"


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

    assert x, "TODO"
    assert axis, "TODO"
    assert keepdims, "TODO"
    assert False, "TODO"


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

    assert x, "TODO"
    assert axis, "TODO"
    assert dtype, "TODO"
    assert keepdims, "TODO"
    assert False, "TODO"


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

    assert x, "TODO"
    assert axis, "TODO"
    assert correction, "TODO"
    assert keepdims, "TODO"
    assert False, "TODO"


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

    assert x, "TODO"
    assert axis, "TODO"
    assert dtype, "TODO"
    assert keepdims, "TODO"
    assert False, "TODO"


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

    assert x, "TODO"
    assert axis, "TODO"
    assert correction, "TODO"
    assert keepdims, "TODO"
    assert False, "TODO"
