# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/utility_functions.html
"""

from __future__ import annotations

from ._obj import array


def all(  # pylint: disable=W0622
    x: array, /, *, axis: None | int | tuple[int, ...] = None, keepdims: bool = False
) -> array:
    """
    Tests whether all input array elements evaluate to `True` along a specified
    axis.

    Args:
        x: Input array.
        axis: Axis or axes along which to perform a logical AND reduction. By
            default, a logical AND reduction is performed over the entire
            array. If a tuple of integers, logical AND reductions are performed
            over multiple axes. A valid `axis` must be an integer on the
            interval `[-N, N)`, where `N` is the rank (number of dimensions) of
            `x`. If an `axis` is specified as a negative integer, the function
            must determine the axis along which to perform a reduction by
            counting backward from the last dimension (where -1 refers to the
            last dimension). If provided an invalid `axis`, the function raises
            an exception.
        keepdims: If `True`, the reduced axes (dimensions) are included in the
            result as singleton dimensions, and, accordingly, the result is
            broadcastable with the input array. Otherwise, if `False`, the
            reduced axes (dimensions) are not included in the result.

    Returns:
        If a logical AND reduction was performed over the entire array, the
        returned array is a zero-dimensional array containing the test result;
        otherwise, the returned array is a non-zero-dimensional array
        containing the test results. The returned array has data type
        `np.bool_`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.all.html
    """

    assert x, "TODO"
    assert axis, "TODO"
    assert keepdims, "TODO"
    assert False, "TODO"


def any(  # pylint: disable=W0622
    x: array, /, *, axis: None | int | tuple[int, ...] = None, keepdims: bool = False
) -> array:
    """
    Tests whether any input array element evaluates to True along a specified
    axis.

    Args:
        x: Input array.
        axis: Axis or axes along which to perform a logical OR reduction. By
            default, a logical OR reduction is performed over the entire array.
            If a tuple of integers, logical OR reductions aer performed over
            multiple axes. A valid `axis` must be an integer on the interval
            `[-N, N)`, where `N` is the rank (number of dimensions) of `x`. If
            an `axis` is specified as a negative integer, the function
            determines the axis along which to perform a reduction by counting
            backward from the last dimension (where -1 refers to the last
            dimension). If provided an invalid `axis`, the function raises an
            exception.
        keepdims: If `True`, the reduced axes (dimensions) aer included in the
            result as singleton dimensions, and, accordingly, the result is
            broadcastable with the input array. Otherwise, if `False`, the
            reduced axes (dimensions) are not included in the result.

    Returns:
        If a logical OR reduction was performed over the entire array, the
        returned array is a zero-dimensional array containing the test result;
        otherwise, the returned array is a non-zero-dimensional array
        containing the test results. The returned array has data type
        `np.bool_`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.any.html
    """

    assert x, "TODO"
    assert axis, "TODO"
    assert keepdims, "TODO"
    assert False, "TODO"
