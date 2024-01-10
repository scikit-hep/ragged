# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/searching_functions.html
"""

from __future__ import annotations

import awkward as ak
import numpy as np

from ._import import device_namespace
from ._spec_array_object import _box, _unbox, array


def _remove_optiontype(x: ak.contents.Content) -> ak.contents.Content:
    if x.is_list:
        return x.copy(content=_remove_optiontype(x.content))
    elif x.is_option:
        return x.content
    else:
        return x


def argmax(x: array, /, *, axis: None | int = None, keepdims: bool = False) -> array:
    """
    Returns the indices of the maximum values along a specified axis.

    When the maximum value occurs multiple times, only the indices
    corresponding to the first occurrence are returned.

    Args:
        x: Input array.
        axis: Axis along which to search. If `None`, the function returns the
            index of the maximum value of the flattened array.
        keepdims: If `True`, the reduced axes (dimensions) are included in the
            result as singleton dimensions, and, accordingly, the result is
            broadcastable with the input array. Otherwise, if `False`, the
            reduced axes (dimensions) are not included in the result.

    Returns:
        If `axis` is `None`, a zero-dimensional array containing the index of
        the first occurrence of the maximum value; otherwise, a
        non-zero-dimensional array containing the indices of the maximum
        values. The returned array has data type `np.int64`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.argmax.html
    """

    out = np.argmax(*_unbox(x), axis=axis, keepdims=keepdims)

    if out is None:
        msg = "cannot compute argmax of an array with no data"
        raise ValueError(msg)

    if isinstance(out, ak.Array):
        if ak.any(ak.is_none(out, axis=-1)):
            msg = f"cannot compute argmax at axis={axis} because some lists at this depth have zero length"
            raise ValueError(msg)
        out = ak.Array(
            _remove_optiontype(out.layout), behavior=out.behavior, attrs=out.attrs
        )

    return _box(type(x), out)


def argmin(x: array, /, *, axis: None | int = None, keepdims: bool = False) -> array:
    """
    Returns the indices of the minimum values along a specified axis.

    When the minimum value occurs multiple times, only the indices
    corresponding to the first occurrence are returned.

    Args:
        x: Input array.
        axis: Axis along which to search. If `None`, the function returns the
            index of the minimum value of the flattened array.
        keepdims: If `True`, the reduced axes (dimensions) are included in the
            result as singleton dimensions, and, accordingly, the result is
            broadcastable with the input array. Otherwise, if `False`, the
            reduced axes (dimensions) are not included in the result.

    Returns:
        If `axis` is `None`, a zero-dimensional array containing the index of
        the first occurrence of the minimum value; otherwise, a
        non-zero-dimensional array containing the indices of the minimum
        values. The returned array has data type `np.int64`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.argmin.html
    """

    out = np.argmin(*_unbox(x), axis=axis, keepdims=keepdims)

    if out is None:
        msg = "cannot compute argmin of an array with no data"
        raise ValueError(msg)

    if isinstance(out, ak.Array):
        if ak.any(ak.is_none(out, axis=-1)):
            msg = f"cannot compute argmin at axis={axis} because some lists at this depth have zero length"
            raise ValueError(msg)
        out = ak.Array(
            _remove_optiontype(out.layout), behavior=out.behavior, attrs=out.attrs
        )

    return _box(type(x), out)


def nonzero(x: array, /) -> tuple[array, ...]:
    """
    Returns the indices of the array elements which are non-zero.

    Args:
        x: Input array. Must have a positive rank. If `x` is zero-dimensional,
        the function raises an exception.

    Returns:
        A tuple of `k` arrays, one for each dimension of `x` and each of size
        `n` (where `n` is the total number of non-zero elements), containing
        the indices of the non-zero elements in that dimension. The indices
        are returned in row-major, C-style order. The returned array has data
        type `np.int64`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.nonzero.html
    """

    (impl,) = _unbox(x)
    if not isinstance(impl, ak.Array):
        impl = ak.Array(impl.reshape((1,)))  # type: ignore[union-attr]

    return tuple(_box(type(x), item) for item in ak.where(impl))


def where(condition: array, x1: array, x2: array, /) -> array:
    """
    Returns elements chosen from `x1` or `x2` depending on `condition`.

    Args:
        condition: When `True`, yield `x1_i`; otherwise, yield `x2_i`. Must be
            broadcastable with `x1` and `x2`.
        x1: First input array. Must be broadcastable with `condition` and `x2`.
        x2: Second input array. Must be broadcastable with `condition` and
            `x1`.

    Returns:
        An array with elements from `x1` where condition is `True`, and
        elements from `x2` elsewhere. The returned array has a data type
        determined by type promotion rules with the arrays `x1` and `x2`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.where.html
    """

    if condition.ndim == x1.ndim == x2.ndim == 0:
        cond_impl, x1_impl, x2_impl = _unbox(condition, x1, x2)
        _, ns = device_namespace(condition.device)
        return _box(type(condition), ns.where(cond_impl, x1_impl, x2_impl))

    else:
        cond_impl, x1_impl, x2_impl = _unbox(condition, x1, x2)
        if not isinstance(cond_impl, ak.Array):
            cond_impl = ak.Array(cond_impl.reshape((1,)))  # type: ignore[union-attr]
        if not isinstance(x1_impl, ak.Array):
            x1_impl = ak.Array(x1_impl.reshape((1,)))  # type: ignore[union-attr]
        if not isinstance(x2_impl, ak.Array):
            x2_impl = ak.Array(x2_impl.reshape((1,)))  # type: ignore[union-attr]

        cond_impl, x1_impl, x2_impl = ak.broadcast_arrays(cond_impl, x1_impl, x2_impl)

        return _box(type(condition), ak.where(cond_impl, x1_impl, x2_impl))
