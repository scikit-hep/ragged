# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/sorting_functions.html
"""

from __future__ import annotations

import awkward as ak

from ._spec_array_object import _box, _unbox, array


def argsort(
    x: array, /, *, axis: int = -1, descending: bool = False, stable: bool = True
) -> array:
    """
    Returns the indices that sort an array `x` along a specified axis.

    Args:
        x: Input array.
        axis: Axis along which to sort. If set to -1, the function sorts along
            the last axis.
        descending: Sort order. If `True`, the returned indices sort `x` in
            descending order (by value). If `False`, the returned indices sort
            `x` in ascending order (by value).
        stable: Sort stability. If `True`, the returned indices will maintain
            the relative order of `x` values which compare as equal. If
            `False`, the returned indices may or may not maintain the relative
            order of `x` values which compare as equal.

    Returns:
        An array of indices. The returned array has the same shape as `x`.
        The returned array has data type `np.int64`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.argsort.html
    """

    (impl,) = _unbox(x)
    if not isinstance(impl, ak.Array):
        msg = f"axis {axis} is out of bounds for array of dimension 0"
        raise ak.errors.AxisError(msg)
    out = ak.argsort(impl, axis=axis, ascending=not descending, stable=stable)
    return _box(type(x), out)


def sort(
    x: array, /, *, axis: int = -1, descending: bool = False, stable: bool = True
) -> array:
    """
    Returns a sorted copy of an input array `x`.

    Args:
        x: Input array.
        axis: Axis along which to sort. If set to -1, the function sorts along
            the last axis.
        descending: Sort order. If `True`, the array is sorted in descending
            order (by value). If `False`, the array is sorted in ascending
            order (by value).
        stable: Sort stability. If `True`, the returned array will maintain the
            relative order of `x` values which compare as equal. If `False`,
            the returned array may or may not maintain the relative order of
            `x` values which compare as equal.

    Returns:
        A sorted array. The returned array has the same data type and shape as
        `x`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.sort.html
    """

    (impl,) = _unbox(x)
    if not isinstance(impl, ak.Array):
        msg = f"axis {axis} is out of bounds for array of dimension 0"
        raise ak.errors.AxisError(msg)
    out = ak.sort(impl, axis=axis, ascending=not descending, stable=stable)
    return _box(type(x), out)
