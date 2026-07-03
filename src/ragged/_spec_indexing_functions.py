# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/indexing_functions.html
"""

from __future__ import annotations

import contextlib
from typing import Any

import awkward as ak
import numpy as np

from ._spec_array_object import _box, _unbox, array


def take(x: array, indices: array, /, *, axis: None | int = None) -> array:
    """
    Returns elements of an array along an axis.

    Conceptually, `take(x, indices, axis=3)` is equivalent to
    `x[:,:,:,indices,...]`.

    Args:
        x: Input array.
        indices: Array indices. The array must be one-dimensional and have an
            integer data type.
        axis: Axis over which to select values. If `axis` is negative, the
            function determines the axis along which to select values by
            counting from the last dimension.

            If `x` is a one-dimensional array, providing an axis is optional;
            however, if `x` has more than one dimension, providing an `axis` is
            required.

    Returns:
        An array having the same data type as `x`. The output array has the
        same rank (i.e., number of dimensions) as `x` and has the same shape as
        `x`, except for the axis specified by `axis` whose size must equal the
        number of elements in indices.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.take.html
    """

    if axis is None:
        if x.ndim <= 1:
            axis = 0
        else:
            msg = f"for an {x.ndim}-dimensional array (greater than 1-dimensional), the 'axis' argument is required"
            raise TypeError(msg)

    original_axis = axis
    if axis < 0:
        axis += x.ndim + 1
    if not 0 <= axis < x.ndim:
        msg = f"axis {original_axis} is out of bounds for array of dimension {x.ndim}"
        raise ak.errors.AxisError(msg)

    toslice = x._impl  # pylint: disable=W0212
    if not isinstance(toslice, ak.Array):
        toslice = ak.Array(toslice[np.newaxis])  # type: ignore[index]

    if not isinstance(indices, array):
        indices = array(indices)  # type: ignore[unreachable]
    indexarray = indices._impl  # pylint: disable=W0212

    slicer = (slice(None),) * axis + (indexarray,)
    return _box(type(x), toslice[slicer])


def take_along_axis(x: array, indices: array, /, *, axis: int = -1) -> array:
    """
    Selects elements from an array using indices along a given axis.

    Args:
        x: Input array.
        indices: Array indices. Must have the same rank as ``x``.
        axis: Axis over which to select values. If negative, counts from the last dimension.
            Default: ``-1``.

    Returns:
        An array having the same data type as ``x`` and the same shape as ``indices``.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.take_along_axis.html
    """
    x_impl, indices_impl = _unbox(x, indices)

    if x.ndim != indices.ndim:
        msg = f"indices and x must have the same rank, got {indices.ndim} and {x.ndim}"
        raise ValueError(msg)

    # Normalize negative axis
    original_axis = axis
    if axis < 0:
        axis += x.ndim
    if not 0 <= axis < x.ndim:
        msg = f"axis {original_axis} is out of bounds for array of dimension {x.ndim}"
        raise ValueError(msg)

    # Fast path: uniform arrays via numpy
    with contextlib.suppress(TypeError, ValueError):
        x_np = ak.to_numpy(x_impl)
        indices_np = ak.to_numpy(indices_impl)
        result_np = np.take_along_axis(x_np, indices_np, axis=axis)
        return _box(type(x), ak.from_numpy(result_np))

    # Ragged path via Awkward Array
    if axis == x.ndim - 1:
        # For the innermost axis, awkward array natively supports gathering
        # via direct indexing: x_impl[indices_impl]
        return _box(type(x), x_impl[indices_impl])  # type: ignore[index]

    # For outer axes in ragged arrays, we fall back to list reconstruction.
    # (Since outer-axis gathering across ragged boundaries is complex and rare).
    def _take_along(arr_obj: Any, idx_obj: Any, current_depth: int) -> Any:
        if current_depth == axis:
            return [arr_obj[i] for i in idx_obj]

        # Zip elements together and recurse
        return [
            _take_along(a_item, i_item, current_depth + 1)
            for a_item, i_item in zip(arr_obj, idx_obj, strict=False)
        ]

    x_list = ak.to_list(x_impl)
    indices_list = ak.to_list(indices_impl)

    return _box(type(x), ak.Array(_take_along(x_list, indices_list, 0)))