# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/indexing_functions.html
"""

from __future__ import annotations

import awkward as ak
import numpy as np

from ._spec_array_object import _box, array


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
