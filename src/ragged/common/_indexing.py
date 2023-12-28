# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/indexing_functions.html
"""

from __future__ import annotations

from ._obj import array


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

    assert x, "TODO"
    assert indices, "TODO"
    assert axis, "TODO"
    assert False, "TODO"
