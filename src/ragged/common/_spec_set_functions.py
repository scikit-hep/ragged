# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/set_functions.html
"""

from __future__ import annotations

from collections import namedtuple

from ._spec_array_object import array

unique_all_result = namedtuple(  # pylint: disable=C0103
    "unique_all_result", ["values", "indices", "inverse_indices", "counts"]
)


def unique_all(x: array, /) -> tuple[array, array, array, array]:
    """
    Returns the unique elements of an input array `x`, the first occurring
    indices for each unique element in `x`, the indices from the set of unique
    elements that reconstruct `x`, and the corresponding `counts` for each
    unique element in `x`.

    Args:
        x: Input array. If `x` has more than one dimension, the function
            flattens `x` and returns the unique elements of the flattened
            array.

    Returns:
        A namedtuple `(values, indices, inverse_indices, counts)` whose

        - first element has the field name `values` and must be an array
          containing the unique elements of `x`. The array has the same data
          type as `x`.
        - second element has the field name `indices` and is an array containing
          the indices (first occurrences) of `x` that result in values. The
          array has the same shape as `values` and has the default array index
          data type.
        - third element has the field name `inverse_indices` and is an array
          containing the indices of values that reconstruct `x`. The array has
          the same shape as `x` and has data type `np.int64`.
        - fourth element has the field name `counts` and is an array containing
          the number of times each unique element occurs in `x`. The returned
          array has same shape as `values` and has data type `np.int64`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.unique_all.html
    """

    assert x, "TODO"
    assert False, "TODO"


unique_counts_result = namedtuple(  # pylint: disable=C0103
    "unique_counts_result", ["values", "counts"]
)


def unique_counts(x: array, /) -> tuple[array, array]:
    """
    Returns the unique elements of an input array `x` and the corresponding
    counts for each unique element in `x`.

    Args:
        x: Input array. If `x` has more than one dimension, the function
            flattens `x` and returns the unique elements of the flattened
            array.

    Returns:
        A namedtuple `(values, counts)` whose

        - first element has the field name `values` and is an array containing
          the unique elements of `x`. The array has the same data type as `x`.
        - second element has the field name `counts` and is an array containing
          the number of times each unique element occurs in `x`. The returned
          array has same shape as `values` and has data type `np.int64`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.unique_counts.html
    """

    assert x, "TODO"
    assert False, "TODO"


unique_inverse_result = namedtuple(  # pylint: disable=C0103
    "unique_inverse_result", ["values", "inverse_indices"]
)


def unique_inverse(x: array, /) -> tuple[array, array]:
    """
    Returns the unique elements of an input array `x` and the indices from the
    set of unique elements that reconstruct `x`.

    Args:
        x: Input array. If `x` has more than one dimension, the function
            flattens `x` and returns the unique elements of the flattened
            array.

    Returns:
        A namedtuple `(values, inverse_indices)` whose

        - first element has the field name `values` and is an array containing
          the unique elements of `x`. The array has the same data type as `x`.
        - second element has the field name `inverse_indices` and is an array
          containing the indices of `values` that reconstruct `x`. The array
          has the same shape as `x` and data type `np.int64`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.unique_inverse.html
    """

    assert x, "TODO"
    assert False, "TODO"


def unique_values(x: array, /) -> array:
    """
    Returns the unique elements of an input array `x`.

    Args:
        x: Input array. If `x` has more than one dimension, the function
            flattens `x` and returns the unique elements of the flattened
            array.

    Returns:
        An array containing the set of unique elements in `x`. The returned
        array has the same data type as `x`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.unique_values.html
    """

    assert x, "TODO"
    assert False, "TODO"
