# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/linear_algebra_functions.html
"""

from __future__ import annotations

from collections.abc import Sequence

from ._spec_array_object import array


def matmul(x1: array, x2: array, /) -> array:
    """
    Computes the matrix product.

    Args:
        x1: First input array. Must have at least one dimension. If `x1` is
        one-dimensional having shape `(M,)` and `x2` has more than one
        dimension, `x1` is promoted to a two-dimensional array by prepending 1
        to its dimensions (i.e., has shape `(1, M)`). After matrix
        multiplication, the prepended dimensions in the returned array are
        removed. If `x1` has more than one dimension (including after
        vector-to-matrix promotion), `shape(x1)[:-2]` is compatible with
        `shape(x2)[:-2]` (after vector-to-matrix promotion). If `x1` has shape
        `(..., M, K)`, the innermost two dimensions form matrices on which to
        perform matrix multiplication.
    x2: Second input array. Must have at least one dimension. If `x2` is
        one-dimensional having shape `(N,)` and `x1` has more than one
        dimension, `x2` is promoted to a two-dimensional array by appending 1
        to its dimensions (i.e., has shape `(N, 1)`). After matrix
        multiplication, the appended dimensions in the returned array are
        removed. If `x2` has more than one dimension (including after
        vector-to-matrix promotion), `shape(x2)[:-2]` is compatible with
        `shape(x1)[:-2]` (after vector-to-matrix promotion). If `x2` has shape
        `(..., K, N)`, the innermost two dimensions form matrices on which to
        perform matrix multiplication.

    Returns:
        If both `x1` and `x2` are one-dimensional arrays having shape `(N,)`, a
        zero-dimensional array containing the inner product as its only
        element.

        If `x1` is a two-dimensional array having shape `(M, K)` and `x2` is a
        two-dimensional array having shape `(K, N)`, a two-dimensional array
        containing the conventional matrix product and having shape `(M, N)`.

        If `x1` is a one-dimensional array having shape `(K,)` and `x2` is an
        array having shape `(..., K, N)`, an array having shape `(..., N)`
        (i.e., prepended dimensions during vector-to-matrix promotion are
        removed) and containing the conventional matrix product.

        If `x1` is an array having shape `(..., M, K)` and `x2` is a
        one-dimensional array having shape `(K,)`, an array having shape
        `(..., M)` (i.e., appended dimensions during vector-to-matrix promotion
        are removed) and containing the conventional matrix product.

        If `x1` is a two-dimensional array having shape `(M, K)` and `x2` is an
        array having shape `(..., K, N)`, an array having shape `(..., M, N)`
        and containing the conventional matrix product for each stacked matrix.

        If `x1` is an array having shape `(..., M, K)` and `x2` is a
        two-dimensional array having shape `(K, N)`, an array having shape
        `(..., M, N)` and containing the conventional matrix product for each
        stacked matrix.

        If either `x1` or `x2` has more than two dimensions, an array having a
        shape determined by broadcasting `shape(x1)[:-2]` against
        `Shape(x2)[:-2]` and containing the conventional matrix product for
        each stacked matrix.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.matmul.html
    """

    x1  # noqa: B018, pylint: disable=W0104
    x2  # noqa: B018, pylint: disable=W0104
    raise NotImplementedError("TODO 110")  # noqa: EM101


def matrix_transpose(x: array, /) -> array:
    """
    Transposes a matrix (or a stack of matrices) x.

    Args:
        x: Input array having shape `(..., M, N)` and whose innermost two
        dimensions form `M` by `N` matrices.

    Returns:
        An array containing the transpose for each matrix and having shape
        `(..., N, M)`. The returned array has the same data type as `x`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.matrix_transpose.html
    """

    x  # noqa: B018, pylint: disable=W0104
    raise NotImplementedError("TODO 111")  # noqa: EM101


def tensordot(
    x1: array, x2: array, /, *, axes: int | tuple[Sequence[int], Sequence[int]] = 2
) -> array:
    """
    Returns a tensor contraction of `x1` and `x2` over specific axes.

    The tensordot function corresponds to the generalized matrix product.

    Args:
        x1: First input array.
        x2: Second input array. Corresponding contracted axes of `x1` and `x2`
            must be equal.
        axes: Number of axes (dimensions) to contract or explicit sequences of
            axes (dimensions) for `x1` and `x2`, respectively.

            If `axes` is an `int` equal to `N`, then contraction is performed
            over the last `N` axes of `x1` and the first `N` axes of `x2` in
            order. The size of each corresponding axis (dimension) match. Must
            be nonnegative.

            If `N` equals 0, the result is the tensor (outer) product.

            If `N` equals 1, the result is the tensor dot product.

            If `N` equals 2, the result is the tensor double contraction.

            If `axes` is a tuple of two sequences `(x1_axes, x2_axes)`, the
            first sequence applies to `x1` and the second sequence to `x2`.
            Both sequences must have the same length. Each axis (dimension)
            `x1_axes[i]` for `x1` must have the same size as the respective
            axis (dimension) `x2_axes[i]` for `x2`. Each sequence must consist
            of unique (nonnegative) integers that specify valid axes for each
            respective array.

    Returns:
        An array containing the tensor contraction whose shape consists of the
        non-contracted axes (dimensions) of the first array `x1`, followed by
        the non-contracted axes (dimensions) of the second array `x2`. The
        returned array has a data type determined by type promotion rules.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.tensordot.html
    """

    x1  # noqa: B018, pylint: disable=W0104
    x2  # noqa: B018, pylint: disable=W0104
    axes  # noqa: B018, pylint: disable=W0104
    raise NotImplementedError("TODO 112")  # noqa: EM101


def vecdot(x1: array, x2: array, /, *, axis: int = -1) -> array:
    r"""
    Computes the (vector) dot product of two arrays.

    Let $\mathbf{a}$ be a vector in `x1` and $\mathbf{b}$ be a corresponding
    vector in `x2`. The dot product is defined as

    $$\mathbf{a} \cdot \mathbf{b} = \sum_{i=0}^{n-1} \overline{a_i}b_i$$

    over the dimension specified by `axis` and where $n$ is the dimension size
    and $\overline{a_i}$ denotes the complex conjugate if $a_i$ is complex and
    the identity if $a_i$ is real-valued.

    Args:
        x1: First input array.
        x2: Second input array. Must be broadcastable with `x1` for all
            non-contracted axes. The size of the axis over which to compute the
            dot product is the same size as the respective axis in `x1`.

            The contracted axis (dimension) is not broadcasted.
        axis: Axis over which to compute the dot product. Must be an integer on
        the interval `[-N, N)`, where `N` is the rank (number of dimensions) of
        the shape determined by broadcasting. If specified as a negative
        integer, the function determines the axis along which to compute the
        dot product by counting backward from the last dimension (where `-1`
        refers to the last dimension).

    Returns:
        If `x1` and `x2` are both one-dimensional arrays, a zero-dimensional
        containing the dot product; otherwise, a non-zero-dimensional array
        containing the dot products and having rank `N - 1`, where `N` is the
        rank (number of dimensions) of the shape determined by broadcasting
        along the non-contracted axes. The returned array has a data type
        determined by type promotion.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.vecdot.html
    """

    x1  # noqa: B018, pylint: disable=W0104
    x2  # noqa: B018, pylint: disable=W0104
    axis  # noqa: B018, pylint: disable=W0104
    raise NotImplementedError("TODO 113")  # noqa: EM101
