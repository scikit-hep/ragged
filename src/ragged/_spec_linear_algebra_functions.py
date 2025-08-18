# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/linear_algebra_functions.html
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import awkward as ak
import numpy as np

from ._helper_functions import is_sorted_descending_all_levels, safe_max_num
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

    if x1.ndim == 0 or x2.ndim == 0:
        msg = "Zero-dimensional arrays are not allowed"
        raise ValueError(msg)

    # --- helper: promote 1D to 2D (list-of-rows) ---
    def _promote_1d_to_2d(arr: array) -> list[list[int | float | complex]]:
        impl_list = ak.to_list(arr._impl)  # pylint: disable=W0212

        if arr.ndim == 1:
            return [impl_list]

        return [
            list(row)
            if isinstance(row, Iterable) and not isinstance(row, (str, bytes))
            else [row]
            for row in impl_list
        ]

    # --- 1Dx1D -> scalar ---
    if x1.ndim == 1 and x2.ndim == 1:
        if len(x1) != len(x2):
            msg = "Shape mismatch for 1D arrays"
            raise ValueError(msg)
        x1_list = ak.to_list(x1._impl)  # pylint: disable=W0212
        x2_list = ak.to_list(x2._impl)  # pylint: disable=W0212
        return array(sum(a * b for a, b in zip(x1_list, x2_list)))

    # --- 1D x 2D / 2D x 1D promotion ---
    if x1.ndim == 1 and x2.ndim == 2:
        x1_list = ak.to_list(x1._impl)  # pylint: disable=W0212
        promoted_x1 = array([x1_list])  # shape (1, K)
        res2d = matmul(promoted_x1, x2)
        return array(res2d[0])  # drop leading 1 safely

    if x1.ndim == 2 and x2.ndim == 1:
        x2_list = ak.to_list(x2._impl)  # pylint: disable=W0212
        promoted_x2 = array([[v] for v in x2_list])  # shape (K, 1)
        res2d = matmul(x1, promoted_x2)  # returns 2D (M,1)

        # --- collapse trailing dimension safely ---
        if res2d.ndim == 2:
            impl_list = ak.to_list(res2d._impl)  # pylint: disable=W0212
            if len(impl_list) > 0:
                return array([row[0] for row in impl_list])
            else:
                return array([])  # empty 2D
        else:
            return array(ak.to_list(res2d._impl))  # pylint: disable=W0212

    # --- output dtype ---
    out_dtype = np.result_type(x1.dtype, x2.dtype)

    # --- ragged-safe 2Dx2D ---
    def _matmul_2d(a_impl: Any, b_impl: Any) -> list[Any]:
        if len(a_impl) == 0 or len(b_impl) == 0:
            return []

        max_cols_a = int(safe_max_num(a_impl, axis=-1))
        max_cols_b = int(safe_max_num(b_impl, axis=-1))
        all_b_rows_empty = all(len(row) == 0 for row in b_impl)
        eff_rows_b = 0 if all_b_rows_empty else len(b_impl)

        if max_cols_a != eff_rows_b and not (max_cols_a == 0 and eff_rows_b == 0):
            msg = "Shape mismatch in core dimensions"
            raise ValueError(msg)

        M, K, N = len(a_impl), max_cols_a, max_cols_b

        mat_a = np.zeros((M, K), dtype=out_dtype)
        for r, row in enumerate(a_impl):
            for c, val in enumerate(row):
                mat_a[r, c] = val

        mat_b = np.zeros((K, N), dtype=out_dtype)
        for r in range(min(len(b_impl), K)):
            row = b_impl[r]
            for c, val in enumerate(row):
                mat_b[r, c] = val

        product = mat_a @ mat_b

        out: list[Any] = []
        for r, orig_row in enumerate(a_impl):
            if len(orig_row) == 0:
                out.append([])
            else:
                out.append(list(product[r, :N]))
        return out

    # --- both <=2D ---
    if x1.ndim <= 2 and x2.ndim <= 2:
        out2d = _matmul_2d(_promote_1d_to_2d(x1), _promote_1d_to_2d(x2))
        x1_impl = x1._impl  # pylint: disable=W0212
        if x1.ndim == x2.ndim == 2:
            rowsA = ak.num(x1_impl, axis=0)
            colsB = safe_max_num(x2, axis=-1)
            if rowsA == 1 and colsB == 1:
                return array([out2d[0][0]])
        return array(out2d)

    # --- batch (>2D) ---
    def _batches(xt: array) -> list[list[int | float]]:
        xt_impl: Any = xt._impl  # pylint: disable=W0212
        if xt.ndim <= 2:
            return [ak.to_list(xt_impl)]
        return [ak.to_list(xt_impl[i]) for i in range(len(xt_impl))]

    x1_batches = _batches(x1)
    x2_batches = _batches(x2)

    B1, B2 = len(x1_batches), len(x2_batches)
    if B1 not in (B2, 1) and B2 not in (B1, 1):
        msg = "Broadcast dimension mismatch"
        raise ValueError(msg)

    n_batches = max(B1, B2)
    results = []
    for i in range(n_batches):
        b1 = x1_batches[i % B1]
        b2 = x2_batches[i % B2]
        results.append(
            _matmul_2d(_promote_1d_to_2d(array(b1)), _promote_1d_to_2d(array(b2)))
        )
    return array(results)


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
    xarray = x._impl  # pylint: disable=protected-access
    if not hasattr(xarray, "ndim") or xarray.ndim < 2:
        msg = "Input must have at least 2 dimensions"
        raise ValueError(msg)

    if not is_sorted_descending_all_levels(x):
        message = "Ragged dimension's lists must be sorted from longest to shortest, which is the only way that makes left-aligned ragged transposition possible."
        raise ValueError(message)

    nested: list[Any] = ak.to_list(xarray)

    def transpose_matrix(
        mat: list[list[float | int]],
    ) -> list[list[float | int]]:
        max_cols = max((len(row) for row in mat), default=0)
        return [[row[i] for row in mat if i < len(row)] for i in range(max_cols)]

    def is_matrix_level(b: list[Any]) -> bool:
        for row in b:
            if (isinstance(row, list) and row) and isinstance(row[0], (int, float)):
                return True
        return False

    def recurse(batch: list[Any]) -> list[Any]:
        if all(isinstance(b, list) for b in batch):
            if is_matrix_level(batch):
                return transpose_matrix(batch)
            return [recurse(b) for b in batch]
        return batch

    transposed = recurse(nested)
    return array(transposed)


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
