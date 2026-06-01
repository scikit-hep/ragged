# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/linear_algebra_functions.html
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import awkward as ak
import numpy as np

if TYPE_CHECKING:
    import ragged as rg


def safe_max_num(arr: rg.array, axis: int | None = None) -> int:
    """
    Compute the maximum number of elements along an axis for a ragged array.
    Returns as an int, even if ak.num returns a scalar-like array.
    """
    counts: int | np.ndarray | list[int] = ak.num(arr, axis=axis)

    if isinstance(counts, int | np.integer):
        return int(counts)

    return int(max(counts))


def matmul(x1: rg.array, x2: rg.array, /) -> rg.array:
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
    import ragged as rg

    if x1.ndim == 0 or x2.ndim == 0:
        msg = "Zero-dimensional arrays are not allowed"
        raise ValueError(msg)

    # --- helper: promote 1D to 2D (list-of-rows) ---
    def _promote_1d_to_2d(arr: rg.array) -> list[list[int | float | complex]]:
        impl_list = ak.to_list(arr._impl)  # pylint: disable=W0212

        if arr.ndim == 1:
            return [impl_list]

        return [
            list(row)
            if isinstance(row, Iterable) and not isinstance(row, str | bytes)
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
        return rg.array(sum(a * b for a, b in zip(x1_list, x2_list, strict=False)))

    # --- 1D x 2D / 2D x 1D promotion ---
    if x1.ndim == 1 and x2.ndim == 2:
        x1_list = ak.to_list(x1._impl)  # pylint: disable=W0212
        promoted_x1 = rg.array([x1_list])  # shape (1, K)
        res2d = matmul(promoted_x1, x2)
        return rg.array(res2d[0])  # drop leading 1 safely

    if x1.ndim == 2 and x2.ndim == 1:
        x2_list = ak.to_list(x2._impl)  # pylint: disable=W0212
        promoted_x2 = rg.array([[v] for v in x2_list])  # shape (K, 1)
        res2d = matmul(x1, promoted_x2)  # returns 2D (M,1)

        # --- collapse trailing dimension safely ---
        if res2d.ndim == 2:
            impl_list = ak.to_list(res2d._impl)  # pylint: disable=W0212
            if len(impl_list) > 0:
                return rg.array([row[0] for row in impl_list])
            else:
                return rg.array([])  # empty 2D
        else:
            return rg.array(ak.to_list(res2d._impl))  # pylint: disable=W0212

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
                return rg.array([out2d[0][0]])
        return rg.array(out2d)

    # --- batch (>2D) ---
    def _batches(xt: rg.array) -> list[list[int | float]]:
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
            _matmul_2d(_promote_1d_to_2d(rg.array(b1)), _promote_1d_to_2d(rg.array(b2)))
        )
    return rg.array(results)


def matrix_transpose(x: rg.array, /) -> rg.array:
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

    # 1. Validation
    if not hasattr(xarray, "ndim") or xarray.ndim < 2:
        msg = "Per Array API, input array must not have fewer than 2 dimensions to have a matrix transpose property."
        raise ValueError(msg)

    counts = ak.num(xarray, axis=-1)
    # pylint: disable=unsubscriptable-object
    if not ak.all(counts[..., :-1] >= counts[..., 1:]):
        msg = (
            "Ragged dimension's lists must be sorted from longest to shortest, "
            "which is the only way that makes left-aligned ragged transposition possible."
        )
        raise ValueError(msg)

    # 2. Extract column indices and flatten the innermost matrix dimensions
    col_indices = ak.local_index(xarray, axis=-1)

    # Flattening at axis=-1 merges the M and N dims into a single flat sequence per matrix.
    # It preserves ALL outer batch dimensions intact.
    flat_data: ak.Array = ak.flatten(xarray, axis=-1)
    flat_cols: ak.Array = ak.flatten(col_indices, axis=-1)

    # 3. Stable sort by column index to group elements by their new row index
    # Because it's a stable sort, it naturally preserves the original row order!
    sorter = ak.argsort(flat_cols, axis=-1, stable=True)
    pivoted_flat = flat_data[sorter]

    # 4. Calculate new row lengths
    # Since sorted_cols is ascending (e.g., 0, 0, 1, 1, 2), its run lengths
    # give us the EXACT sizes of the new transposed rows.
    sorted_cols = flat_cols[sorter]
    new_row_lengths = ak.run_lengths(sorted_cols)
    # pylint: enable=unsubscriptable-object

    # 5. Structure Reconstruction
    # Flatten counts to 1D. Awkward's unflatten with axis=-1 will sequentially
    # consume this 1D array of counts, rebuilding the matrices perfectly.
    counts_1d = ak.to_numpy(ak.flatten(new_row_lengths, axis=None))
    transposed_ak = ak.unflatten(pivoted_flat, counts_1d, axis=-1)

    import ragged as rg

    return rg.array(transposed_ak)


def tensordot(
    x1: rg.array,
    x2: rg.array,
    /,
    *,
    axes: int | tuple[Sequence[int], Sequence[int]] = 2,
) -> rg.array:
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


def vecdot(x1: rg.array, x2: rg.array, /, *, axis: int = -1) -> rg.array:
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
