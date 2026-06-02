# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
CF Conventions ragged-array I/O helpers.

The `CF Conventions <https://cfconventions.org/>`_ (Climate and Forecast)
define two standard ways to store variable-length (ragged) arrays in NetCDF
files.  Both representations flatten a 2-D ragged array into a pair of 1-D
arrays so that every value is stored in a simple contiguous buffer.

Contiguous ragged array (CF §9.3.3 / Appendix H.3.1)
    All values are concatenated into a single *content* array.  A companion
    *counts* (``row_size``) array records how many values belong to each row.

    Example::

        [[1, 2, 3], [4, 5], [6]]
        -> content = [1, 2, 3, 4, 5, 6]
           counts  = [3, 2, 1]

Indexed ragged array (CF §9.3.3 / Appendix H.3.2)
    All values are concatenated into a single *content* array.  A companion
    *index* (``parentIndex``) array records which row each value belongs to.
    Values belonging to the same row need not be contiguous.

    Example::

        [[1, 2, 3], [4, 5], [6]]
        -> content = [1, 2, 3, 4, 5, 6]
           index   = [0, 0, 0, 1, 1, 2]

Both encodings are restricted to **2-D** ragged arrays; higher-dimensional
inputs are not supported by the CF standard and will raise ``ValueError``.
"""

from __future__ import annotations

import awkward as ak
import numpy as np

from .._import import device_namespace
from .._spec_array_object import _box, _unbox, array


def _require_2d_ragged(x: array, name: str) -> None:
    """Raise ValueError if *x* is not a 2-D ragged.array."""
    if not isinstance(x, array):
        msg = f"{name}: expected a ragged.array, got {type(x).__name__!r}"
        raise TypeError(msg)
    if x.ndim != 2:
        msg = (
            f"{name}: input must be a 2-D ragged array (ndim=2), "
            f"got ndim={x.ndim}. "
            "CF Conventions define contiguous/indexed encoding only for "
            "2-D ragged arrays."
        )
        raise ValueError(msg)


def _require_1d(arr: array, param: str, func: str) -> None:
    """Raise ValueError if *arr* is not 1-D."""
    if not isinstance(arr, array):
        msg = f"{func}: {param!r} must be a ragged.array, got {type(arr).__name__!r}"
        raise TypeError(msg)
    if arr.ndim != 1:
        msg = f"{func}: {param!r} must be a 1-D array (ndim=1), " f"got ndim={arr.ndim}"
        raise ValueError(msg)


def to_cf_contiguous(x: array) -> tuple[array, array]:
    """
    Encode a 2-D ragged array as a CF contiguous ragged array.

    The returned pair ``(content, counts)`` satisfies::

        from_cf_contiguous(content, counts) == x

    Args:
        x: A 2-D ragged array.  Each row may have a different number of
            elements.

    Returns:
        content: 1-D array of all values, in row-major order.
        counts: 1-D integer array whose ``i``-th element gives the number of
            values in row ``i`` of ``x``.  Equivalent to the CF
            ``row_size`` variable.

    Raises:
        TypeError: If *x* is not a ``ragged.array``.
        ValueError: If *x* does not have exactly 2 dimensions.

    Examples:
        >>> import ragged
        >>> a = ragged.array([[1, 2, 3], [4, 5], [6]])
        >>> content, counts = ragged.io.to_cf_contiguous(a)
        >>> content.tolist()
        [1, 2, 3, 4, 5, 6]
        >>> counts.tolist()
        [3, 2, 1]
    """
    _require_2d_ragged(x, "to_cf_contiguous")

    (y,) = _unbox(x)
    return _box(type(x), ak.flatten(y)), _box(type(x), ak.num(y))


def from_cf_contiguous(content: array, counts: array) -> array:
    """
    Decode a CF contiguous ragged array into a 2-D ragged array.

    This is the inverse of :func:`to_cf_contiguous`::

        to_cf_contiguous(from_cf_contiguous(content, counts)) == (content, counts)

    Args:
        content: 1-D array of all values concatenated in row-major order.
        counts: 1-D non-negative integer array.  ``counts[i]`` is the
            number of values that belong to row ``i``.  The sum of *counts*
            must equal ``len(content)``.

    Returns:
        A 2-D ragged array whose ``i``-th row contains the ``counts[i]``
        values starting at the appropriate offset in *content*.

    Raises:
        TypeError: If either argument is not a ``ragged.array``.
        ValueError: If either argument is not 1-D, if any count is negative,
            or if ``sum(counts) != len(content)``.

    Examples:
        >>> import ragged, numpy as np
        >>> content = ragged.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
        >>> counts = ragged.array([3, 2, 1], dtype=np.int64)
        >>> ragged.io.from_cf_contiguous(content, counts).tolist()
        [[1, 2, 3], [4, 5], [6]]
    """
    _require_1d(content, "content", "from_cf_contiguous")
    _require_1d(counts, "counts", "from_cf_contiguous")

    counts_np = ak.to_numpy(_unbox(counts)[0])
    if np.any(counts_np < 0):
        msg = "from_cf_contiguous: all counts must be non-negative"
        raise ValueError(msg)
    total = int(counts_np.sum())
    if total != len(content):
        msg = (
            f"from_cf_contiguous: sum(counts)={total} does not match "
            f"len(content)={len(content)}"
        )
        raise ValueError(msg)

    cont, cnts = _unbox(content, counts)
    return _box(type(content), ak.unflatten(cont, cnts))


def to_cf_indexed(x: array) -> tuple[array, array]:
    """
    Encode a 2-D ragged array as a CF indexed ragged array.

    The returned pair ``(content, index)`` satisfies::

        from_cf_indexed(content, index) == x

    Args:
        x: A 2-D ragged array.

    Returns:
        content: 1-D array of all values, in row-major order.
        index: 1-D integer array of the same length as *content*.
            ``index[k]`` gives the row index that element ``content[k]``
            belongs to.  Equivalent to the CF ``parentIndex`` variable.

    Raises:
        TypeError: If *x* is not a ``ragged.array``.
        ValueError: If *x* does not have exactly 2 dimensions.

    Examples:
        >>> import ragged
        >>> a = ragged.array([[1, 2, 3], [4, 5], [6]])
        >>> content, index = ragged.io.to_cf_indexed(a)
        >>> content.tolist()
        [1, 2, 3, 4, 5, 6]
        >>> index.tolist()
        [0, 0, 0, 1, 1, 2]
    """
    _require_2d_ragged(x, "to_cf_indexed")

    _, ns = device_namespace(x.device)
    (y,) = _unbox(x)

    index, _ = ak.broadcast_arrays(ns.arange(len(x), dtype=ns.int64), y)
    return _box(type(x), ak.flatten(y)), _box(type(x), ak.flatten(index))


def from_cf_indexed(content: array, index: array) -> array:
    """
    Decode a CF indexed ragged array into a 2-D ragged array.

    This is the inverse of :func:`to_cf_indexed`.  Elements belonging to
    the same row need not be contiguous in *content*; they are gathered
    according to *index* and placed into ascending row order.  Within each
    row the original relative order of elements is preserved.

    Args:
        content: 1-D array of all values.
        index: 1-D non-negative integer array of the same length as
            *content*.  ``index[k]`` is the row that ``content[k]`` belongs
            to.

    Returns:
        A 2-D ragged array with ``max(index) + 1`` rows.  Row ``i`` contains
        all elements of *content* whose corresponding *index* value equals
        ``i``, in their original relative order.

    Raises:
        TypeError: If either argument is not a ``ragged.array``.
        ValueError: If either argument is not 1-D, if *index* contains
            negative values, or if ``len(content) != len(index)``.

    Examples:
        >>> import ragged, numpy as np
        >>> content = ragged.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
        >>> index = ragged.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
        >>> ragged.io.from_cf_indexed(content, index).tolist()
        [[1, 2, 3], [4, 5], [6]]
        >>> # Unsorted index: elements are assigned to rows in stated order
        >>> content2 = ragged.array([10, 20, 30, 40, 50], dtype=np.int64)
        >>> index2 = ragged.array([1, 0, 1, 0, 1], dtype=np.int64)
        >>> ragged.io.from_cf_indexed(content2, index2).tolist()
        [[20, 40], [10, 30, 50]]
    """
    _require_1d(content, "content", "from_cf_indexed")
    _require_1d(index, "index", "from_cf_indexed")

    if len(content) != len(index):
        msg = (
            f"from_cf_indexed: len(content)={len(content)} does not match "
            f"len(index)={len(index)}"
        )
        raise ValueError(msg)

    _, ns = device_namespace(content.device)
    cont, ind = _unbox(content, index)

    ind_np = ns.asarray(ind, dtype=ns.int64)
    if int(ind_np.min()) < 0 if len(ind_np) > 0 else False:
        msg = "from_cf_indexed: index values must be non-negative"
        raise ValueError(msg)

    n_rows = int(ind_np.max()) + 1 if len(ind_np) > 0 else 0

    # Count elements per row, preserving original order within each row.
    counts = ns.zeros(n_rows, dtype=ns.int64)
    ns.add.at(counts, ind_np, 1)

    # Use a stable sort on the index so that elements within each row
    # retain their original relative order.
    sorter = ns.argsort(ind_np, kind="stable")

    return _box(type(content), ak.unflatten(cont[sorter], counts))  # type: ignore[index]


__all__ = ["to_cf_contiguous", "from_cf_contiguous", "to_cf_indexed", "from_cf_indexed"]
