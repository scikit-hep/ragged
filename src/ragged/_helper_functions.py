# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE
from __future__ import annotations

from typing import Any

import awkward as ak
import numpy as np
from awkward.contents import Content, ListArray, ListOffsetArray

from ._spec_array_object import array


def uniform_axis_size(x: array, axis: int) -> int | None:
    """
    Return the size of ``x`` along ``axis`` if it is uniform across every
    entry, otherwise ``None`` (the axis is ragged / non-uniform).

    A ``ragged.array`` always stores its variable-length axes internally, so
    ``shape[axis]`` may be ``None`` even when every list happens to have the
    same length.  This walks the actual element counts with ``ak.num`` and
    checks whether they all agree, using vectorised ``ak.min``/``ak.max``
    rather than materialising Python lists.

    Returns ``None`` for an empty selection (no counts to compare), matching the
    behaviour of the call sites this consolidates.
    """
    counts: Any = ak.num(x._impl, axis=axis)  # pylint: disable=protected-access

    # ``ak.num`` with ``axis=0`` returns a 0-d NumPy scalar (the outer length),
    # which is always uniform; an int can come back for the same reason.
    if not isinstance(counts, ak.Array):
        return int(counts)

    lo = ak.min(counts, axis=None)
    hi = ak.max(counts, axis=None)
    # Empty selection -> ak.min/ak.max return None: treat as non-uniform.
    if lo is None or hi is None:
        return None
    if int(lo) != int(hi):
        return None
    return int(lo)


def regularise_to_float(t: np.dtype, /) -> np.dtype:
    # Ensure compatibility with numpy 2.0.0
    if np.__version__ >= "2.1":
        # Just pass and return the input type if the numpy version is not 2.0.0
        return t

    if t in [np.int8, np.uint8, np.bool_, bool]:
        return np.float16
    elif t in [np.int16, np.uint16]:
        return np.float32
    elif t in [np.int32, np.uint32, np.int64, np.uint64]:
        return np.float64
    else:
        return t


def is_sorted_descending_all_levels(x: array, /) -> bool:
    """
    Checks whether all nested lists in the array are sorted by descending length
    at every level of the array (ignoring leaves).

    Returns:
        bool: True if all nested lists are sorted descending by length, False otherwise.
    """
    array_ak = ak.Array(x._impl)  # pylint: disable=protected-access
    layout: Content = ak.to_layout(array_ak)

    def check(node: Content) -> bool:
        if isinstance(node, ListOffsetArray | ListArray):
            lengths: ak.Array = ak.num(node, axis=1)
            if not ak.all(lengths[:-1] >= lengths[1:]):  # pylint: disable=E1136
                return False
            return check(node.content)
        else:
            return True

    return check(layout)


def is_effectively_regular(x: array) -> bool:
    try:
        if not hasattr(x, "__len__"):
            return False

        if all(hasattr(row, "__len__") for row in x):
            row_length = len(x[0])
            return all(len(row) == row_length for row in x)

        for batch in x:
            if not hasattr(batch, "__len__"):
                return False
            if not all(hasattr(row, "__len__") for row in batch):
                return False

        outer_len = len(x[0])
        inner_len = len(x[0][0])

        for batch in x:
            if len(batch) != outer_len:
                return False
            for row in batch:
                if len(row) != inner_len:
                    return False

        return True
    except (TypeError, AttributeError, IndexError):
        return False


def is_regular_or_effectively_regular(x: Any) -> bool:
    try:
        layout = x.layout
        layout = x._impl.layout  # pylint: disable=W0212
        if isinstance(layout, ak.contents.RegularArray) and (
            isinstance(layout.content, ak.contents.NumpyArray)
            or (
                isinstance(layout.content, ak.contents.RegularArray)
                and isinstance(layout.content.content, ak.contents.NumpyArray)
            )
        ):
            return True
    except (TypeError, AttributeError):
        pass

    return is_effectively_regular(x)
