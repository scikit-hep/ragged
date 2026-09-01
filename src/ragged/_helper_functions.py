# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
Internal helper/utility functions shared across the ragged package.

These are not part of the public Array API surface; they handle dtype
normalisation and structural regularity checks that several spec modules
need in common.
"""

from __future__ import annotations

from typing import Any

import awkward as ak
import numpy as np
from awkward.contents import Content, ListArray, ListOffsetArray

from ._spec_array_object import array


def regularise_to_float(t: np.dtype, /) -> np.dtype:
    """
    Promote an integer or boolean dtype to a suitable floating-point dtype.

    This is a compatibility shim for NumPy 2.0.0, which removed implicit
    integer-to-float promotion in certain operations.  For NumPy 2.1 and
    later the function is a no-op, because that version restored compatible
    behaviour.

    The promotion rules mirror those of NumPy's own type-promotion ladder:

    * ``bool``, ``int8``, ``uint8``   → ``float16``
    * ``int16``, ``uint16``           → ``float32``
    * ``int32``, ``uint32``, ``int64``, ``uint64`` → ``float64``
    * All other dtypes (e.g. floating-point) are returned unchanged.

    Args:
        t: The NumPy dtype to (potentially) promote.

    Returns:
        A floating-point NumPy dtype that can represent values of ``t``,
        or ``t`` itself when no promotion is required.
    """
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
    """
    Return ``True`` if *x* behaves like a rectangular (non-ragged) array.

    Unlike :func:`is_regular_or_effectively_regular`, this function does
    **not** inspect the Awkward Array layout; it instead walks the Python
    object directly using ``len`` and iteration.  This makes it useful as a
    fallback when the layout-based check is unavailable or inconclusive.

    A 1-D sequence is considered effectively regular if every element has the
    same ``len``.  A 2-D (or higher) sequence is considered effectively regular
    if every outer entry has the same length *and* every inner entry of those
    outer entries has the same length.

    Args:
        x: The array (or array-like object) to inspect.

    Returns:
        ``True`` if all rows (and, for 3-D inputs, all sub-rows) have uniform
        length; ``False`` if any dimension is ragged, or if the object does not
        support ``len`` / iteration at all.
    """
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
    """
    Return ``True`` if *x* is backed by a regular (rectangular) layout or
    behaves like one at the Python level.

    The check is performed in two stages:

    1. **Layout-based (fast path):** If *x* exposes an Awkward Array layout
       via ``x._impl.layout``, the function accepts *x* as regular when the
       outermost layout node is a :class:`ak.contents.RegularArray` whose
       content is either a :class:`ak.contents.NumpyArray` (2-D) or another
       ``RegularArray`` wrapping a ``NumpyArray`` (3-D).
    2. **Iteration-based (fallback):** If the layout check raises
       ``TypeError`` or ``AttributeError`` (e.g. the object is not an
       Awkward-backed ``ragged.array``), the function delegates to
       :func:`is_effectively_regular`.

    Args:
        x: Any object to inspect — typically a :class:`ragged.array`, but
           plain Python sequences and NumPy arrays are also accepted (they
           fall through to the iteration-based check).

    Returns:
        ``True`` if *x* is structurally regular; ``False`` otherwise.
    """
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
