# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE
from __future__ import annotations

from typing import Any

import awkward as ak
import numpy as np
from awkward.contents import Content, ListArray, ListOffsetArray

from . import _spec_array_object


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


def is_effectively_regular(x: _spec_array_object.array) -> bool:
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
