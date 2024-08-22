# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE
from __future__ import annotations

import numpy as np


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
