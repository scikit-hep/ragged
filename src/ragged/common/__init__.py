# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

from __future__ import annotations

import awkward as ak

from ._typing import Device, Dtype, NestedSequence, SupportsDLPack


class array:
    def __init__(
        self,
        array_like: (
            array
            | ak.Array
            | SupportsDLPack
            | bool
            | int
            | float
            | NestedSequence[bool | int | float]
        ),
        dtype: None | Dtype = None,
        device: None | Device = None,
    ):
        ...
