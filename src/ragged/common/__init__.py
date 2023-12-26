# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

from __future__ import annotations

import awkward as ak

from ._typing import Device, Dtype, NestedSequence, SupportsDLPack


class array:  # pylint: disable=C0103
    """
    Ragged array class and constructor.
    """

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
        assert array_like is not None
        assert dtype is None
        assert device is None
