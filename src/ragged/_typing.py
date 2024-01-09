# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
Borrows liberally from https://github.com/numpy/numpy/blob/main/numpy/array_api/_typing.py
"""

from __future__ import annotations

import enum
import sys
from typing import Any, Literal, Optional, Protocol, TypeVar, Union

import numpy as np

T_co = TypeVar("T_co", covariant=True)


if sys.version_info >= (3, 12):
    from collections.abc import (  # pylint: disable=W0611
        Buffer as SupportsBufferProtocol,
    )
else:
    SupportsBufferProtocol = Any


# not actually checked because of https://github.com/python/typing/discussions/1145
class NestedSequence(Protocol[T_co]):
    def __getitem__(self, key: int, /) -> T_co | NestedSequence[T_co]:
        ...

    def __len__(self, /) -> int:
        ...


PyCapsule = Any


class SupportsDLPack(Protocol):
    def __dlpack__(self, /, *, stream: None = ...) -> PyCapsule:
        ...

    def __dlpack_device__(self, /) -> tuple[enum.Enum, int]:
        ...


Shape = tuple[Optional[int], ...]

Dtype = np.dtype[
    Union[
        np.bool_,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    ]
]

numeric_types = (
    np.bool_,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
)

Device = Literal["cpu", "cuda"]
