# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

from __future__ import annotations

import numbers
from typing import Any, Literal, Optional, Protocol, TypeVar, Union

import numpy as np

T_co = TypeVar("T_co", covariant=True)


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

    def item(self) -> numbers.Number:
        ...


Shape = tuple[Optional[int], ...]

Dtype = np.dtype[
    Union[
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
    ]
]

Device = Union[Literal["cpu"], Literal["cuda"]]
