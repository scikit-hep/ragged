# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

from __future__ import annotations

import numbers

import awkward as ak
import numpy as np
from awkward.contents import (
    Content,
    ListArray,
    ListOffsetArray,
    NumpyArray,
    RegularArray,
)

from . import _import
from ._typing import Device, Dtype, NestedSequence, Shape, SupportsDLPack


def _shape_dtype(layout: Content) -> tuple[Shape, Dtype]:
    node = layout
    shape: Shape = (len(layout),)
    while isinstance(node, (ListArray, ListOffsetArray, RegularArray)):
        if isinstance(node, RegularArray):
            shape = (*shape, node.size)
        else:
            shape = (*shape, None)
        node = node.content

    if isinstance(node, NumpyArray):
        shape = shape + node.data.shape[1:]
        return shape, node.data.dtype

    msg = f"Awkward Array type must have regular and irregular lists only, not {layout.form.type!s}"
    raise TypeError(msg)


class array:  # pylint: disable=C0103
    """
    Ragged array class and constructor.
    """

    _impl: ak.Array | SupportsDLPack
    _shape: Shape
    _dtype: Dtype
    _device: Device

    @classmethod
    def _new(cls, impl: ak.Array, shape: Shape, dtype: Dtype, device: Device) -> array:
        out = cls.__new__(cls)
        out._impl = impl
        out._shape = shape
        out._dtype = dtype
        out._device = device
        return out

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
        if isinstance(array_like, array):
            self._impl = array_like._impl
            self._shape, self._dtype = array_like._shape, array_like._dtype

        elif isinstance(array_like, ak.Array):
            self._impl = array_like
            self._shape, self._dtype = _shape_dtype(self._impl.layout)

        elif isinstance(array_like, (bool, numbers.Real)):
            self._impl = np.array(array_like)
            self._shape, self._dtype = (), self._impl.dtype

        else:
            self._impl = ak.Array(array_like)
            self._shape, self._dtype = _shape_dtype(self._impl.layout)

        if dtype is not None and dtype != self._dtype:
            if isinstance(self._impl, ak.Array):
                self._impl = ak.values_astype(self._impl, dtype)
                self._shape, self._dtype = _shape_dtype(self._impl.layout)
            else:
                self._impl = np.array(array_like, dtype=dtype)
                self._dtype = dtype

        if device is not None:
            if isinstance(self._impl, ak.Array) and device != ak.backend(self._impl):
                self._impl = ak.to_backend(self._impl, device)
            elif isinstance(self._impl, np.ndarray) and device == "cuda":
                cp = _import.cupy()
                self._impl = cp.array(self._impl.item())

    def __str__(self) -> str:
        if len(self._shape) == 0:
            return f"{self._impl.item()}"
        elif len(self._shape) == 1:
            return f"{ak._prettyprint.valuestr(self._impl, 1, 80)}"
        else:
            prep = ak._prettyprint.valuestr(self._impl, 20, 80 - 4)[1:-1].replace(
                "\n ", "\n    "
            )
            return f"[\n    {prep}\n]"

    def __repr__(self) -> str:
        if len(self._shape) == 0:
            return f"ragged.array({self._impl.item()})"
        elif len(self._shape) == 1:
            return f"ragged.array({ak._prettyprint.valuestr(self._impl, 1, 80 - 14)})"
        else:
            prep = ak._prettyprint.valuestr(self._impl, 20, 80 - 4)[1:-1].replace(
                "\n ", "\n    "
            )
            return f"ragged.array([\n    {prep}\n])"

    @property
    def dtype(self) -> Dtype:
        return self._dtype

    @property
    def device(self) -> Device:
        return self._device

    @property
    def mT(self) -> array:
        raise RuntimeError()

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def shape(self) -> Shape:
        return self._shape

    @property
    def size(self) -> None | int:
        if len(self._shape) == 0:
            return 1
        else:
            return int(ak.count(self._impl))

    @property
    def T(self) -> array:
        raise RuntimeError()
