# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

from __future__ import annotations

import enum
from numbers import Real
from typing import TYPE_CHECKING, Any, Union

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
from ._typing import (
    Device,
    Dtype,
    NestedSequence,
    PyCapsule,
    Shape,
    SupportsDLPack,
)


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


# https://github.com/python/typing/issues/684#issuecomment-548203158
if TYPE_CHECKING:
    from enum import Enum

    class ellipsis(Enum):  # pylint: disable=C0103
        Ellipsis = "..."  # pylint: disable=C0103

    Ellipsis = ellipsis.Ellipsis  # pylint: disable=W0622

else:
    ellipsis = type(...)  # pylint: disable=C0103

GetSliceKey = Union[
    int,
    slice,
    ellipsis,
    None,
    tuple[Union[int, slice, ellipsis, None], ...],
    "array",
]

SetSliceKey = Union[
    int, slice, ellipsis, tuple[Union[int, slice, ellipsis], ...], "array"
]


class array:  # pylint: disable=C0103
    """
    Ragged array class and constructor.

    https://data-apis.org/array-api/latest/API_specification/array_object.html
    """

    # Constructors, internal functions, and other methods that are unbound by
    # the Array API specification.

    _impl: ak.Array | SupportsDLPack  # ndim > 0 ak.Array or ndim == 0 NumPy or CuPy
    _shape: Shape
    _dtype: Dtype
    _device: Device

    @classmethod
    def _new(cls, impl: ak.Array, shape: Shape, dtype: Dtype, device: Device) -> array:
        """
        Simple/fast array constructor for internal code.
        """

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
        dtype: None | Dtype | type | str = None,
        device: None | Device = None,
    ):
        """
        Primary array constructor.

        Args:
            array_like: Data to use as or convert into a ragged array.
            dtype: NumPy dtype describing the data (subclass of `np.number`,
                without `shape` or `fields`).
            device: If `"cpu"`, the array is backed by NumPy and resides in
                main memory; if `"cuda"`, the array is backed by CuPy and
                resides in CUDA global memory.
        """

        if isinstance(array_like, array):
            self._impl = array_like._impl
            self._shape, self._dtype = array_like._shape, array_like._dtype

        elif isinstance(array_like, ak.Array):
            self._impl = array_like
            self._shape, self._dtype = _shape_dtype(self._impl.layout)

        elif isinstance(array_like, (bool, Real)):
            self._impl = np.array(array_like)
            self._shape, self._dtype = (), self._impl.dtype

        else:
            self._impl = ak.Array(array_like)
            self._shape, self._dtype = _shape_dtype(self._impl.layout)

        if not isinstance(dtype, np.dtype):
            dtype = np.dtype(dtype)

        if dtype is not None and dtype != self._dtype:
            if isinstance(self._impl, ak.Array):
                self._impl = ak.values_astype(self._impl, dtype)
                self._shape, self._dtype = _shape_dtype(self._impl.layout)
            else:
                self._impl = np.array(array_like, dtype=dtype)
                self._dtype = dtype

        if self._dtype.fields is not None:
            msg = f"dtype must not have fields: dtype.fields = {self._dtype.fields}"
            raise TypeError(msg)

        if self._dtype.shape != ():
            msg = f"dtype must not have a shape: dtype.shape = {self._dtype.shape}"
            raise TypeError(msg)

        if not issubclass(self._dtype.type, np.number):
            msg = f"dtype must be numeric: dtype.type = {self._dtype.type}"
            raise TypeError(msg)

        if device is not None:
            if isinstance(self._impl, ak.Array) and device != ak.backend(self._impl):
                self._impl = ak.to_backend(self._impl, device)
            elif isinstance(self._impl, np.ndarray) and device == "cuda":
                cp = _import.cupy()
                self._impl = cp.array(self._impl.item())

    def __str__(self) -> str:
        """
        String representation of the array.
        """

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
        """
        REPL-string representation of the array.
        """

        if len(self._shape) == 0:
            return f"ragged.array({self._impl.item()})"
        elif len(self._shape) == 1:
            return f"ragged.array({ak._prettyprint.valuestr(self._impl, 1, 80 - 14)})"
        else:
            prep = ak._prettyprint.valuestr(self._impl, 20, 80 - 4)[1:-1].replace(
                "\n ", "\n    "
            )
            return f"ragged.array([\n    {prep}\n])"

    # Attributes: https://data-apis.org/array-api/latest/API_specification/array_object.html#attributes

    @property
    def dtype(self) -> Dtype:
        """
        Data type of the array elements.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.dtype.html
        """

        return self._dtype

    @property
    def device(self) -> Device:
        """
        Hardware device the array data resides on.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.device.html
        """

        return self._device

    @property
    def mT(self) -> array:
        """
        Transpose of a matrix (or a stack of matrices).

        Raises:
            ValueError: If any ragged dimension's lists are not sorted from longest
                to shortest, which is the only way that left-aligned ragged
                transposition is possible.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.mT.html
        """

        msg = "not implemented yet, but will be"
        raise RuntimeError(msg)

    @property
    def ndim(self) -> int:
        """
        Number of array dimensions (axes).

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.ndim.html
        """

        return len(self._shape)

    @property
    def shape(self) -> Shape:
        """
        Array dimensions.

        Regular dimensions are represented by `int` values in the `shape` and
        irregular (ragged) dimensions are represented by `None`.

        According to the specification, "An array dimension must be `None` if
        and only if a dimension is unknown," which is a different
        interpretation than we are making here.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.shape.html
        """

        return self._shape

    @property
    def size(self) -> None | int:
        """
        Number of elements in an array.

        This property never returns `None` because we do not consider
        dimensions to be unknown, and numerical values within ragged
        lists can be counted.

        Example:
            An array like `ragged.array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])` has
            a size of 5 because it contains 5 numerical values.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.size.html
        """

        if len(self._shape) == 0:
            return 1
        else:
            return int(ak.count(self._impl))

    @property
    def T(self) -> array:
        """
        Transpose of the array.

        Raises:
            ValueError: If any ragged dimension's lists are not sorted from longest
                to shortest, which is the only way that left-aligned ragged
                transposition is possible.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.T.html
        """

        msg = "not implemented yet, but will be"
        raise RuntimeError(msg)

    # methods: https://data-apis.org/array-api/latest/API_specification/array_object.html#methods

    def __abs__(self) -> array:
        """
        Calculates the absolute value for each element of an array instance.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__abs__.html
        """

        msg = "not implemented yet, but will be"
        raise RuntimeError(msg)

    def __add__(self, other: int | float | array, /) -> array:
        """
        Calculates the sum for each element of an array instance with the
        respective element of the array other.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__add__.html
        """

        msg = "not implemented yet, but will be"
        raise RuntimeError(msg)

    def __and__(self, other: int | bool | array, /) -> array:
        """
        Evaluates `self_i & other_i` for each element of an array instance with
        the respective element of the array other.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__and__.html
        """

        msg = "not implemented yet, but will be"
        raise RuntimeError(msg)

    def __array_namespace__(self, *, api_version: None | str = None) -> Any:
        """
        Returns an object that has all the array API functions on it.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__array_namespace__.html
        """

        assert api_version is None, "FIXME"

        msg = "not implemented yet, but will be"
        raise RuntimeError(msg)

    def __bool__(self) -> bool:  # FIXME pylint: disable=E0304
        """
        Converts a zero-dimensional array to a Python `bool` object.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__bool__.html
        """

        msg = "not implemented yet, but will be"
        raise RuntimeError(msg)

    def __complex__(self) -> complex:
        """
        Converts a zero-dimensional array to a Python `complex` object.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__complex__.html
        """

        msg = "not implemented yet, but will be"
        raise RuntimeError(msg)

    def __dlpack__(self, *, stream: None | int | Any = None) -> PyCapsule:
        """
        Exports the array for consumption by `from_dlpack()` as a DLPack
        capsule.

        Args:
            stream: CuPy Stream object (https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.Stream.html)
                if not `None`.

        Raises:
            ValueError: If any dimensions are ragged.

            https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack__.html
        """

        assert stream is None, "FIXME"

        msg = "not implemented yet, but will be"
        raise RuntimeError(msg)

    def __dlpack_device__(self) -> tuple[enum.Enum, int]:
        """
        Returns device type and device ID in DLPack format.

        Raises:
            ValueError: If any dimensions are ragged.

            https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack_device__.html
        """

        msg = "not implemented yet, but will be"
        raise RuntimeError(msg)

    def __eq__(self, other: int | float | bool | array, /) -> array:  # type: ignore[override]
        """
        Computes the truth value of `self_i == other_i` for each element of an
        array instance with the respective element of the array other.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__eq__.html
        """

        msg = "not implemented yet, but will be"
        raise RuntimeError(msg)

    def __float__(self) -> float:
        """
        Converts a zero-dimensional array to a Python `float` object.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__float__.html
        """

        msg = "not implemented yet, but will be"
        raise RuntimeError(msg)

    def __floordiv__(self, other: int | float | array, /) -> array:
        """
        Evaluates `self_i // other_i` for each element of an array instance
        with the respective element of the array other.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__floordiv__.html
        """

        msg = "not implemented yet, but will be"
        raise RuntimeError(msg)

    def __ge__(self, other: int | float | array, /) -> array:
        """
        Computes the truth value of `self_i >= other_i` for each element of an
        array instance with the respective element of the array other.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__ge__.html
        """

        msg = "not implemented yet, but will be"
        raise RuntimeError(msg)

    def __getitem__(self, key: GetSliceKey, /) -> array:
        """
        Returns self[key].

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__getitem__.html
        """

        msg = "not implemented yet, but will be"
        raise RuntimeError(msg)

    def __gt__(self, other: int | float | array, /) -> array:
        """
        Computes the truth value of `self_i > other_i` for each element of an
        array instance with the respective element of the array other.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__gt__.html
        """

        msg = "not implemented yet, but will be"
        raise RuntimeError(msg)

    def __index__(self) -> int:  # FIXME pylint: disable=E0305
        """
        Converts a zero-dimensional integer array to a Python `int` object.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__index__.html
        """

        msg = "not implemented yet, but will be"
        raise RuntimeError(msg)

    def __int__(self) -> int:
        """
        Converts a zero-dimensional array to a Python `int` object.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__int__.html
        """

        msg = "not implemented yet, but will be"
        raise RuntimeError(msg)

    def __invert__(self) -> array:
        """
        Evaluates `~self_i` for each element of an array instance.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__invert__.html
        """

        msg = "not implemented yet, but will be"
        raise RuntimeError(msg)

    def __le__(self, other: int | float | array, /) -> array:
        """
        Computes the truth value of `self_i <= other_i` for each element of an
        array instance with the respective element of the array other.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__le__.html
        """

        msg = "not implemented yet, but will be"
        raise RuntimeError(msg)

    def __lshift__(self, other: int | array, /) -> array:
        """
        Evaluates `self_i << other_i` for each element of an array instance
        with the respective element of the array other.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__lshift__.html
        """

        msg = "not implemented yet, but will be"
        raise RuntimeError(msg)

    def __lt__(self, other: int | float | array, /) -> array:
        """
        Computes the truth value of `self_i < other_i` for each element of an
        array instance with the respective element of the array other.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__lt__.html
        """

        msg = "not implemented yet, but will be"
        raise RuntimeError(msg)

    def __matmul__(self, other: array, /) -> array:
        """
        Computes the matrix product.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__matmul__.html
        """

        msg = "not implemented yet, but will be"
        raise RuntimeError(msg)

    def __mod__(self, other: int | float | array, /) -> array:
        """
        Evaluates `self_i % other_i` for each element of an array instance with
        the respective element of the array other.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__mod__.html
        """

        msg = "not implemented yet, but will be"
        raise RuntimeError(msg)

    def __mul__(self, other: int | float | array, /) -> array:
        """
        Calculates the product for each element of an array instance with the
        respective element of the array other.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__mul__.html
        """

        msg = "not implemented yet, but will be"
        raise RuntimeError(msg)

    def __ne__(self, other: int | float | bool | array, /) -> array:  # type: ignore[override]
        """
        Computes the truth value of `self_i != other_i` for each element of an
        array instance with the respective element of the array other.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__ne__.html
        """

        msg = "not implemented yet, but will be"
        raise RuntimeError(msg)

    def __neg__(self) -> array:
        """
        Evaluates `-self_i` for each element of an array instance.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__neg__.html
        """

        msg = "not implemented yet, but will be"
        raise RuntimeError(msg)

    def __or__(self, other: int | bool | array, /) -> array:
        """
        Evaluates `self_i | other_i` for each element of an array instance with
        the respective element of the array other.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__or__.html
        """

        msg = "not implemented yet, but will be"
        raise RuntimeError(msg)

    def __pos__(self) -> array:
        """
        Evaluates `+self_i` for each element of an array instance.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__pos__.html
        """

        msg = "not implemented yet, but will be"
        raise RuntimeError(msg)

    def __pow__(self, other: int | float | array, /) -> array:
        """
        Calculates an implementation-dependent approximation of exponentiation
        by raising each element (the base) of an array instance to the power of
        `other_i` (the exponent), where `other_i` is the corresponding element
        of the array `other`.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__pow__.html
        """

        msg = "not implemented yet, but will be"
        raise RuntimeError(msg)

    def __rshift__(self, other: int | array, /) -> array:
        """
        Evaluates `self_i >> other_i` for each element of an array instance
        with the respective element of the array other.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__rshift__.html
        """

        msg = "not implemented yet, but will be"
        raise RuntimeError(msg)

    def __setitem__(
        self, key: SetSliceKey, value: int | float | bool | array, /
    ) -> None:
        """
        Sets `self[key]` to value.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__setitem__.html
        """

        msg = "not implemented yet, but will be"
        raise RuntimeError(msg)

    def __sub__(self, other: int | float | array, /) -> array:
        """
        Calculates the difference for each element of an array instance with
        the respective element of the array other.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__sub__.html
        """

        msg = "not implemented yet, but will be"
        raise RuntimeError(msg)

    def __truediv__(self, other: int | float | array, /) -> array:
        """
        Evaluates `self_i / other_i` for each element of an array instance with
        the respective element of the array other.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__truediv__.html
        """

        msg = "not implemented yet, but will be"
        raise RuntimeError(msg)

    def __xor__(self, other: int | bool | array, /) -> array:
        """
        Evaluates `self_i ^ other_i` for each element of an array instance with
        the respective element of the array other.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__xor__.html
        """

        msg = "not implemented yet, but will be"
        raise RuntimeError(msg)

    def to_device(self, device: Device, /, *, stream: None | int | Any = None) -> array:
        """
        Copy the array from the device on which it currently resides to the
        specified device.

        Args:
            device: If `"cpu"`, the array is backed by NumPy and resides in
                main memory; if `"cuda"`, the array is backed by CuPy and
                resides in CUDA global memory.
            stream: CuPy Stream object (https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.Stream.html)
                for `device="cuda"`.

        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.to_device.html
        """

        if isinstance(self._impl, ak.Array) and device != ak.backend(self._impl):
            assert stream is None, "FIXME: use CuPy stream"
            impl = ak.to_backend(self._impl, device)

        elif isinstance(self._impl, np.ndarray):
            if device == "cuda":
                assert stream is None, "FIXME: use CuPy stream"
                cp = _import.cupy()
                impl = cp.array(self._impl.item())
            else:
                impl = self._impl

        else:
            impl = np.array(self._impl.item()) if device == "cpu" else self._impl

        return self._new(impl, self._shape, self._dtype, device)
