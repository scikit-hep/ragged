# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/data_type_functions.html
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._spec_array_object import _box, _unbox, array
from ._typing import Dtype

_type = type


def astype(x: array, dtype: Dtype, /, *, copy: bool = True) -> array:
    """
    Copies an array to a specified data type irrespective of type promotion rules.

    Args:
        x: Array to cast.
        dtype: Desired data type.
        copy: Ignored because `ragged.array` data buffers are immutable.

    Returns:
        An array having the specified data type. The returned array has the
        same `shape` as `x`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.astype.html
    """

    copy  # noqa: B018, argument is ignored, pylint: disable=W0104

    return _box(type(x), *_unbox(x), dtype=dtype)


def can_cast(from_: Dtype | array, to: Dtype, /) -> bool:
    """
    Determines if one data type can be cast to another data type according type
    promotion rules.

    Args:
        from: Input data type or array from which to cast.
        to: Desired data type.

    Returns:
        `True` if the cast can occur according to type promotion rules;
        otherwise, `False`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.can_cast.html
    """

    return bool(np.can_cast(from_, to))


@dataclass
class finfo_object:  # pylint: disable=C0103
    """
    Output of `ragged.finfo` with the following attributes.

    - bits (int): number of bits occupied by the real-valued floating-point
      data type.
    - eps (float): difference between 1.0 and the next smallest representable
      real-valued floating-point number larger than 1.0 according to the
      IEEE-754 standard.
    - max (float): largest representable real-valued number.
    - min (float): smallest representable real-valued number.
    - smallest_normal (float): smallest positive real-valued floating-point
      number with full precision.
    - dtype (np.dtype): real-valued floating-point data type.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.finfo.html
    """

    bits: int
    eps: float
    max: float
    min: float
    smallest_normal: float
    dtype: np.dtype


def finfo(type: Dtype | array, /) -> finfo_object:  # pylint: disable=W0622
    """
    Machine limits for floating-point data types.

    Args:
        type: the kind of floating-point data-type about which to get
            information. If complex, the information is about its component
            data type.

    Returns:
        An object having the following attributes:

        - bits (int): number of bits occupied by the real-valued floating-point
          data type.
        - eps (float): difference between 1.0 and the next smallest
          representable real-valued floating-point number larger than 1.0
          according to the IEEE-754 standard.
        - max (float): largest representable real-valued number.
        - min (float): smallest representable real-valued number.
        - smallest_normal (float): smallest positive real-valued floating-point
          number with full precision.
        - dtype (np.dtype): real-valued floating-point data type.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.finfo.html
    """

    if not isinstance(type, np.dtype):
        if not isinstance(type, _type) and hasattr(type, "dtype"):
            out = np.finfo(type.dtype)
        else:
            out = np.finfo(np.dtype(type))
    else:
        out = np.finfo(type)
    return finfo_object(
        out.bits, out.eps, out.max, out.min, out.smallest_normal, out.dtype
    )


@dataclass
class iinfo_object:  # pylint: disable=C0103
    """
    Output of `ragged.iinfo` with the following attributes.

    - bits (int): number of bits occupied by the type.
    - max (int): largest representable number.
    - min (int): smallest representable number.
    - dtype (np.dtype): integer data type.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.iinfo.html
    """

    bits: int
    max: int
    min: int
    dtype: np.dtype


def iinfo(type: Dtype | array, /) -> iinfo_object:  # pylint: disable=W0622
    """
    Machine limits for integer data types.

    Args:
        type: The kind of integer data-type about which to get information.

    Returns:
        An object having the following attributes:

        - bits (int): number of bits occupied by the type.
        - max (int): largest representable number.
        - min (int): smallest representable number.
        - dtype (np.dtype): integer data type.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.iinfo.html
    """

    if not isinstance(type, np.dtype):
        if not isinstance(type, _type) and hasattr(type, "dtype"):
            out = np.iinfo(type.dtype)
        else:
            out = np.iinfo(np.dtype(type))
    else:
        out = np.iinfo(type)
    return iinfo_object(out.bits, out.max, out.min, out.dtype)


def isdtype(dtype: Dtype, kind: Dtype | str | tuple[Dtype | str, ...]) -> bool:
    """
    Returns a boolean indicating whether a provided dtype is of a specified data type "kind".

    Args:
        dtype: The input dtype.
        kind: Data type kind.
            If `kind` is a `dtype`, the function returns a boolean indicating
            whether the input `dtype` is equal to the dtype specified by `kind`.

            If `kind` is a string, the function returns a boolean indicating
            whether the input `dtype` is of a specified data type kind. The
            following dtype kinds must be supported:

                - `"bool"`: boolean data types (e.g., bool).
                - `"signed integer"`: signed integer data types (e.g., `int8`,
                  `int16`, `int32`, `int64`).
                - `"unsigned integer"`: unsigned integer data types (e.g.,
                  `uint8`, `uint16`, `uint32`, `uint64`).
                - `"integral"`: integer data types. Shorthand for
                  (`"signed integer"`, `"unsigned integer"`).
                - `"real floating"`: real-valued floating-point data types
                  (e.g., `float32`, `float64`).
                - `"complex floating"`: complex floating-point data types
                  (e.g., `complex64`, `complex128`).
                - `"numeric"`: numeric data types. Shorthand for (`"integral"`,
                  `"real floating"`, `"complex floating"`).

            If `kind` is a tuple, the tuple specifies a union of dtypes and/or
            kinds, and the function returns a boolean indicating whether the
            input `dtype` is either equal to a specified dtype or belongs to at
            least one specified data type kind.

    Returns:
        Boolean indicating whether a provided dtype is of a specified data type
        kind.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.isdtype.html
    """

    _BUILTIN_MAP: dict[type, str] = {
        bool: "bool",
        int: "signed integer",
        float: "real floating",
        complex: "complex floating",
        str: "str",
    }
    _STRING_TO_CHARS: dict[str, set[str]] = {
        "bool": {"b"},
        "signed integer": {"i"},
        "unsigned integer": {"u"},
        "integral": {"i", "u"},
        "real floating": {"f"},
        "complex floating": {"c"},
        "numeric": {"i", "u", "f", "c"},
        "str": {"string"},
        "string": {"string"},
    }

    if isinstance(kind, tuple):
        return any(isdtype(dtype, k) for k in kind)

    # Normalise Python builtins -> string kind name
    if kind in _BUILTIN_MAP:
        kind = _BUILTIN_MAP[kind]  # type: ignore[index]

    # ------------------------------------------------------------------
    # Try to interpret dtype as a plain numpy dtype first.  This covers
    # the common case (np.dtype objects, numpy scalar types) without any
    # awkward-array type introspection.
    # ------------------------------------------------------------------
    np_dtype: np.dtype[Any] | None
    try:
        dtype_any: Any = dtype
        np_dtype = np.dtype(dtype_any)
    except (TypeError, ValueError):
        np_dtype = None

    if np_dtype is not None:
        actual_kind = np_dtype.kind  # one of 'b','i','u','f','c','U','S',…
        if isinstance(kind, str):
            accepted = _STRING_TO_CHARS.get(kind.lower())
            if accepted is None:
                return False
            return bool(actual_kind in accepted)
        # kind is a dtype
        try:
            return bool(np_dtype == np.dtype(kind))
        except (TypeError, ValueError):
            return False

    # ------------------------------------------------------------------
    # dtype is an awkward-array type object.  Walk .content to the leaf.
    # ------------------------------------------------------------------
    parameters: dict[str, Any] = {}
    current: object = dtype
    while hasattr(current, "content"):
        if hasattr(current, "parameters") and current.parameters:
            parameters = current.parameters
        current = current.content
    if hasattr(current, "parameters") and current.parameters:
        parameters = current.parameters

    if parameters.get("__array__") in {"string", "char"}:
        # String array: only matches "str" / "string" kind strings
        if isinstance(kind, str):
            return kind.lower() in {"str", "string"}
        return False

    primitive = getattr(current, "primitive", None)
    if primitive is None:
        return False
    try:
        leaf_dtype = np.dtype(primitive)
    except (TypeError, ValueError):
        return False

    actual_kind = leaf_dtype.kind
    if isinstance(kind, str):
        accepted = _STRING_TO_CHARS.get(kind.lower())
        if accepted is None:
            return False
        return actual_kind in accepted
    # kind is a dtype
    try:
        return bool(leaf_dtype == np.dtype(kind))
    except (TypeError, ValueError):
        return False


def result_type(*arrays_and_dtypes: array | Dtype) -> Dtype:
    """
    Returns the dtype that results from applying the type promotion rules to
    the arguments.

    Args:
        arrays_and_dtypes: An arbitrary number of input arrays and/or dtypes.

    Returns:
        The dtype resulting from an operation involving the input arrays and dtypes.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.result_type.html
    """

    return np.result_type(*arrays_and_dtypes)
