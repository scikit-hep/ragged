# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/data_type_functions.html
"""

from __future__ import annotations

from typing import Any

import awkward as ak
import numpy as np
import pytest

import ragged

devices = ["cpu"]
try:
    import cupy as cp

    devices.append("cuda")
except ModuleNotFoundError:
    cp = None


def first(x: ragged.array) -> Any:
    out = ak.flatten(x._impl, axis=None)[0] if x.shape != () else x._impl
    return np.asarray(out.item(), dtype=x.dtype)


def test_existence():
    assert ragged.astype is not None
    assert ragged.can_cast is not None
    assert ragged.finfo is not None
    assert ragged.iinfo is not None
    assert ragged.isdtype is not None
    assert ragged.result_type is not None


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dt", ["float64", np.float64, np.dtype(np.float64)])
def test_astype(device, x_int, dt):
    x = x_int.to_device(device)
    y = ragged.astype(x, dt)
    assert first(y) == first(x)
    assert y.dtype == np.dtype(np.float64)
    assert y.device == x.device


def test_can_cast():
    assert ragged.can_cast(np.float32, np.complex128)
    assert not ragged.can_cast(np.complex128, np.float32)


def test_finfo():
    f = ragged.finfo(np.float64)
    assert f.bits == 64
    assert f.eps == 2.220446049250313e-16
    assert f.max == 1.7976931348623157e308
    assert f.min == -1.7976931348623157e308
    assert f.smallest_normal == 2.2250738585072014e-308
    assert f.dtype == np.dtype(np.float64)


def test_finfo_array():
    f = ragged.finfo(np.array([1.1, 2.2, 3.3]))
    assert f.bits == 64
    assert f.dtype == np.dtype(np.float64)


def test_finfo_array2():
    f = ragged.finfo(ragged.array([1.1, 2.2, 3.3]))
    assert f.bits == 64
    assert f.dtype == np.dtype(np.float64)


def test_iinfo():
    f = ragged.iinfo(np.int16)
    assert f.bits == 16
    assert f.max == 32767
    assert f.min == -32768
    assert f.dtype == np.dtype(np.int16)


def test_iinfo_array():
    f = ragged.iinfo(np.array([1, 2, 3], np.int16))
    assert f.bits == 16
    assert f.dtype == np.dtype(np.int16)


def test_iinfo_array2():
    f = ragged.iinfo(ragged.array([1, 2, 3], np.int16))
    assert f.bits == 16
    assert f.dtype == np.dtype(np.int16)


def test_result_type():
    dt = ragged.result_type(ragged.array([1, 2, 3]), ragged.array([1.1, 2.2, 3.3]))
    assert dt == np.dtype(np.float64)


def test_isdtype_bool():
    x = ragged.array([[True, False]])
    dtype = x._impl.type
    assert ragged.isdtype(dtype, bool)
    assert ragged.isdtype(dtype, "bool")
    assert not ragged.isdtype(dtype, "numeric")
    assert not ragged.isdtype(dtype, "str")
    assert ragged.isdtype(dtype, (int, "bool"))


def test_isdtype_int():
    x = ragged.array([[1, 2], [3]])
    dtype = x._impl.type
    assert ragged.isdtype(dtype, int)
    assert ragged.isdtype(dtype, "signed integer")
    assert ragged.isdtype(dtype, "integral")
    assert ragged.isdtype(dtype, "numeric")
    assert not ragged.isdtype(dtype, "str")
    assert ragged.isdtype(dtype, (int, "int"))


def test_isdtype_float():
    x = ragged.array([[1.1, 2.2], [3.3]])
    dtype = x._impl.type
    assert ragged.isdtype(dtype, float)
    assert ragged.isdtype(dtype, "real floating")
    assert ragged.isdtype(dtype, "numeric")
    assert not ragged.isdtype(dtype, "integral")
    assert ragged.isdtype(dtype, (float, "real floating"))


def test_isdtype_complex():
    x = ragged.array([[1.1 + 4.4j, 2.2 - 7.7j], [3.3 + 9.9j]])
    dtype = x._impl.type
    assert ragged.isdtype(dtype, complex)
    assert ragged.isdtype(dtype, "complex floating")
    assert ragged.isdtype(dtype, "numeric")
    assert not ragged.isdtype(dtype, "integral")
    assert not ragged.isdtype(dtype, "real floating")
    assert ragged.isdtype(dtype, (complex, "complex floating"))


def test_isdtype_str():
    x = ragged.array([["one", "two"], ["three"]])
    dtype = x._impl.type
    assert ragged.isdtype(dtype, "str")
    assert ragged.isdtype(dtype, "string")
    assert not ragged.isdtype(dtype, "int")
    assert not ragged.isdtype(dtype, "bool")
    assert ragged.isdtype(dtype, (str, "str"))


def test_isdtype_unknown():
    x = ragged.array([[], []])
    dtype = x._impl.type
    assert not ragged.isdtype(dtype, bool)
    assert not ragged.isdtype(dtype, int)
    assert not ragged.isdtype(dtype, float)
    assert not ragged.isdtype(dtype, complex)
    assert not ragged.isdtype(dtype, "str")
    assert not ragged.isdtype(dtype, "real floating")
    assert not ragged.isdtype(dtype, "complex floating")
    assert not ragged.isdtype(dtype, "signed integer")
    assert not ragged.isdtype(dtype, "unsigned integer")
    assert not ragged.isdtype(dtype, "numeric")
    assert not ragged.isdtype(dtype, (str, "str"))
    assert not ragged.isdtype(dtype, (int, "bool"))
    assert not ragged.isdtype(dtype, (int, "integral"))
    assert not ragged.isdtype(dtype, (float, "real floating"))
    assert not ragged.isdtype(dtype, (complex, "complex floating"))
