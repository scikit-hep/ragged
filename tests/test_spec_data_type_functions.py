# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/data_type_functions.html
"""

from __future__ import annotations

import numpy as np

import ragged


def test_existence():
    assert ragged.astype is not None
    assert ragged.can_cast is not None
    assert ragged.finfo is not None
    assert ragged.iinfo is not None
    assert ragged.isdtype is not None
    assert ragged.result_type is not None


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
