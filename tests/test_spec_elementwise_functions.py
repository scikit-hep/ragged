# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/elementwise_functions.html
"""

from __future__ import annotations

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


@pytest.fixture(params=["regular", "irregular", "scalar"])
def x(request):
    if request.param == "regular":
        return ragged.array(np.array([10, 20, 30]))
    elif request.param == "irregular":
        return ragged.array(ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]))
    else:  # request.param == "scalar"
        return ragged.array(np.array(100))


@pytest.fixture(params=["regular", "irregular", "scalar"])
def y(request):
    if request.param == "regular":
        return ragged.array(np.array([10, 20, 30]))
    elif request.param == "irregular":
        return ragged.array(ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]))
    else:  # request.param == "scalar"
        return ragged.array(np.array(100))


def test_existence():
    assert ragged.abs is not None
    assert ragged.acos is not None
    assert ragged.acosh is not None
    assert ragged.add is not None
    assert ragged.asin is not None
    assert ragged.asinh is not None
    assert ragged.atan is not None
    assert ragged.atan2 is not None
    assert ragged.atanh is not None
    assert ragged.bitwise_and is not None
    assert ragged.bitwise_invert is not None
    assert ragged.bitwise_left_shift is not None
    assert ragged.bitwise_or is not None
    assert ragged.bitwise_right_shift is not None
    assert ragged.bitwise_xor is not None
    assert ragged.ceil is not None
    assert ragged.conj is not None
    assert ragged.cos is not None
    assert ragged.cosh is not None
    assert ragged.divide is not None
    assert ragged.equal is not None
    assert ragged.exp is not None
    assert ragged.expm1 is not None
    assert ragged.floor is not None
    assert ragged.floor_divide is not None
    assert ragged.greater is not None
    assert ragged.greater_equal is not None
    assert ragged.imag is not None
    assert ragged.isfinite is not None
    assert ragged.isinf is not None
    assert ragged.isnan is not None
    assert ragged.less is not None
    assert ragged.less_equal is not None
    assert ragged.log is not None
    assert ragged.log1p is not None
    assert ragged.log2 is not None
    assert ragged.log10 is not None
    assert ragged.logaddexp is not None
    assert ragged.logical_and is not None
    assert ragged.logical_not is not None
    assert ragged.logical_or is not None
    assert ragged.logical_xor is not None
    assert ragged.multiply is not None
    assert ragged.negative is not None
    assert ragged.not_equal is not None
    assert ragged.positive is not None
    assert ragged.pow is not None
    assert ragged.real is not None
    assert ragged.remainder is not None
    assert ragged.round is not None
    assert ragged.sign is not None
    assert ragged.sin is not None
    assert ragged.sinh is not None
    assert ragged.square is not None
    assert ragged.sqrt is not None
    assert ragged.subtract is not None
    assert ragged.tan is not None
    assert ragged.tanh is not None
    assert ragged.trunc is not None


@pytest.mark.parametrize("device", devices)
def test_add(device, x, y):
    result = ragged.add(x.to_device(device), y.to_device(device))
    assert type(result) is type(x) is type(y)
    assert result.shape in (x.shape, y.shape)
    assert result.dtype in (x.dtype, y.dtype)
