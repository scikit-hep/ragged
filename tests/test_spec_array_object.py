# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/array_object.html
"""

from __future__ import annotations

import numpy as np
import pytest

import ragged

devices = ["cpu"]
try:
    import cupy as cp

    devices.append("cuda")
except ModuleNotFoundError:
    cp = None


def test_existence():
    assert ragged.array is not None


def test_item():
    a = ragged.array(np.asarray(123)).item()
    assert isinstance(a, int)
    assert a == 123

    a = ragged.array(np.asarray([123])).item()
    assert isinstance(a, int)
    assert a == 123

    a = ragged.array(np.asarray([[123]])).item()
    assert isinstance(a, int)
    assert a == 123


def test_contains():
    a = ragged.array([[1, 2, 3], [], [4, 5]])
    assert 4 in a
    assert 6 not in a

    b = a[0, 0]
    assert 1 in b
    assert 2 not in b


def test_len():
    assert len(ragged.array([1, 2, 3])) == 3
    with pytest.raises(TypeError, match="unsized object"):
        len(ragged.array(123))


def test_iter():
    a = list(ragged.array([1, 2, 3]))
    assert isinstance(a[0], ragged.array)
    assert isinstance(a[1], ragged.array)
    assert isinstance(a[2], ragged.array)
    assert a[0] == 1
    assert a[1] == 2
    assert a[2] == 3

    b = list(ragged.array([[1], [2, 3]]))
    assert isinstance(b[0], ragged.array)
    assert isinstance(b[1], ragged.array)
    assert b[0].tolist() == [1]
    assert b[1].tolist() == [2, 3]

    with pytest.raises(TypeError, match="0-d array"):
        list(ragged.array(123))


def test_namespace():
    assert ragged.array(123).__array_namespace__() is ragged
    assert (
        ragged.array(123).__array_namespace__(api_version=ragged.__array_api_version__)
        is ragged
    )
    with pytest.raises(NotImplementedError):
        ragged.array(123).__array_namespace__(api_version="does not exist")


def test_bool():
    assert bool(ragged.array(True)) is True
    assert bool(ragged.array(False)) is False


def test_complex():
    assert isinstance(complex(ragged.array(1.1 + 0.1j)), complex)
    assert complex(ragged.array(1.1 + 0.1j)) == 1.1 + 0.1j


@pytest.mark.parametrize("device", devices)
def test_dlpack(device):
    lib = np if device == "cpu" else cp

    if not hasattr(lib, "from_dlpack"):
        return

    a = ragged.array(lib.arange(2 * 3 * 5).reshape(2, 3, 5), device=device)
    assert a.device == device
    assert isinstance(a._impl.layout.data, lib.ndarray)  # type: ignore[union-attr]

    b = lib.from_dlpack(a)
    assert isinstance(b, lib.ndarray)
    assert b.shape == a.shape
    assert b.dtype == a.dtype
    assert b.tolist() == a.tolist()

    a = ragged.array(lib.asarray(123), device=device)
    assert a.device == device
    assert isinstance(a._impl, lib.ndarray)

    b = lib.from_dlpack(a)
    assert isinstance(b, lib.ndarray)
    assert b.shape == a.shape
    assert b.dtype == a.dtype
    assert b.item() == a.item() == 123


def test_float():
    assert isinstance(float(ragged.array(1.1)), float)
    assert float(ragged.array(1.1)) == 1.1


def test_getitem():
    # slices are extensively tested in Awkward Array
    a = ragged.array([[1, 2, 3], [4], [5, 6, 7, 8]])
    assert a[..., 1:].tolist() == [[2, 3], [], [6, 7, 8]]  # type: ignore[comparison-overlap,index]


def test_index():
    assert isinstance(ragged.array(10).__index__(), int)
    assert ragged.array(10).__index__() == 10


def test_int():
    assert isinstance(int(ragged.array(10)), int)
    assert int(ragged.array(10)) == 10
