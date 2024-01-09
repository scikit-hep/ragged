# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/creation_functions.html
"""

from __future__ import annotations

import numpy as np
import pytest

import ragged

devices = ["cpu"]
ns = {"cpu": np}
try:
    import cupy as cp

    devices.append("cuda")
    ns["cuda"] = cp
except ModuleNotFoundError:
    cp = None


def test_existence():
    assert ragged.arange is not None
    assert ragged.asarray is not None
    assert ragged.empty is not None
    assert ragged.empty_like is not None
    assert ragged.eye is not None
    assert ragged.from_dlpack is not None
    assert ragged.full is not None
    assert ragged.full_like is not None
    assert ragged.linspace is not None
    assert ragged.meshgrid is not None
    assert ragged.ones is not None
    assert ragged.ones_like is not None
    assert ragged.tril is not None
    assert ragged.triu is not None
    assert ragged.zeros is not None
    assert ragged.zeros_like is not None


@pytest.mark.parametrize("device", devices)
def test_arange(device):
    a = ragged.arange(5, 10, 2, device=device)
    assert a.tolist() == [5, 7, 9]
    assert isinstance(a._impl.layout.data, ns[device].ndarray)  # type: ignore[union-attr]


@pytest.mark.parametrize("device", devices)
def test_empty(device):
    a = ragged.empty((2, 3, 5), device=device)
    assert a.shape == (2, 3, 5)
    assert isinstance(a._impl.layout.data, ns[device].ndarray)  # type: ignore[union-attr]


@pytest.mark.parametrize("device", devices)
def test_eye(device):
    a = ragged.eye(3, 5, k=1, device=device)
    assert a.tolist() == [[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]]
    assert isinstance(a._impl.layout.data, ns[device].ndarray)  # type: ignore[union-attr]


@pytest.mark.parametrize("device", devices)
def test_from_dlpack(device):
    a = ns[device].array([1, 2, 3, 4, 5])
    b = ragged.from_dlpack(a)
    assert b.tolist() == [1, 2, 3, 4, 5]
    assert isinstance(b._impl.layout.data, ns[device].ndarray)  # type: ignore[union-attr]
