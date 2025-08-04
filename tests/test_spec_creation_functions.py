# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/creation_functions.html
"""

from __future__ import annotations

from typing import cast

import awkward as ak
import numpy as np
import pytest

import ragged
from ragged._helper_functions import (
    is_effectively_regular,
    is_regular_or_effectively_regular,
)

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
def test_empty_ndim0(device):
    a = ragged.empty((), device=device)
    assert a.ndim == 0
    assert a.shape == ()
    assert isinstance(a._impl, ns[device].ndarray)


@pytest.mark.parametrize("device", devices)
def test_empty_like(device):
    a = ragged.array([[1, 2, 3], [], [4, 5]], device=device)
    b = ragged.empty_like(a)
    assert (b * 0).tolist() == [[0, 0, 0], [], [0, 0]]  # type: ignore[comparison-overlap]
    assert a.dtype == b.dtype
    assert a.device == b.device == device


@pytest.mark.parametrize("device", devices)
def test_eye(device):
    a = ragged.eye(3, 5, k=1, device=device)
    assert a.tolist() == [[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]]
    assert isinstance(a._impl.layout.data, ns[device].ndarray)  # type: ignore[union-attr]


@pytest.mark.skipif(
    not hasattr(np, "from_dlpack"), reason=f"np.from_dlpack not in {np.__version__}"
)
@pytest.mark.parametrize("device", devices)
def test_from_dlpack(device):
    a = ns[device].array([1, 2, 3, 4, 5])
    b = ragged.from_dlpack(a)
    assert b.tolist() == [1, 2, 3, 4, 5]
    assert isinstance(b._impl.layout.data, ns[device].ndarray)  # type: ignore[union-attr]


@pytest.mark.parametrize("device", devices)
def test_full(device):
    a = ragged.full(5, 3, device=device)
    assert a.tolist() == [3, 3, 3, 3, 3]
    assert isinstance(a._impl.layout.data, ns[device].ndarray)  # type: ignore[union-attr]


@pytest.mark.parametrize("device", devices)
def test_full_ndim0(device):
    a = ragged.full((), 3, device=device)
    assert a.ndim == 0
    assert a.shape == ()
    assert a == 3
    assert isinstance(a._impl, ns[device].ndarray)


@pytest.mark.parametrize("device", devices)
def test_full_like(device):
    a = ragged.array([[1, 2, 3], [], [4, 5]], device=device)
    b = ragged.full_like(a, 5)
    assert b.tolist() == [[5, 5, 5], [], [5, 5]]  # type: ignore[comparison-overlap]
    assert a.dtype == b.dtype
    assert a.device == b.device == device


@pytest.mark.parametrize("device", devices)
def test_linspace(device):
    a = ragged.linspace(5, 8, 5, device=device)
    assert a.tolist() == [5, 5.75, 6.5, 7.25, 8]
    assert isinstance(a._impl.layout.data, ns[device].ndarray)  # type: ignore[union-attr]


@pytest.mark.parametrize("device", devices)
def test_ones(device):
    a = ragged.ones(5, device=device)
    assert a.tolist() == [1, 1, 1, 1, 1]
    assert isinstance(a._impl.layout.data, ns[device].ndarray)  # type: ignore[union-attr]


@pytest.mark.parametrize("device", devices)
def test_ones_ndim0(device):
    a = ragged.ones((), device=device)
    assert a.ndim == 0
    assert a.shape == ()
    assert a == 1
    assert isinstance(a._impl, ns[device].ndarray)


@pytest.mark.parametrize("device", devices)
def test_ones_like(device):
    a = ragged.array([[1, 2, 3], [], [4, 5]], device=device)
    b = ragged.ones_like(a)
    assert b.tolist() == [[1, 1, 1], [], [1, 1]]  # type: ignore[comparison-overlap]
    assert a.dtype == b.dtype
    assert a.device == b.device == device


@pytest.mark.parametrize("device", devices)
def test_zeros(device):
    a = ragged.zeros(5, device=device)
    assert a.tolist() == [0, 0, 0, 0, 0]
    assert isinstance(a._impl.layout.data, ns[device].ndarray)  # type: ignore[union-attr]


@pytest.mark.parametrize("device", devices)
def test_zeros_ndim0(device):
    a = ragged.zeros((), device=device)
    assert a.ndim == 0
    assert a.shape == ()
    assert a == 0
    assert isinstance(a._impl, ns[device].ndarray)


@pytest.mark.parametrize("device", devices)
def test_zeros_like(device):
    a = ragged.array([[1, 2, 3], [], [4, 5]], device=device)
    b = ragged.zeros_like(a)
    assert b.tolist() == [[0, 0, 0], [], [0, 0]]  # type: ignore[comparison-overlap]
    assert a.dtype == b.dtype
    assert a.device == b.device == device


def test_tril_output_shape_and_dtype():
    x = ragged.array(
        [[[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], [[7.7, 8.8, 9.9], [10.0, 11.1, 12.2]]]
    )
    result = ragged.tril(x)
    assert result.shape == x.shape
    assert result.dtype == x.dtype


def test_tril_dtype_and_device_consistency():
    x = ragged.array(
        [[[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], [[7.7, 8.8, 9.9], [10.0, 11.1, 12.2]]]
    )
    result = ragged.tril(x)
    assert isinstance(result._impl, type(x._impl))
    assert (
        np.asarray(ak.flatten(x, axis=None)).dtype
        == np.asarray(ak.flatten(result, axis=None)).dtype
    )
    x_layout = cast(ak.Array, x._impl).layout
    result_layout = cast(ak.Array, result._impl).layout
    assert x_layout.backend == result_layout.backend


def test_tril_numpy_equivalence():
    x = ragged.array(
        [[[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], [[7.7, 8.8, 9.9], [10.0, 11.1, 12.2]]]
    )
    y = np.array(
        [[[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], [[7.7, 8.8, 9.9], [10.0, 11.1, 12.2]]]
    )
    assert ak.to_list(ragged.tril(x)) == ak.to_list(np.tril(y))
    assert ak.to_list(ragged.tril(x, k=1)) == ak.to_list(np.tril(y, k=1))
    assert ak.to_list(ragged.tril(x, k=-1)) == ak.to_list(np.tril(y, k=-1))
    assert ragged.tril(x).dtype == np.tril(y).dtype


def test_is_effectively_regular_2d():
    x = ragged.array([[1, 2], [3, 4]])
    assert is_effectively_regular(x) is True


def test_is_effectively_regular_3d():
    x = ragged.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert is_effectively_regular(x) is True


def test_is_effectively_regular_irregular():
    x = ragged.array([[1], [2, 3]])
    assert is_effectively_regular(x) is False


def test_is_effectively_regular_empty():
    x = ragged.array([])
    assert is_effectively_regular(x) is False  # or True?


def test_is_effectively_regular_with_empty():
    x = ragged.array([[[1, 2], [3, 4]], [[5, 6], []]])
    assert is_effectively_regular(x) is True


def test_is_regular_backend_regular():
    x = ak.Array([[1.0, 2.0], [3.0, 4.0]])
    reg = ragged.array(x)
    assert is_regular_or_effectively_regular(reg) is True


def test_is_regular_backend_irregular():
    x = ak.Array([[1.0], [2.0, 3.0]])
    irreg = ragged.array(x)
    assert is_regular_or_effectively_regular(irreg) is False


def test_is_regular_falls_back_to_effectively_regular():
    nested = ragged.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert is_regular_or_effectively_regular(nested) is True
