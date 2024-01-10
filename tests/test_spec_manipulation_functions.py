# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/manipulation_functions.html
"""

from __future__ import annotations

import pytest

import ragged

devices = ["cpu"]
try:
    import cupy as cp

    devices.append("cuda")
except ModuleNotFoundError:
    cp = None


def test_existence():
    assert ragged.broadcast_arrays is not None
    assert ragged.broadcast_to is not None
    assert ragged.concat is not None
    assert ragged.expand_dims is not None
    assert ragged.flip is not None
    assert ragged.permute_dims is not None
    assert ragged.reshape is not None
    assert ragged.roll is not None
    assert ragged.squeeze is not None
    assert ragged.stack is not None


@pytest.mark.parametrize("device", devices)
def test_broadcast_arrays(device, x, y):
    x_bc, y_bc = ragged.broadcast_arrays(x.to_device(device), y.to_device(device))
    if x.shape == () and y.shape == ():
        assert x_bc.shape == ()
        assert y_bc.shape == ()
    else:
        assert x_bc.shape == y_bc.shape
        if x_bc.shape == (3,):
            assert (x_bc * 0).tolist() == (y_bc * 0).tolist() == [0, 0, 0]
        if x_bc.shape == (3, None):
            assert (x_bc * 0).tolist() == (y_bc * 0).tolist() == [[0, 0, 0], [], [0, 0]]  # type: ignore[comparison-overlap]
