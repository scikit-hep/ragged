# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/manipulation_functions.html
"""

from __future__ import annotations

import awkward as ak
import pytest

import ragged
from ragged._spec_manipulation_functions import broadcast_to

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


def test_concat(x, y):
    if x.ndim != y.ndim:
        with pytest.raises(ValueError, match="same number of dimensions"):
            ragged.concat([x, y])

    elif x.ndim == 0:
        with pytest.raises(ValueError, match="zero-dimensional"):
            ragged.concat([x, y])

    elif x.ndim == 1:
        assert ragged.concat([x, y], axis=None).tolist() == x.tolist() + y.tolist()
        assert ragged.concat([x, y], axis=0).tolist() == x.tolist() + y.tolist()

    else:
        assert ragged.concat([x, y], axis=None).tolist() == [
            1.1,
            2.2,
            3.3,
            4.4,
            5.5,
            1.1,
            2.2,
            3.3,
            4.4,
            5.5,
        ]
        assert ragged.concat([x, y], axis=0).tolist() == [  # type: ignore[comparison-overlap]
            [1.1, 2.2, 3.3],
            [],
            [4.4, 5.5],
            [1.1, 2.2, 3.3],
            [],
            [4.4, 5.5],
        ]
        assert ragged.concat([x, y], axis=1).tolist() == [  # type: ignore[comparison-overlap]
            [1.1, 2.2, 3.3, 1.1, 2.2, 3.3],
            [],
            [4.4, 5.5, 4.4, 5.5],
        ]


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_expand_dims(x, axis):
    if 0 <= axis <= x.ndim:
        a = ragged.expand_dims(x, axis=axis)
        assert a.shape == x.shape[:axis] + (1,) + x.shape[axis:]
        assert str(a._impl.type) == " * ".join(  # type: ignore[union-attr]
            ["var" if ai is None else str(ai) for ai in a.shape] + [str(a.dtype)]
        )

    else:
        with pytest.raises(ak.errors.AxisError):
            ragged.expand_dims(x, axis=axis)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_squeeze(x, axis):
    if 0 <= axis <= x.ndim:
        a = ragged.expand_dims(x, axis=axis)
        b = ragged.squeeze(a, axis=axis)
        assert b.shape == x.shape
        assert b.tolist() == x.tolist()


def test_positional_vs_keyword_arguments():
    x = ragged.array([1, 2, 3])
    # shape must be positional-only, copy must be keyword
    with pytest.raises(TypeError):
        broadcast_to(x, (3, 3), True)  # extra positional instead of keyword


def test_invalid_shape_type():
    x = ragged.array([1, 2, 3])
    with pytest.raises(TypeError):
        broadcast_to(x, [3, 3])  # list, not tuple


def test_shape_contains_non_int():
    x = ragged.array([1, 2])
    with pytest.raises(TypeError):
        broadcast_to(x, (3, "3"))  # invalid


def test_shape_contains_negative_other_than_minus_one():
    x = ragged.array([1])
    with pytest.raises(ValueError, match="Shape dimensions must be >= -1"):
        broadcast_to(x, (-2,))  # -1 is the only allowed negative


def test_broadcast_scalar_raises():
    x = 10
    with pytest.raises(ValueError, match="does not support scalar inputs"):
        broadcast_to(x, (2, 3))


def test_broadcast_zero_dim_array_raises():
    x = ragged.array(42)
    with pytest.raises(ValueError, match="does not support 0-dimensional arrays"):
        broadcast_to(x, ())


def test_broadcast_incompatible_shape():
    x = ragged.array([1, 2, 3])
    with pytest.raises(ValueError, match="Cannot broadcast array of shape"):
        broadcast_to(x, (2, 2))  # cannot broadcast


def test_shape_must_be_tuple_of_ints():
    x = ragged.array([1])
    with pytest.raises(TypeError):
        broadcast_to(x, (3.0,))  # float not allowed


def test_shape_must_not_be_empty_tuple_for_non_scalar():
    x = ragged.array([1, 2])
    with pytest.raises(ValueError, match="Shape must be a tuple of ints"):
        broadcast_to(x, ())  # empty tuple → scalar, mismatch


def test_positional_only_for_x_and_shape():
    from inspect import signature

    sig = signature(broadcast_to)
    assert list(sig.parameters.keys())[:2] == ["x", "shape"]
