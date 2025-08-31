# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/manipulation_functions.html
"""

from __future__ import annotations

from typing import cast

import awkward as ak
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
        assert a.shape == (*x.shape[:axis], 1, *x.shape[axis:])
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


def test_flip_none():
    arr = ragged.array(
        [[[1.1, 2.2, 3.3], []], [[4.4]], [], [[5.5, 6.6, 7.7, 8.8], [9.9]]]
    )
    arr_flipped = ragged.array(
        [[[9.9], [8.8, 7.7, 6.6, 5.5]], [], [[4.4]], [[], [3.3, 2.2, 1.1]]]
    )
    assert ak.to_list(ragged.flip(arr)) == ak.to_list(arr_flipped)


def test_flip_zero():
    arr = ragged.array(
        [[[1.1, 2.2, 3.3], []], [[4.4]], [], [[5.5, 6.6, 7.7, 8.8], [9.9]]]
    )
    arr_flipped = ragged.array(
        [[[5.5, 6.6, 7.7, 8.8], [9.9]], [], [[4.4]], [[1.1, 2.2, 3.3], []]]
    )
    assert ak.to_list(ragged.flip(arr, axis=0)) == ak.to_list(arr_flipped)


def test_flip_minus_two():
    arr = ragged.array(
        [[[1.1, 2.2, 3.3], []], [[4.4]], [], [[5.5, 6.6, 7.7, 8.8], [9.9]]]
    )
    arr_flipped = ragged.array(
        [[[], [1.1, 2.2, 3.3]], [[4.4]], [], [[9.9], [5.5, 6.6, 7.7, 8.8]]]
    )
    assert ak.to_list(ragged.flip(arr, axis=-2)) == ak.to_list(arr_flipped)
    assert ak.to_list(ragged.flip(arr, axis=1)) == ak.to_list(ragged.flip(arr, axis=-2))
    arr_type = cast(ak.Array, arr._impl).type
    flipped_type = cast(ak.Array, ragged.flip(arr)._impl).type
    assert arr_type == flipped_type


def test_flip_tuple():
    arr = ragged.array(
        [[[1.1, 2.2, 3.3], []], [[4.4]], [], [[5.5, 6.6, 7.7, 8.8], [9.9]]]
    )
    flipped = ragged.array(
        [[[9.9], [5.5, 6.6, 7.7, 8.8]], [], [[4.4]], [[], [1.1, 2.2, 3.3]]]
    )
    assert ak.to_list(ragged.flip(arr, axis=(0, 1))) == ak.to_list(flipped)


def test_flip_empty_tuple():
    arr = ragged.array(
        [[[1.1, 2.2, 3.3], []], [[4.4]], [], [[5.5, 6.6, 7.7, 8.8], [9.9]]]
    )
    result = ragged.flip(arr, axis=())
    assert ak.to_list(result) == ak.to_list(arr)


def test_flip_outofboundary():
    arr = ragged.array(
        [[[1.1, 2.2, 3.3], []], [[4.4]], [], [[5.5, 6.6, 7.7, 8.8], [9.9]]]
    )
    with pytest.raises(ValueError, match="axis"):
        ragged.flip(arr, axis=5)
    with pytest.raises(ValueError, match="axis"):
        ragged.flip(arr, axis=-5)


def test_stack_axis0():
    x = ragged.array([[1, 2], [3]])
    y = ragged.array([[4, 5], [6]])
    result = ragged.stack([x, y], axis=0)
    expected = ragged.array([[[1, 2], [3]], [[4, 5], [6]]])
    assert result.tolist() == expected.tolist()


def test_stack_axis1():
    x = ragged.array([[1, 2], [3]])
    y = ragged.array([[4, 5], [6]])
    result = ragged.stack([x, y], axis=1)
    expected = ragged.array([[[1, 2], [4, 5]], [[3], [6]]])
    assert ak.to_list(result) == ak.to_list(expected)


def test_stack_axis_minus1():
    x = ragged.array([[1, 2], [3]])
    y = ragged.array([[4, 5], [6]])

    result = ragged.stack([x, y], axis=-1)
    expected = ragged.array([[[1, 2], [4, 5]], [[3], [6]]])
    assert result.tolist() == expected.tolist()


def test_stack_invalid_axis_raises():
    x = ragged.array([[1, 2]])
    print("ndim", x.ndim)
    with pytest.raises(ValueError, match="axis=2 is out of bounds"):
        ragged.stack([x, x], axis=2)
    with pytest.raises(ValueError, match="axis=-3 is out of bounds"):
        ragged.stack([x, x], axis=-3)


def test_roll_basic_1d():
    a = ragged.array([1, 2, 3, 4])
    assert ragged.roll(a, 1, axis=0).tolist() == [4, 1, 2, 3]
    assert ragged.roll(a, -1, axis=0).tolist() == [2, 3, 4, 1]
    assert ragged.roll(a, 0, axis=0).tolist() == [1, 2, 3, 4]


def test_roll_axis_none_flatten_restore():
    a = ragged.array([[1, 2], [3, 4, 5]])
    rolled = ragged.roll(a, 2, axis=None)
    assert ak.to_list(rolled) == [[4, 5], [1, 2, 3]]


def test_roll_multi_axis_tuple_shift():
    a = ragged.array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ]
    )
    rolled = ragged.roll(a, (1, -1), axis=(0, 1))
    expected = [[[7, 8], [5, 6]], [[3, 4], [1, 2]]]
    assert ak.to_list(rolled) == expected


def test_roll_scalar_shift_with_tuple_axis():
    a = ragged.array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ]
    )
    rolled = ragged.roll(a, 1, axis=(0, 1))
    expected = [[[7, 8], [5, 6]], [[3, 4], [1, 2]]]
    assert ak.to_list(rolled) == expected


def test_roll_negative_axis():
    a = ragged.array([[1, 2], [3, 4]])
    assert ak.to_list(ragged.roll(a, 1, axis=-1)) == [[2, 1], [4, 3]]


def test_roll_large_shift():
    a = ragged.array([1, 2, 3])
    assert ak.to_list(ragged.roll(a, 10, axis=0)) == [3, 1, 2]  # 10 % 3 == 1


def test_roll_empty_inner_lists():
    a = ragged.array([[], [1, 2], []])
    rolled = ragged.roll(a, 1, axis=0)
    assert all(isinstance(lst, list) for lst in ak.to_list(rolled))


def test_roll_preserves_dtype_and_shape():
    a = ragged.array([[1.0, 2.0], [3.0, 4.0]])
    out = ragged.roll(a, 1, axis=0)
    assert out.dtype == a.dtype
    assert out.shape == a.shape


def test_roll_axis_none_ragged():
    a = ragged.array([[1], [2, 3], [4]])
    rolled = ragged.roll(a, 2, axis=None)
    assert ak.to_list(rolled) == [[3], [4, 1], [2]]


def test_roll_invalid_axis_type():
    a = ragged.array([1, 2, 3])
    with pytest.raises(TypeError):
        ragged.roll(a, 1, axis=1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ragged.roll(a, 1, axis=(0, "1"))  # type: ignore[arg-type]


def test_roll_invalid_shift_type():
    a = ragged.array([1, 2, 3])
    with pytest.raises(TypeError):
        ragged.roll(a, 2.0, axis=0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ragged.roll(a, (1, "2"), axis=(0, 0))  # type: ignore[arg-type]


def test_roll_shift_axis_length_mismatch():
    a = ragged.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="shift and axis must have the same length"):
        ragged.roll(a, (1, 2), axis=(0,))
    with pytest.raises(ValueError, match="shift and axis must have the same length"):
        ragged.roll(a, (1,), axis=(0, 1))


def test_roll_invalid_shift_type_message():
    a = ragged.array([1, 2, 3])
    # Passing a float shift
    with pytest.raises(
        TypeError, match=r"shift must be int or tuple of ints, got <class 'float'>"
    ):
        ragged.roll(a, 2.0, axis=0)  # type: ignore[arg-type]
    # Passing a tuple with a non-int element
    with pytest.raises(
        TypeError, match=r"shift must be int or tuple of ints, got <class 'tuple'>"
    ):
        ragged.roll(a, (1, "2"), axis=(0, 1))  # type: ignore[arg-type]
