# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
Tests for ragged.roll.

Coverage
--------
1-D arrays
  - axis=None: same as axis=0 for 1-D
  - axis=0: positive shift, negative shift, shift > n, shift=0

2-D uniform arrays (fast path via numpy)
  - axis=0: row reorder
  - axis=1: element reorder within rows
  - axis=-1: same as axis=1
  - negative shift
  - shift > n (wraps)
  - tuple of axes with tuple of shifts
  - int shift with tuple of axes

2-D ragged arrays
  - axis=0: row reorder (O(1) slice path)
  - axis=1: inner roll per-row
  - axis=None: flatten, roll, restore structure
  - shift=0: identity
  - negative shift
  - empty rows handled
  - single-row array
  - shift wraps around (shift > row_len)

3-D uniform arrays (fast path)
  - axis=0, axis=1, axis=2

3-D ragged arrays (list fallback)
  - axis=None: flatten and restore nested structure

result type and dtype
  - always returns ragged.array
  - dtype preserved

error paths
  - shift tuple without axis tuple
  - shift tuple length != axis tuple length
  - axis out of range
"""

from __future__ import annotations

import numpy as np
import pytest

import ragged


def _make(nested, dtype=None) -> ragged.array:
    return ragged.array(nested, dtype=dtype)


def _np(x: ragged.array) -> np.ndarray:
    return np.array(x.tolist())


# ---------------------------------------------------------------------------
# 1-D arrays
# ---------------------------------------------------------------------------


class TestRoll1D:
    def test_axis_none_positive(self):
        a = _make([1, 2, 3, 4, 5], dtype=np.float64)
        result = ragged.roll(a, 2)
        np.testing.assert_array_equal(_np(result), [4, 5, 1, 2, 3])

    def test_axis_0_positive(self):
        a = _make([1, 2, 3, 4, 5], dtype=np.float64)
        result = ragged.roll(a, 2, axis=0)
        np.testing.assert_array_equal(_np(result), [4, 5, 1, 2, 3])

    def test_axis_0_negative(self):
        a = _make([1, 2, 3, 4, 5], dtype=np.float64)
        result = ragged.roll(a, -1, axis=0)
        np.testing.assert_array_equal(_np(result), [2, 3, 4, 5, 1])

    def test_shift_zero(self):
        a = _make([1, 2, 3], dtype=np.float64)
        result = ragged.roll(a, 0, axis=0)
        np.testing.assert_array_equal(_np(result), [1, 2, 3])

    def test_shift_larger_than_n(self):
        a = _make([1, 2, 3], dtype=np.float64)
        result = ragged.roll(a, 7, axis=0)  # 7 % 3 == 1
        np.testing.assert_array_equal(_np(result), [3, 1, 2])

    def test_result_is_ragged_array(self):
        a = _make([1.0, 2.0, 3.0])
        assert isinstance(ragged.roll(a, 1), ragged.array)

    def test_dtype_preserved(self):
        a = _make([1, 2, 3], dtype=np.float32)
        assert ragged.roll(a, 1).dtype == np.float32


# ---------------------------------------------------------------------------
# 2-D uniform arrays (fast path)
# ---------------------------------------------------------------------------


class TestRoll2DUniform:
    def test_axis_0(self):
        a_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        a = _make(a_np.tolist(), dtype=np.float64)
        result = ragged.roll(a, 1, axis=0)
        np.testing.assert_array_equal(_np(result), np.roll(a_np, 1, axis=0))

    def test_axis_1(self):
        a_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        a = _make(a_np.tolist(), dtype=np.float64)
        result = ragged.roll(a, 1, axis=1)
        np.testing.assert_array_equal(_np(result), np.roll(a_np, 1, axis=1))

    def test_negative_axis(self):
        a_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        a = _make(a_np.tolist(), dtype=np.float64)
        r_neg = ragged.roll(a, 1, axis=-1)
        r_pos = ragged.roll(a, 1, axis=1)
        np.testing.assert_array_equal(_np(r_neg), _np(r_pos))

    def test_negative_shift(self):
        a_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        a = _make(a_np.tolist(), dtype=np.float64)
        result = ragged.roll(a, -1, axis=1)
        np.testing.assert_array_equal(_np(result), np.roll(a_np, -1, axis=1))

    def test_shift_wraps(self):
        a_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        a = _make(a_np.tolist(), dtype=np.float64)
        result = ragged.roll(a, 7, axis=1)  # 7 % 3 == 1
        np.testing.assert_array_equal(_np(result), np.roll(a_np, 7, axis=1))

    def test_axis_none(self):
        a_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        a = _make(a_np.tolist(), dtype=np.float64)
        result = ragged.roll(a, 2)
        np.testing.assert_array_equal(_np(result), np.roll(a_np, 2))

    def test_tuple_axes_tuple_shifts(self):
        a_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        a = _make(a_np.tolist(), dtype=np.float64)
        result = ragged.roll(a, (1, 1), axis=(0, 1))
        expected = np.roll(np.roll(a_np, 1, axis=0), 1, axis=1)
        np.testing.assert_array_equal(_np(result), expected)

    def test_int_shift_tuple_axis(self):
        a_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        a = _make(a_np.tolist(), dtype=np.float64)
        result = ragged.roll(a, 1, axis=(0, 1))
        expected = np.roll(np.roll(a_np, 1, axis=0), 1, axis=1)
        np.testing.assert_array_equal(_np(result), expected)

    def test_dtype_preserved(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float32)
        assert ragged.roll(a, 1, axis=0).dtype == np.float32


# ---------------------------------------------------------------------------
# 2-D ragged arrays
# ---------------------------------------------------------------------------


class TestRoll2DRagged:
    def test_axis_0_shift_1(self):
        a = _make([[1, 2, 3], [4, 5], [6]], dtype=np.float64)
        result = ragged.roll(a, 1, axis=0)
        assert result.tolist() == [[6.0], [1.0, 2.0, 3.0], [4.0, 5.0]]

    def test_axis_0_shift_neg1(self):
        a = _make([[1, 2, 3], [4, 5], [6]], dtype=np.float64)
        result = ragged.roll(a, -1, axis=0)
        assert result.tolist() == [[4.0, 5.0], [6.0], [1.0, 2.0, 3.0]]

    def test_axis_0_shift_0(self):
        a = _make([[1, 2, 3], [4, 5], [6]], dtype=np.float64)
        result = ragged.roll(a, 0, axis=0)
        assert result.tolist() == a.tolist()

    def test_axis_1_shift_1(self):
        a = _make([[1, 2, 3], [4, 5], [6]], dtype=np.float64)
        result = ragged.roll(a, 1, axis=1)
        assert result.tolist() == [[3.0, 1.0, 2.0], [5.0, 4.0], [6.0]]

    def test_axis_1_shift_neg1(self):
        a = _make([[1, 2, 3], [4, 5], [6]], dtype=np.float64)
        result = ragged.roll(a, -1, axis=1)
        assert result.tolist() == [[2.0, 3.0, 1.0], [5.0, 4.0], [6.0]]

    def test_axis_1_negative_axis(self):
        a = _make([[1, 2, 3], [4, 5], [6]], dtype=np.float64)
        r_neg = ragged.roll(a, 1, axis=-1)
        r_pos = ragged.roll(a, 1, axis=1)
        assert r_neg.tolist() == r_pos.tolist()

    def test_axis_none(self):
        a = _make([[1, 2, 3], [4, 5], [6]], dtype=np.float64)
        result = ragged.roll(a, 2)
        # flat=[1,2,3,4,5,6], roll 2 -> [5,6,1,2,3,4], restore [[5,6,1],[2,3],[4]]
        assert result.tolist() == [[5.0, 6.0, 1.0], [2.0, 3.0], [4.0]]

    def test_empty_row_axis_0(self):
        a = _make([[], [1, 2], [3]], dtype=np.float64)
        result = ragged.roll(a, 1, axis=0)
        assert result.tolist() == [[3.0], [], [1.0, 2.0]]  # type: ignore[comparison-overlap]

    def test_empty_row_axis_1(self):
        a = _make([[], [1, 2], [3]], dtype=np.float64)
        result = ragged.roll(a, 1, axis=1)
        assert result.tolist() == [[], [2.0, 1.0], [3.0]]  # type: ignore[comparison-overlap]

    def test_single_row(self):
        a = _make([[1, 2, 3, 4]], dtype=np.float64)
        result = ragged.roll(a, 2, axis=1)
        assert result.tolist() == [[3.0, 4.0, 1.0, 2.0]]

    def test_shift_wraps_inner(self):
        a = _make([[1, 2, 3], [4, 5]], dtype=np.float64)
        result = ragged.roll(a, 7, axis=1)  # 7%3=1, 7%2=1
        assert result.tolist() == [[3.0, 1.0, 2.0], [5.0, 4.0]]

    def test_round_trip_axis_0(self):
        a = _make([[1, 2, 3], [4, 5], [6]], dtype=np.float64)
        result = ragged.roll(ragged.roll(a, 2, axis=0), -2, axis=0)
        assert result.tolist() == a.tolist()

    def test_round_trip_axis_1(self):
        a = _make([[1, 2, 3], [4, 5], [6]], dtype=np.float64)
        result = ragged.roll(ragged.roll(a, 2, axis=1), -2, axis=1)
        assert result.tolist() == a.tolist()

    def test_dtype_preserved(self):
        a = _make([[1.5, 2.5], [3.5]], dtype=np.float64)
        assert ragged.roll(a, 1, axis=0).dtype == np.float64

    def test_result_is_ragged_array(self):
        a = _make([[1, 2], [3]], dtype=np.float64)
        assert isinstance(ragged.roll(a, 1, axis=0), ragged.array)


# ---------------------------------------------------------------------------
# 3-D uniform arrays (fast path)
# ---------------------------------------------------------------------------


class TestRoll3DUniform:
    def test_axis_0(self):
        a_np = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        a = _make(a_np.tolist(), dtype=np.float64)
        result = ragged.roll(a, 1, axis=0)
        np.testing.assert_array_equal(_np(result), np.roll(a_np, 1, axis=0))

    def test_axis_2(self):
        a_np = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        a = _make(a_np.tolist(), dtype=np.float64)
        result = ragged.roll(a, 2, axis=2)
        np.testing.assert_array_equal(_np(result), np.roll(a_np, 2, axis=2))

    def test_axis_none(self):
        a_np = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        a = _make(a_np.tolist(), dtype=np.float64)
        result = ragged.roll(a, 5)
        np.testing.assert_array_equal(_np(result), np.roll(a_np, 5))


# ---------------------------------------------------------------------------
# 3-D ragged arrays (list fallback)
# ---------------------------------------------------------------------------


class TestRoll3DRagged:
    def test_axis_none(self):
        a = _make([[[1, 2], [3]], [[4], [5, 6, 7]]], dtype=np.float64)
        result = ragged.roll(a, 2)
        # flat=[1,2,3,4,5,6,7], roll 2 -> [6,7,1,2,3,4,5]
        # restore with inner counts [2,1,1,3] then outer counts [2,2]
        assert result.tolist() == [[[6.0, 7.0], [1.0]], [[2.0], [3.0, 4.0, 5.0]]]

    def test_axis_0(self):
        a = _make([[[1, 2], [3]], [[4], [5, 6, 7]]], dtype=np.float64)
        result = ragged.roll(a, 1, axis=0)
        assert result.tolist() == [[[4.0], [5.0, 6.0, 7.0]], [[1.0, 2.0], [3.0]]]


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestRollErrors:
    def test_shift_tuple_axis_none_raises(self):
        a = _make([1, 2, 3], dtype=np.float64)
        with pytest.raises(TypeError, match="int"):
            ragged.roll(a, (1, 2))  # tuple shift, axis=None

    def test_shift_tuple_axis_int_raises(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        with pytest.raises(ValueError, match="same length"):
            ragged.roll(a, (1, 2), axis=0)  # mismatched lengths

    def test_shift_tuple_axis_tuple_length_mismatch(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        with pytest.raises(ValueError, match="same length"):
            ragged.roll(a, (1, 2, 3), axis=(0, 1))
