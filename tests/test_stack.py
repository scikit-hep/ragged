# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
Tests for ragged.stack.

Coverage
--------
1-D arrays
  - axis=0: new leading dim -> (N, n)
  - axis=1: new trailing dim -> (n, N)
  - axis=-1: same as axis=1 for 1-D
  - three arrays stacked

2-D uniform arrays (fast path via numpy)
  - axis=0: (N, m, n)
  - axis=1: (m, N, n)
  - axis=2: (m, n, N)
  - axis=-1: same as axis=2
  - negative axis=-2: same as axis=1

2-D ragged arrays
  - axis=0: (N, m, None)
  - axis=1: pairs each row -> (m, N, None)

3-D uniform arrays (fast path)
  - axis=0
  - axis=2

result type and dtype
  - always returns ragged.array
  - dtype preserved

error paths
  - empty input raises ValueError
  - ndim mismatch raises ValueError
  - dtype mismatch raises ValueError
  - axis out of range raises ValueError
  - non-array input raises TypeError
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


class TestStack1D:
    def test_axis_0_two(self):
        a = _make([1.0, 2.0, 3.0])
        b = _make([4.0, 5.0, 6.0])
        result = ragged.stack([a, b], axis=0)
        np.testing.assert_array_equal(_np(result), [[1, 2, 3], [4, 5, 6]])

    def test_axis_1(self):
        a = _make([1.0, 2.0, 3.0])
        b = _make([4.0, 5.0, 6.0])
        result = ragged.stack([a, b], axis=1)
        np.testing.assert_array_equal(_np(result), [[1, 4], [2, 5], [3, 6]])

    def test_axis_neg1(self):
        a = _make([1.0, 2.0, 3.0])
        b = _make([4.0, 5.0, 6.0])
        r_neg = ragged.stack([a, b], axis=-1)
        r_pos = ragged.stack([a, b], axis=1)
        np.testing.assert_array_equal(_np(r_neg), _np(r_pos))

    def test_three_arrays(self):
        a = _make([1.0, 2.0])
        b = _make([3.0, 4.0])
        c = _make([5.0, 6.0])
        result = ragged.stack([a, b, c], axis=0)
        np.testing.assert_array_equal(_np(result), [[1, 2], [3, 4], [5, 6]])

    def test_result_is_ragged_array(self):
        a = _make([1.0, 2.0])
        b = _make([3.0, 4.0])
        assert isinstance(ragged.stack([a, b]), ragged.array)

    def test_dtype_preserved_float32(self):
        a = _make([1.0, 2.0], dtype=np.float32)
        b = _make([3.0, 4.0], dtype=np.float32)
        assert ragged.stack([a, b]).dtype == np.float32

    def test_dtype_preserved_int64(self):
        a = _make([1, 2, 3], dtype=np.int64)
        b = _make([4, 5, 6], dtype=np.int64)
        assert ragged.stack([a, b]).dtype == np.int64


# ---------------------------------------------------------------------------
# 2-D uniform arrays (fast path)
# ---------------------------------------------------------------------------


class TestStack2DUniform:
    def test_axis_0(self):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
        b_np = np.array([[5, 6], [7, 8]], dtype=np.float64)
        a, b = (
            _make(a_np.tolist(), dtype=np.float64),
            _make(b_np.tolist(), dtype=np.float64),
        )
        result = ragged.stack([a, b], axis=0)
        np.testing.assert_array_equal(_np(result), np.stack([a_np, b_np], axis=0))

    def test_axis_1(self):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
        b_np = np.array([[5, 6], [7, 8]], dtype=np.float64)
        a, b = (
            _make(a_np.tolist(), dtype=np.float64),
            _make(b_np.tolist(), dtype=np.float64),
        )
        result = ragged.stack([a, b], axis=1)
        np.testing.assert_array_equal(_np(result), np.stack([a_np, b_np], axis=1))

    def test_axis_2(self):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
        b_np = np.array([[5, 6], [7, 8]], dtype=np.float64)
        a, b = (
            _make(a_np.tolist(), dtype=np.float64),
            _make(b_np.tolist(), dtype=np.float64),
        )
        result = ragged.stack([a, b], axis=2)
        np.testing.assert_array_equal(_np(result), np.stack([a_np, b_np], axis=2))

    def test_axis_neg1(self):
        a_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        b_np = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.float64)
        a, b = (
            _make(a_np.tolist(), dtype=np.float64),
            _make(b_np.tolist(), dtype=np.float64),
        )
        r_neg = ragged.stack([a, b], axis=-1)
        r_pos = ragged.stack([a, b], axis=2)
        np.testing.assert_array_equal(_np(r_neg), _np(r_pos))

    def test_axis_neg2(self):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
        b_np = np.array([[5, 6], [7, 8]], dtype=np.float64)
        a, b = (
            _make(a_np.tolist(), dtype=np.float64),
            _make(b_np.tolist(), dtype=np.float64),
        )
        r_neg = ragged.stack([a, b], axis=-2)
        r_pos = ragged.stack([a, b], axis=1)
        np.testing.assert_array_equal(_np(r_neg), _np(r_pos))

    def test_dtype_preserved(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float32)
        b = _make([[5, 6], [7, 8]], dtype=np.float32)
        assert ragged.stack([a, b], axis=0).dtype == np.float32


# ---------------------------------------------------------------------------
# 2-D ragged arrays
# ---------------------------------------------------------------------------


class TestStack2DRagged:
    def test_axis_0(self):
        a = _make([[1, 2, 3], [4, 5]], dtype=np.float64)
        b = _make([[6, 7, 8], [9, 10]], dtype=np.float64)
        result = ragged.stack([a, b], axis=0)
        assert result.shape == (2, 2, None)
        assert result.tolist() == [
            [[1.0, 2.0, 3.0], [4.0, 5.0]],
            [[6.0, 7.0, 8.0], [9.0, 10.0]],
        ]

    def test_axis_1(self):
        a = _make([[1, 2, 3], [4, 5]], dtype=np.float64)
        b = _make([[6, 7, 8], [9, 10]], dtype=np.float64)
        result = ragged.stack([a, b], axis=1)
        assert result.shape == (2, 2, None)
        assert result.tolist() == [
            [[1.0, 2.0, 3.0], [6.0, 7.0, 8.0]],
            [[4.0, 5.0], [9.0, 10.0]],
        ]

    def test_three_ragged(self):
        a = _make([[1, 2], [3]], dtype=np.float64)
        b = _make([[4, 5], [6]], dtype=np.float64)
        c = _make([[7, 8], [9]], dtype=np.float64)
        result = ragged.stack([a, b, c], axis=0)
        assert result.shape == (3, 2, None)
        assert result.tolist() == [
            [[1.0, 2.0], [3.0]],
            [[4.0, 5.0], [6.0]],
            [[7.0, 8.0], [9.0]],
        ]

    def test_dtype_preserved(self):
        a = _make([[1.5], [2.5]], dtype=np.float64)
        b = _make([[3.5], [4.5]], dtype=np.float64)
        assert ragged.stack([a, b], axis=0).dtype == np.float64

    def test_result_is_ragged_array(self):
        a = _make([[1, 2], [3]], dtype=np.float64)
        b = _make([[4, 5], [6]], dtype=np.float64)
        assert isinstance(ragged.stack([a, b], axis=0), ragged.array)


# ---------------------------------------------------------------------------
# 3-D uniform arrays (fast path)
# ---------------------------------------------------------------------------


class TestStack3DUniform:
    def test_axis_0(self):
        a_np = np.arange(6, dtype=np.float64).reshape(2, 3)
        b_np = np.arange(6, 12, dtype=np.float64).reshape(2, 3)
        a, b = (
            _make(a_np.tolist(), dtype=np.float64),
            _make(b_np.tolist(), dtype=np.float64),
        )
        result = ragged.stack([a, b, a], axis=0)
        np.testing.assert_array_equal(_np(result), np.stack([a_np, b_np, a_np], axis=0))

    def test_axis_2(self):
        a_np = np.arange(6, dtype=np.float64).reshape(2, 3)
        b_np = np.arange(6, 12, dtype=np.float64).reshape(2, 3)
        a, b = (
            _make(a_np.tolist(), dtype=np.float64),
            _make(b_np.tolist(), dtype=np.float64),
        )
        result = ragged.stack([a, b], axis=2)
        np.testing.assert_array_equal(_np(result), np.stack([a_np, b_np], axis=2))


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestStackErrors:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            ragged.stack([])

    def test_ndim_mismatch_raises(self):
        a = _make([1.0, 2.0])
        b = _make([[1.0, 2.0]])
        with pytest.raises(ValueError, match="dimensions"):
            ragged.stack([a, b])

    def test_dtype_mismatch_raises(self):
        a = _make([1.0, 2.0], dtype=np.float32)
        b = _make([3.0, 4.0], dtype=np.float64)
        with pytest.raises(ValueError, match="dtype"):
            ragged.stack([a, b])

    def test_axis_out_of_range_raises(self):
        a = _make([[1.0, 2.0]])
        b = _make([[3.0, 4.0]])
        with pytest.raises(ValueError, match="out of bounds"):
            ragged.stack([a, b], axis=3)

    def test_axis_negative_out_of_range_raises(self):
        a = _make([1.0, 2.0])
        b = _make([3.0, 4.0])
        with pytest.raises(ValueError, match="out of bounds"):
            ragged.stack([a, b], axis=-3)

    def test_non_array_input_raises(self):
        from typing import Any

        bad: Any = [[1, 2], [3, 4]]
        with pytest.raises(TypeError):
            ragged.stack(bad)
