# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
Tests for ragged.permute_dims.

Coverage
--------
identity permutation
  - (0,) on 1-D
  - (0, 1) on 2-D: no-op

2-D uniform arrays (fast path via numpy)
  - (1, 0): standard matrix transpose
  - dtype preserved

2-D ragged arrays (medium path — awkward native)
  - (1, 0): ragged transpose, shorter rows produce shorter columns
  - empty rows handled
  - single-row array
  - dtype preserved

3-D uniform arrays (fast path via numpy)
  - (1, 0, 2): swap first two axes
  - (2, 1, 0): full reversal of axes
  - (0, 2, 1): swap last two axes

3-D ragged arrays (list fallback)
  - (1, 0, 2): swap outer two axes

result type
  - always returns ragged.array

error paths
  - axes wrong length raises ValueError
  - duplicate axis raises ValueError
  - axis out of range raises ValueError
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
# Identity permutation
# ---------------------------------------------------------------------------


class TestPermuteDimsIdentity:
    def test_1d_identity(self):
        a = _make([1.0, 2.0, 3.0])
        result = ragged.permute_dims(a, (0,))
        np.testing.assert_array_equal(_np(result), [1.0, 2.0, 3.0])

    def test_2d_identity(self):
        a = _make([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        result = ragged.permute_dims(a, (0, 1))
        np.testing.assert_array_equal(_np(result), [[1, 2, 3], [4, 5, 6]])

    def test_3d_identity(self):
        a_np = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        a = _make(a_np.tolist(), dtype=np.float64)
        result = ragged.permute_dims(a, (0, 1, 2))
        np.testing.assert_array_equal(_np(result), a_np)


# ---------------------------------------------------------------------------
# 2-D uniform arrays (fast path)
# ---------------------------------------------------------------------------


class TestPermuteDims2DUniform:
    def test_transpose(self):
        a_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        a = _make(a_np.tolist(), dtype=np.float64)
        result = ragged.permute_dims(a, (1, 0))
        np.testing.assert_array_equal(_np(result), a_np.T)

    def test_square_transpose(self):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
        a = _make(a_np.tolist(), dtype=np.float64)
        result = ragged.permute_dims(a, (1, 0))
        np.testing.assert_array_equal(_np(result), a_np.T)

    def test_dtype_preserved_float32(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float32)
        assert ragged.permute_dims(a, (1, 0)).dtype == np.float32

    def test_result_is_ragged_array(self):
        a = _make([[1.0, 2.0], [3.0, 4.0]])
        assert isinstance(ragged.permute_dims(a, (1, 0)), ragged.array)


# ---------------------------------------------------------------------------
# 2-D ragged arrays (awkward-native medium path)
# ---------------------------------------------------------------------------


class TestPermuteDims2DRagged:
    def test_basic_transpose(self):
        # [[1,2,3],[4,5]] -> [[1,4],[2,5],[3]]
        a = _make([[1, 2, 3], [4, 5]], dtype=np.float64)
        result = ragged.permute_dims(a, (1, 0))
        assert result.tolist() == [[1.0, 4.0], [2.0, 5.0], [3.0]]

    def test_three_rows(self):
        a = _make([[1, 2, 3], [4, 5], [6]], dtype=np.float64)
        result = ragged.permute_dims(a, (1, 0))
        assert result.tolist() == [[1.0, 4.0, 6.0], [2.0, 5.0], [3.0]]

    def test_empty_row(self):
        a = _make([[], [1, 2], [3]], dtype=np.float64)
        result = ragged.permute_dims(a, (1, 0))
        assert result.tolist() == [[1.0, 3.0], [2.0]]

    def test_all_empty_rows(self):
        a = _make([[], [], []], dtype=np.float64)
        result = ragged.permute_dims(a, (1, 0))
        assert result.tolist() == []

    def test_single_row(self):
        a = _make([[1, 2, 3]], dtype=np.float64)
        result = ragged.permute_dims(a, (1, 0))
        assert result.tolist() == [[1.0], [2.0], [3.0]]

    def test_single_column(self):
        a = _make([[1], [2], [3]], dtype=np.float64)
        result = ragged.permute_dims(a, (1, 0))
        assert result.tolist() == [[1.0, 2.0, 3.0]]

    def test_dtype_preserved(self):
        a = _make([[1.5, 2.5], [3.5]], dtype=np.float64)
        assert ragged.permute_dims(a, (1, 0)).dtype == np.float64

    def test_round_trip(self):
        a = _make([[1, 2, 3], [4, 5], [6]], dtype=np.float64)
        result = ragged.permute_dims(ragged.permute_dims(a, (1, 0)), (1, 0))
        assert result.tolist() == a.tolist()


# ---------------------------------------------------------------------------
# 3-D uniform arrays (fast path)
# ---------------------------------------------------------------------------


class TestPermuteDims3DUniform:
    def test_swap_first_two(self):
        a_np = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        a = _make(a_np.tolist(), dtype=np.float64)
        result = ragged.permute_dims(a, (1, 0, 2))
        np.testing.assert_array_equal(_np(result), np.transpose(a_np, (1, 0, 2)))

    def test_full_reverse(self):
        a_np = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        a = _make(a_np.tolist(), dtype=np.float64)
        result = ragged.permute_dims(a, (2, 1, 0))
        np.testing.assert_array_equal(_np(result), np.transpose(a_np, (2, 1, 0)))

    def test_swap_last_two(self):
        a_np = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        a = _make(a_np.tolist(), dtype=np.float64)
        result = ragged.permute_dims(a, (0, 2, 1))
        np.testing.assert_array_equal(_np(result), np.transpose(a_np, (0, 2, 1)))


# ---------------------------------------------------------------------------
# 3-D ragged (list fallback)
# ---------------------------------------------------------------------------


class TestPermuteDims3DRagged:
    def test_swap_outer_two(self):
        a = _make([[[1.1, 2.2], [3.3]], [[4.4], [5.5, 6.6, 7.7]]], dtype=np.float64)
        result = ragged.permute_dims(a, (1, 0, 2))
        assert result.tolist() == [[[1.1, 2.2], [4.4]], [[3.3], [5.5, 6.6, 7.7]]]


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestPermuteDimsErrors:
    def test_wrong_length_raises(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        with pytest.raises(ValueError, match="permutation"):
            ragged.permute_dims(a, (1,))

    def test_duplicate_axis_raises(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        with pytest.raises(ValueError, match="permutation"):
            ragged.permute_dims(a, (0, 0))

    def test_out_of_range_raises(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        with pytest.raises(ValueError, match="permutation"):
            ragged.permute_dims(a, (0, 2))
