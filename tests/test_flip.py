# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
Tests for ragged.flip.

Coverage
--------
1-D arrays
  - axis=None reverses all elements
  - axis=0 reverses all elements (same as None for 1-D)

2-D uniform arrays (fast path via numpy)
  - axis=None reverses both dimensions
  - axis=0 reverses row order
  - axis=1 reverses elements within each row
  - negative axis: axis=-1 == axis=1

2-D ragged arrays (slow path)
  - axis=None reverses rows and reverses within each row
  - axis=0 reverses row order, preserving inner order
  - axis=1 reverses within each row, preserving row order
  - empty rows handled

3-D uniform arrays
  - axis=0, axis=1, axis=2
  - tuple of axes

result type and shape
  - always returns ragged.array
  - shape is preserved

error paths
  - axis out of range (positive)
  - axis out of range (negative)
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


class TestFlip1D:
    def test_axis_none(self):
        a = _make([1, 2, 3, 4, 5], dtype=np.float64)
        result = ragged.flip(a)
        np.testing.assert_array_equal(_np(result), [5, 4, 3, 2, 1])

    def test_axis_0(self):
        a = _make([1, 2, 3], dtype=np.int64)
        result = ragged.flip(a, axis=0)
        np.testing.assert_array_equal(_np(result), [3, 2, 1])

    def test_result_is_ragged_array(self):
        a = _make([1.0, 2.0, 3.0])
        assert isinstance(ragged.flip(a), ragged.array)

    def test_dtype_preserved(self):
        a = _make([1, 2, 3], dtype=np.float32)
        assert ragged.flip(a).dtype == np.float32


# ---------------------------------------------------------------------------
# 2-D uniform arrays (fast path)
# ---------------------------------------------------------------------------


class TestFlip2DUniform:
    def test_axis_none(self):
        a = _make([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        result = ragged.flip(a)
        np.testing.assert_array_equal(_np(result), [[6, 5, 4], [3, 2, 1]])

    def test_axis_0(self):
        a = _make([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        result = ragged.flip(a, axis=0)
        np.testing.assert_array_equal(_np(result), [[4, 5, 6], [1, 2, 3]])

    def test_axis_1(self):
        a = _make([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        result = ragged.flip(a, axis=1)
        np.testing.assert_array_equal(_np(result), [[3, 2, 1], [6, 5, 4]])

    def test_negative_axis(self):
        a = _make([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        r_neg = ragged.flip(a, axis=-1)
        r_pos = ragged.flip(a, axis=1)
        np.testing.assert_array_equal(_np(r_neg), _np(r_pos))

    def test_axis_negative_0(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        r_neg = ragged.flip(a, axis=-2)
        r_pos = ragged.flip(a, axis=0)
        np.testing.assert_array_equal(_np(r_neg), _np(r_pos))

    def test_ndim_preserved(self):
        a = _make([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        assert ragged.flip(a, axis=0).ndim == a.ndim


# ---------------------------------------------------------------------------
# 2-D ragged arrays (slow path)
# ---------------------------------------------------------------------------


class TestFlip2DRagged:
    def test_axis_0(self):
        a = _make([[1, 2, 3], [4, 5], [6]], dtype=np.float64)
        result = ragged.flip(a, axis=0)
        assert result.tolist() == [[6], [4, 5], [1, 2, 3]]

    def test_axis_1(self):
        a = _make([[1, 2, 3], [4, 5], [6]], dtype=np.float64)
        result = ragged.flip(a, axis=1)
        assert result.tolist() == [[3, 2, 1], [5, 4], [6]]

    def test_axis_none(self):
        a = _make([[1, 2, 3], [4, 5], [6]], dtype=np.float64)
        result = ragged.flip(a)
        assert result.tolist() == [[6], [5, 4], [3, 2, 1]]

    def test_empty_row(self):
        a = _make([[], [1, 2], [3]], dtype=np.float64)
        result = ragged.flip(a, axis=0)
        assert result.tolist() == [[3], [1, 2], []]  # type: ignore[comparison-overlap]

    def test_empty_row_axis1(self):
        a = _make([[], [1, 2], [3]], dtype=np.float64)
        result = ragged.flip(a, axis=1)
        assert result.tolist() == [[], [2, 1], [3]]  # type: ignore[comparison-overlap]

    def test_result_is_ragged_array(self):
        a = _make([[1, 2], [3]], dtype=np.float64)
        assert isinstance(ragged.flip(a, axis=0), ragged.array)


# ---------------------------------------------------------------------------
# 3-D uniform arrays
# ---------------------------------------------------------------------------


class TestFlip3D:
    def test_axis_0(self):
        a_np = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        a = _make(a_np.tolist(), dtype=np.float64)
        result = ragged.flip(a, axis=0)
        np.testing.assert_array_equal(_np(result), np.flip(a_np, axis=0))

    def test_axis_2(self):
        a_np = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        a = _make(a_np.tolist(), dtype=np.float64)
        result = ragged.flip(a, axis=2)
        np.testing.assert_array_equal(_np(result), np.flip(a_np, axis=2))

    def test_tuple_of_axes(self):
        a_np = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        a = _make(a_np.tolist(), dtype=np.float64)
        result = ragged.flip(a, axis=(0, 2))
        np.testing.assert_array_equal(_np(result), np.flip(a_np, axis=(0, 2)))


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestFlipErrors:
    def test_axis_out_of_range_positive(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        with pytest.raises(ValueError, match="out of range|axis"):
            ragged.flip(a, axis=2)

    def test_axis_out_of_range_negative(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        with pytest.raises(ValueError, match="out of range|axis"):
            ragged.flip(a, axis=-3)


# ---------------------------------------------------------------------------
# Shape convention: the uniform (numpy fast path) and ragged paths must agree
# ---------------------------------------------------------------------------


class TestFlipShapeConvention:
    """flip preserves the input shape signature regardless of whether the data
    happens to be uniform. A ``(2, None)``-typed input stays ``(2, None)``,
    matching the convention of the ragged path (and of permute_dims / roll)."""

    def test_uniform_2d_inner_dim_stays_none(self):
        uniform = _make([[1.0, 2.0], [3.0, 4.0]])  # uniform data, ragged layout
        assert uniform.shape == (2, None)
        assert ragged.flip(uniform, axis=1).shape == (2, None)
        assert ragged.flip(uniform, axis=0).shape == (2, None)
        assert ragged.flip(uniform).shape == (2, None)

    def test_uniform_matches_ragged_signature(self):
        uniform = _make([[1.0, 2.0], [3.0, 4.0]])  # fast (numpy) path
        ragged_in = _make([[1.0, 2.0], [3.0]])  # general (awkward) path
        assert (
            ragged.flip(uniform, axis=1).shape == ragged.flip(ragged_in, axis=1).shape
        )

    def test_uniform_3d_inner_dims_stay_none(self):
        a_np = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        a = _make(a_np.tolist(), dtype=np.float64)
        assert ragged.flip(a, axis=0).shape == (2, None, None)
        assert ragged.flip(a, axis=2).shape == (2, None, None)
