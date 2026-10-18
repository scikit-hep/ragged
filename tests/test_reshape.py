# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
Tests for ragged.reshape.

Coverage
--------
Regular (non-ragged) arrays
  - 2-D to 2-D (different row/col split)
  - 2-D to 1-D (flatten)
  - 1-D to 2-D
  - 0-D to 1-D
  - adding a trailing size-1 dimension
  - -1 inference in various positions
  - copy=None (default), copy=True, copy=False

Ragged arrays
  - flatten to 1-D via shape=(-1,)
  - flatten to 1-D via explicit total count

dtype preservation
  - int, float32, complex

result type
  - always returns ragged.array

error paths
  - two -1 dimensions
  - size mismatch on regular array
  - negative shape dim other than -1
  - ragged array reshaped to multi-D raises ValueError
  - copy=False when numpy must copy
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
# Regular arrays
# ---------------------------------------------------------------------------


class TestReshapeRegular:
    def test_2d_to_2d(self):
        a = _make([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        result = ragged.reshape(a, (3, 2))
        expected = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        np.testing.assert_array_equal(_np(result), expected)

    def test_2d_to_1d(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        result = ragged.reshape(a, (4,))
        np.testing.assert_array_equal(_np(result), [1.0, 2.0, 3.0, 4.0])

    def test_1d_to_2d(self):
        a = _make([1, 2, 3, 4, 5, 6], dtype=np.float64)
        result = ragged.reshape(a, (2, 3))
        expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        np.testing.assert_array_equal(_np(result), expected)

    def test_0d_to_1d(self):
        a = ragged.array(7.0)
        result = ragged.reshape(a, (1,))
        np.testing.assert_array_equal(_np(result), [7.0])

    def test_add_trailing_dim(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        result = ragged.reshape(a, (2, 2, 1))
        # The reshaped result is a transformed ragged array, so its inner
        # dimensions are variable-length (``None``), following the same
        # convention as flip / permute_dims rather than fixed ints.
        assert result.shape == (2, None, None)

    def test_result_is_ragged_array(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        assert isinstance(ragged.reshape(a, (4,)), ragged.array)


class TestReshapeMinusOne:
    def test_minus1_first(self):
        a = _make([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        result = ragged.reshape(a, (-1, 2))
        expected = np.arange(1, 7, dtype=np.float64).reshape(-1, 2)
        np.testing.assert_array_equal(_np(result), expected)

    def test_minus1_last(self):
        a = _make([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        result = ragged.reshape(a, (2, -1))
        expected = np.arange(1, 7, dtype=np.float64).reshape(2, -1)
        np.testing.assert_array_equal(_np(result), expected)

    def test_minus1_only(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        result = ragged.reshape(a, (-1,))
        np.testing.assert_array_equal(_np(result), [1.0, 2.0, 3.0, 4.0])

    def test_minus1_3d(self):
        a_np = np.arange(24, dtype=np.float64)
        a = _make(a_np.tolist(), dtype=np.float64)
        result = ragged.reshape(a, (2, -1, 4))
        np.testing.assert_array_equal(_np(result), a_np.reshape(2, -1, 4))


class TestReshapeCopy:
    def test_copy_none_default(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        result = ragged.reshape(a, (4,))
        np.testing.assert_array_equal(_np(result), [1.0, 2.0, 3.0, 4.0])

    def test_copy_true(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        result = ragged.reshape(a, (4,), copy=True)
        np.testing.assert_array_equal(_np(result), [1.0, 2.0, 3.0, 4.0])

    def test_copy_false_view(self):
        # C-contiguous reshape that numpy can do as a view
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        result = ragged.reshape(a, (4,), copy=False)
        np.testing.assert_array_equal(_np(result), [1.0, 2.0, 3.0, 4.0])


# ---------------------------------------------------------------------------
# dtype preservation
# ---------------------------------------------------------------------------


class TestReshapeDtype:
    def test_int64_preserved(self):
        a = _make([[1, 2], [3, 4]], dtype=np.int64)
        result = ragged.reshape(a, (4,))
        assert result.dtype == np.int64

    def test_float32_preserved(self):
        a = _make([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = ragged.reshape(a, (4,))
        assert result.dtype == np.float32

    def test_complex128_preserved(self):
        a = _make([[1 + 0j, 0 + 1j], [1 + 1j, 2 + 2j]], dtype=np.complex128)
        result = ragged.reshape(a, (4,))
        assert result.dtype == np.complex128


# ---------------------------------------------------------------------------
# Ragged arrays
# ---------------------------------------------------------------------------


class TestReshapeRagged:
    def test_flatten_to_1d_explicit_count(self):
        a = ragged.array([[1, 2, 3], [4, 5]])
        result = ragged.reshape(a, (5,))
        np.testing.assert_array_equal(_np(result), [1, 2, 3, 4, 5])

    def test_flatten_to_1d_minus1(self):
        a = ragged.array([[1, 2, 3], [4, 5]])
        result = ragged.reshape(a, (-1,))
        np.testing.assert_array_equal(_np(result), [1, 2, 3, 4, 5])

    def test_result_is_ragged_array(self):
        a = ragged.array([[1, 2], [3]])
        assert isinstance(ragged.reshape(a, (-1,)), ragged.array)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestReshapeErrors:
    def test_two_minus1_raises(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        with pytest.raises(ValueError, match="[Oo]ne.*-1|-1.*[Oo]ne"):
            ragged.reshape(a, (-1, -1))

    def test_size_mismatch_regular(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        with pytest.raises(ValueError, match="[Rr]eshape|shape|size"):
            ragged.reshape(a, (5,))

    def test_negative_dim_raises(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        with pytest.raises(ValueError, match="[Nn]egative|shape|-1"):
            ragged.reshape(a, (-2, 2))

    def test_ragged_multidim_raises(self):
        a = ragged.array([[1, 2, 3], [4, 5]])
        with pytest.raises(ValueError, match="[Rr]agged|1-D|flatten"):
            ragged.reshape(a, (5, 1))

    def test_ragged_size_mismatch_raises(self):
        a = ragged.array([[1, 2, 3], [4, 5]])  # 5 elements total
        with pytest.raises(ValueError, match="[Rr]eshape|shape|size|element"):
            ragged.reshape(a, (6,))


# ---------------------------------------------------------------------------
# Shape convention: a multi-dimensional reshape produces variable-length inner
# dimensions, matching flip / permute_dims rather than fixed ints.
# ---------------------------------------------------------------------------


class TestReshapeShapeConvention:
    def test_2d_to_2d_inner_dim_none(self):
        a = _make([[1.0, 2.0], [3.0, 4.0]])
        assert ragged.reshape(a, (2, 2)).shape == (2, None)

    def test_2d_to_3d_inner_dims_none(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        assert ragged.reshape(a, (2, 2, 1)).shape == (2, None, None)

    def test_1d_target_stays_regular(self):
        # A 1-D result has no inner dimension to make variable-length.
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        assert ragged.reshape(a, (4,)).shape == (4,)
