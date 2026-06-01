# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
Tests for ragged.tensordot (Array API: linalg.tensordot / top-level tensordot).

Coverage
--------
axes as int
  - axes=0: outer product of two 1-D arrays
  - axes=1: dot product (1-D x 1-D, 2-D x 2-D)
  - axes=2: double contraction (default, scalar result for 2-D square)
  - axes=N on higher-dimensional arrays

axes as explicit sequences
  - ([i], [j]) notation matching numpy semantics
  - negative axis indices

dtype promotion
  - int x float -> float

result type
  - always returns ragged.array

error paths
  - axes is negative int
  - axes exceeds ndim
  - contracted axis sequences of unequal length
  - contracted dimension size mismatch
  - contracted axis is ragged (None)
  - duplicate axis in sequence
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
# axes as int
# ---------------------------------------------------------------------------


class TestTensordotAxesInt:
    def test_axes0_outer_product_1d(self):
        a = _make([1.0, 2.0])
        b = _make([3.0, 4.0, 5.0])
        result = ragged.tensordot(a, b, axes=0)
        expected = np.tensordot([1, 2], [3, 4, 5], axes=0)
        np.testing.assert_array_equal(_np(result), expected)

    def test_axes0_outer_product_2d(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        b = _make([[5, 6], [7, 8]], dtype=np.float64)
        result = ragged.tensordot(a, b, axes=0)
        expected = np.tensordot([[1, 2], [3, 4]], [[5, 6], [7, 8]], axes=0)
        np.testing.assert_array_equal(_np(result), expected)

    def test_axes1_dot_1d(self):
        a = _make([1.0, 2.0, 3.0])
        b = _make([4.0, 5.0, 6.0])
        result = ragged.tensordot(a, b, axes=1)
        expected = np.tensordot([1, 2, 3], [4, 5, 6], axes=1)
        np.testing.assert_allclose(np.float64(result.tolist()), np.float64(expected))

    def test_axes1_matmul_2d(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        b = _make([[5, 6], [7, 8]], dtype=np.float64)
        result = ragged.tensordot(a, b, axes=1)
        expected = np.tensordot([[1, 2], [3, 4]], [[5, 6], [7, 8]], axes=1)
        np.testing.assert_array_equal(_np(result), expected)

    def test_axes2_double_contraction(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        b = _make([[5, 6], [7, 8]], dtype=np.float64)
        result = ragged.tensordot(a, b, axes=2)
        expected = np.tensordot([[1, 2], [3, 4]], [[5, 6], [7, 8]], axes=2)
        assert np.float64(result.tolist()) == pytest.approx(np.float64(expected))

    def test_axes2_default(self):
        # axes=2 is the default
        a = _make([[1.0, 0.0], [0.0, 1.0]])
        b = _make([[2.0, 3.0], [4.0, 5.0]])
        r_explicit = ragged.tensordot(a, b, axes=2)
        r_default = ragged.tensordot(a, b)
        np.testing.assert_array_equal(_np(r_explicit), _np(r_default))

    def test_axes_higher_dim(self):
        # (2,3,4) x (4,2) contracting last 1 of x1 (size 4) vs first 1 of x2 (size 4)
        a_np = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        b_np = np.arange(8, dtype=np.float64).reshape(4, 2)
        a = _make(a_np.tolist(), dtype=np.float64)
        b = _make(b_np.tolist(), dtype=np.float64)
        result = ragged.tensordot(a, b, axes=1)
        expected = np.tensordot(a_np, b_np, axes=1)
        np.testing.assert_allclose(_np(result), expected)


# ---------------------------------------------------------------------------
# axes as explicit sequences
# ---------------------------------------------------------------------------


class TestTensordotAxesSequence:
    def test_explicit_match_int_axes1(self):
        a = _make([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        b = _make([[7, 8], [9, 10], [11, 12]], dtype=np.float64)
        r_seq = ragged.tensordot(a, b, axes=([1], [0]))
        r_int = ragged.tensordot(a, b, axes=1)
        np.testing.assert_array_equal(_np(r_seq), _np(r_int))

    def test_explicit_non_trivial_axes(self):
        a_np = np.arange(60, dtype=np.float64).reshape(3, 4, 5)
        b_np = np.arange(24, dtype=np.float64).reshape(4, 3, 2)
        a = _make(a_np.tolist(), dtype=np.float64)
        b = _make(b_np.tolist(), dtype=np.float64)
        result = ragged.tensordot(a, b, axes=([1, 0], [0, 1]))
        expected = np.tensordot(a_np, b_np, axes=([1, 0], [0, 1]))
        np.testing.assert_allclose(_np(result), expected)

    def test_negative_axis_indices(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        b = _make([[5, 6], [7, 8]], dtype=np.float64)
        r_neg = ragged.tensordot(a, b, axes=([-1], [0]))
        r_pos = ragged.tensordot(a, b, axes=([1], [0]))
        np.testing.assert_array_equal(_np(r_neg), _np(r_pos))


# ---------------------------------------------------------------------------
# dtype promotion
# ---------------------------------------------------------------------------


class TestTensordotDtype:
    def test_int_times_float_gives_float(self):
        a = _make([[1, 2], [3, 4]], dtype=np.int32)
        b = _make([[1.5, 0.5], [0.5, 1.5]], dtype=np.float64)
        result = ragged.tensordot(a, b, axes=1)
        assert np.issubdtype(result.dtype, np.floating)

    def test_float32_preserved(self):
        a = _make([[1, 0], [0, 1]], dtype=np.float32)
        b = _make([[2, 3], [4, 5]], dtype=np.float32)
        result = ragged.tensordot(a, b, axes=1)
        assert result.dtype == np.float32

    def test_result_is_ragged_array(self):
        a = _make([[1.0, 2.0], [3.0, 4.0]])
        b = _make([[5.0, 6.0], [7.0, 8.0]])
        assert isinstance(ragged.tensordot(a, b, axes=1), ragged.array)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestTensordotErrors:
    def test_negative_int_axes_raises(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        b = _make([[5, 6], [7, 8]], dtype=np.float64)
        with pytest.raises(ValueError, match="[Nn]on-negative|axes"):
            ragged.tensordot(a, b, axes=-1)

    def test_axes_exceeds_ndim_raises(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        b = _make([[5, 6], [7, 8]], dtype=np.float64)
        with pytest.raises(ValueError, match="[Ee]xceeds|ndim|axes"):
            ragged.tensordot(a, b, axes=3)

    def test_unequal_sequence_lengths_raises(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        b = _make([[5, 6], [7, 8]], dtype=np.float64)
        with pytest.raises(ValueError, match="[Ee]qual [Ll]ength|length"):
            ragged.tensordot(a, b, axes=([0, 1], [0]))

    def test_dimension_size_mismatch_raises(self):
        # (2,3) contracted axis 1 (size 3) vs (2,2) axis 0 (size 2)
        a = _make([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        b = _make([[1, 0], [0, 1]], dtype=np.float64)
        with pytest.raises(ValueError, match="[Mm]ismatch|dimension"):
            ragged.tensordot(a, b, axes=([1], [0]))

    def test_ragged_contracted_axis_raises(self):
        # last axis of a is ragged (rows have different lengths)
        a = ragged.array([[1, 2, 3], [4, 5]])
        b = _make([[1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=np.float64)
        with pytest.raises(ValueError, match="[Rr]agged|contracted"):
            ragged.tensordot(a, b, axes=([1], [0]))

    def test_duplicate_axes_raises(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        b = _make([[5, 6], [7, 8]], dtype=np.float64)
        with pytest.raises(ValueError, match="[Uu]nique"):
            ragged.tensordot(a, b, axes=([0, 0], [0, 1]))
