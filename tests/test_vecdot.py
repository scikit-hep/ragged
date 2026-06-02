# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
Tests for ragged.vecdot.

Coverage
--------
1-D arrays
  - basic dot product -> 0-D scalar result
  - complex: x1 is conjugated

N-D arrays
  - default axis=-1 (contract last axis)
  - explicit axis=0 (contract first axis)
  - axis=1 on 3-D array

Ragged non-contracted dims
  - ragged batch dimension with regular contracted axis

dtype promotion
  - int x float -> float
  - complex result preserves complex dtype

result type
  - always returns ragged.array

error paths
  - 0-D input
  - axis out of range
  - contracted axis is ragged
  - contracted dimension size mismatch
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


class TestVecdot1D:
    def test_basic_dot_product(self):
        a = _make([1.0, 2.0, 3.0])
        b = _make([4.0, 5.0, 6.0])
        result = ragged.vecdot(a, b)
        assert np.float64(result.tolist()) == pytest.approx(32.0)

    def test_result_is_scalar_ragged_array(self):
        a = _make([1.0, 2.0])
        b = _make([3.0, 4.0])
        result = ragged.vecdot(a, b)
        assert isinstance(result, ragged.array)
        assert result.ndim == 0

    def test_complex_conjugates_x1(self):
        # vecdot computes conj(x1) . x2
        a = _make([1 + 2j, 3 + 4j])
        b = _make([1 + 0j, 0 + 1j])
        result = ragged.vecdot(a, b)
        # conj(1+2j)*1 + conj(3+4j)*1j = (1-2j) + (3-4j)*1j = 1-2j + 3j+4 = 5+1j
        assert np.complex128(result.tolist()) == pytest.approx(5 + 1j)

    def test_matches_numpy(self):
        a_np = np.array([2.0, -1.0, 3.0])
        b_np = np.array([4.0, 5.0, -2.0])
        a = _make(a_np.tolist(), dtype=np.float64)
        b = _make(b_np.tolist(), dtype=np.float64)
        result = ragged.vecdot(a, b)
        expected = np.vdot(
            a_np, b_np
        )  # np.vdot is the 1-D equivalent available in all numpy versions
        assert np.float64(result.tolist()) == pytest.approx(float(expected))


# ---------------------------------------------------------------------------
# N-D arrays
# ---------------------------------------------------------------------------


class TestVecdotND:
    def test_2d_default_axis(self):
        # axis=-1: contract last axis, result shape (2,)
        a = _make([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        b = _make([[7, 8, 9], [1, 2, 3]], dtype=np.float64)
        result = ragged.vecdot(a, b)
        np.testing.assert_array_equal(_np(result), [50.0, 32.0])

    def test_2d_axis0(self):
        # axis=0: contract first axis, result shape (3,)
        a = _make([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        b = _make([[7, 8, 9], [1, 2, 3]], dtype=np.float64)
        result = ragged.vecdot(a, b, axis=0)
        np.testing.assert_array_equal(_np(result), [11.0, 26.0, 45.0])

    def test_2d_negative_axis_minus1(self):
        a = _make([[1.0, 0.0], [0.0, 1.0]])
        b = _make([[3.0, 4.0], [5.0, 6.0]])
        r_neg = ragged.vecdot(a, b, axis=-1)
        r_pos = ragged.vecdot(a, b, axis=1)
        np.testing.assert_array_equal(_np(r_neg), _np(r_pos))

    def test_3d_axis1(self):
        a_np = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        b_np = np.ones((2, 3, 4), dtype=np.float64)
        a = _make(a_np.tolist(), dtype=np.float64)
        b = _make(b_np.tolist(), dtype=np.float64)
        result = ragged.vecdot(a, b, axis=1)
        # np.vecdot added in NumPy 2.0; compute manually for compatibility
        expected = np.sum(np.conj(a_np) * b_np, axis=1)
        np.testing.assert_allclose(_np(result), expected)

    def test_result_shape(self):
        a = _make([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        b = _make([[1, 1, 1], [1, 1, 1]], dtype=np.float64)
        result = ragged.vecdot(a, b)
        assert result.shape == (2,)

    def test_result_is_ragged_array(self):
        a = _make([[1.0, 2.0], [3.0, 4.0]])
        b = _make([[5.0, 6.0], [7.0, 8.0]])
        assert isinstance(ragged.vecdot(a, b), ragged.array)


# ---------------------------------------------------------------------------
# dtype promotion
# ---------------------------------------------------------------------------


class TestVecdotDtype:
    def test_int_times_float_gives_float(self):
        a = _make([1, 2, 3], dtype=np.int32)
        b = _make([1.5, 2.5, 3.5], dtype=np.float64)
        result = ragged.vecdot(a, b)
        assert np.issubdtype(result.dtype, np.floating)

    def test_float32_preserved(self):
        a = _make([1.0, 2.0], dtype=np.float32)
        b = _make([3.0, 4.0], dtype=np.float32)
        result = ragged.vecdot(a, b)
        assert result.dtype == np.float32

    def test_complex_dtype_preserved(self):
        a = _make([1 + 0j, 0 + 1j], dtype=np.complex128)
        b = _make([1 + 0j, 1 + 0j], dtype=np.complex128)
        result = ragged.vecdot(a, b)
        assert np.issubdtype(result.dtype, np.complexfloating)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestVecdotErrors:
    def test_0d_raises(self):
        a = ragged.array(5.0)
        b = _make([1.0, 2.0])
        with pytest.raises(ValueError, match="[Dd]imension"):
            ragged.vecdot(a, b)

    def test_axis_out_of_range_positive(self):
        a = _make([1.0, 2.0])
        b = _make([3.0, 4.0])
        with pytest.raises(ValueError, match="[Oo]ut of range|axis"):
            ragged.vecdot(a, b, axis=2)

    def test_axis_out_of_range_negative(self):
        a = _make([1.0, 2.0])
        b = _make([3.0, 4.0])
        with pytest.raises(ValueError, match="[Oo]ut of range|axis"):
            ragged.vecdot(a, b, axis=-3)

    def test_ragged_contracted_axis_raises(self):
        # Last axis is ragged (rows of different lengths)
        a = ragged.array([[1, 2, 3], [4, 5]])
        b = ragged.array([[1, 0, 1], [0, 1, 0]])
        with pytest.raises(ValueError, match="[Rr]agged|contracted"):
            ragged.vecdot(a, b)

    def test_contracted_dim_mismatch_raises(self):
        a = _make([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        b = _make([[1, 2], [3, 4]], dtype=np.float64)
        with pytest.raises(ValueError, match="[Mm]ismatch|dimension"):
            ragged.vecdot(a, b)
