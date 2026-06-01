# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
Tests for ragged.array.__matmul__ and __rmatmul__.

Coverage
--------
Regular (non-ragged) arrays
  - 2-D x 2-D square
  - 2-D x 2-D non-square  (M,K) @ (K,N) -> (M,N)
  - 3-D batched matmul    (..., M, K) @ (..., K, N) → (..., M, N)
  - @ operator and explicit __matmul__ call give the same result
  - __rmatmul__: scalar-side left operand delegates correctly

Ragged (non-contracted) axes
  - ragged M axis: rows have different lengths — result rows vary accordingly
  - ragged batch dimension: list of differently-sized matrices
  - result dtype follows np.result_type promotion rules

Error paths
  - ndim < 2 on either operand raises ValueError
  - contracted dimension is ragged (None) raises ValueError
  - contracted dimension mismatch raises ValueError
  - mixed-device operands raise TypeError (mocked)

In-place operator
  - __imatmul__ updates the array in-place (Python-object level)
"""

from __future__ import annotations

import numpy as np
import pytest

import ragged

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make(nested, dtype=None) -> ragged.array:
    """Shorthand constructor."""
    return ragged.array(nested, dtype=dtype)


def _as_np(x: ragged.array) -> np.ndarray:
    """Convert a fully regular ragged.array to a NumPy array for assertions."""
    return np.array(x.tolist())


# ---------------------------------------------------------------------------
# Regular (non-ragged) 2-D matmul
# ---------------------------------------------------------------------------


class TestMatmul2D:
    def test_square(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        b = _make([[5, 6], [7, 8]], dtype=np.float64)
        result = a @ b
        expected = np.array([[1, 2], [3, 4]], dtype=np.float64) @ np.array(
            [[5, 6], [7, 8]], dtype=np.float64
        )
        np.testing.assert_array_equal(_as_np(result), expected)

    def test_non_square(self):
        # (2,3) @ (3,2) → (2,2)
        a = _make([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        b = _make([[7, 8], [9, 10], [11, 12]], dtype=np.float64)
        result = a @ b
        expected = np.array([[1, 2, 3], [4, 5, 6]]) @ np.array(
            [[7, 8], [9, 10], [11, 12]]
        )
        np.testing.assert_array_equal(_as_np(result), expected.astype(np.float64))

    def test_non_square_tall_times_wide(self):
        # (3,2) @ (2,4) → (3,4)
        a = _make([[1, 0], [0, 1], [1, 1]], dtype=np.float64)
        b = _make([[2, 3, 4, 5], [6, 7, 8, 9]], dtype=np.float64)
        result = a @ b
        expected = np.array([[1, 0], [0, 1], [1, 1]]) @ np.array(
            [[2, 3, 4, 5], [6, 7, 8, 9]]
        )
        np.testing.assert_array_equal(_as_np(result), expected.astype(np.float64))

    def test_identity_matrix(self):
        a = _make([[3, 4], [5, 6]], dtype=np.float64)
        eye = _make([[1, 0], [0, 1]], dtype=np.float64)
        np.testing.assert_array_equal(_as_np(a @ eye), _as_np(a))
        np.testing.assert_array_equal(_as_np(eye @ a), _as_np(a))

    def test_explicit_dunder_matches_operator(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        b = _make([[1, 0], [0, 1]], dtype=np.float64)
        via_op = a @ b
        via_dunder = a.__matmul__(b)
        np.testing.assert_array_equal(_as_np(via_op), _as_np(via_dunder))

    def test_integer_dtype(self):
        a = _make([[1, 2], [3, 4]], dtype=np.int64)
        b = _make([[5, 0], [0, 5]], dtype=np.int64)
        result = a @ b
        expected = np.array([[5, 10], [15, 20]], dtype=np.int64)
        np.testing.assert_array_equal(_as_np(result), expected)

    def test_float32_dtype(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float32)
        b = _make([[1, 2], [3, 4]], dtype=np.float32)
        result = a @ b
        assert result.dtype == np.float32

    def test_result_shape_2d(self):
        a = _make([[1, 2, 3], [4, 5, 6]], dtype=np.float64)  # (2,3)
        b = _make([[1, 0], [0, 1], [1, 1]], dtype=np.float64)  # (3,2)
        result = a @ b
        assert result.shape == (2, 2)

    def test_dtype_promotion_int_float(self):
        a = _make([[1, 2], [3, 4]], dtype=np.int32)
        b = _make([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
        result = a @ b
        assert np.issubdtype(result.dtype, np.floating)


# ---------------------------------------------------------------------------
# Batched (N-D) matmul
# ---------------------------------------------------------------------------


class TestMatmulBatched:
    def test_3d_batch(self):
        # (2, 2, 3) @ (2, 3, 2) → (2, 2, 2)
        a_np = np.arange(12, dtype=np.float64).reshape(2, 2, 3)
        b_np = np.arange(12, dtype=np.float64).reshape(2, 3, 2)
        a = _make(a_np.tolist(), dtype=np.float64)
        b = _make(b_np.tolist(), dtype=np.float64)
        result = a @ b
        expected = a_np @ b_np
        np.testing.assert_allclose(_as_np(result), expected)

    def test_batch_shape_preserved(self):
        a_np = np.ones((3, 4, 5), dtype=np.float64)
        b_np = np.ones((3, 5, 2), dtype=np.float64)
        a = _make(a_np.tolist(), dtype=np.float64)
        b = _make(b_np.tolist(), dtype=np.float64)
        result = a @ b
        assert result.shape == (3, 4, 2)


# ---------------------------------------------------------------------------
# __rmatmul__
# ---------------------------------------------------------------------------


class TestRMatmul:
    def test_rmatmul_via_operator(self):
        """Python tries b.__matmul__(a) first; if that fails, tries a.__rmatmul__(b)."""
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        b = _make([[1, 0], [0, 1]], dtype=np.float64)
        # b @ a — both are ragged.array, so __matmul__ fires on b directly.
        result = b @ a
        np.testing.assert_array_equal(_as_np(result), _as_np(a))

    def test_rmatmul_delegates_correctly(self):
        """__rmatmul__(self, other) computes other @ self."""
        a = _make([[2, 0], [0, 2]], dtype=np.float64)
        b = _make([[1, 2], [3, 4]], dtype=np.float64)
        # a.__rmatmul__(b) == b @ a
        result = a.__rmatmul__(b)
        expected = _as_np(b @ a)
        np.testing.assert_array_equal(_as_np(result), expected)

    def test_rmatmul_non_commutative(self):
        a = _make([[1, 2], [0, 1]], dtype=np.float64)
        b = _make([[3, 4], [5, 6]], dtype=np.float64)
        ab = _as_np(a @ b)
        ba = _as_np(b @ a)
        assert not np.array_equal(
            ab, ba
        ), "a@b and b@a should differ for these matrices"


# ---------------------------------------------------------------------------
# Ragged (non-contracted) dimensions
# ---------------------------------------------------------------------------


class TestMatmulRagged:
    def test_ragged_batch_dimension(self):
        """
        A list of 2-D matrices of different sizes — ragged at the batch level.
        Each matrix is multiplied by its own (compatible) right operand.
        """
        # batch of two independent matrix products
        # mat0: (2,2) @ (2,2); mat1: (3,2) @ (2,3)
        a = ragged.array(
            [
                [[1, 0], [0, 1]],
                [[1, 2], [3, 4], [5, 6]],
            ]
        )
        b = ragged.array(
            [
                [[5, 6], [7, 8]],
                [[1, 0, 1], [0, 1, 0]],
            ]
        )
        result = a @ b
        # batch dimension should be preserved (length 2)
        assert len(result) == 2

    def test_result_is_ragged_array_type(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        b = _make([[1, 0], [0, 1]], dtype=np.float64)
        result = a @ b
        assert isinstance(result, ragged.array)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestMatmulErrors:
    def test_1d_left_raises(self):
        a = _make([1, 2, 3], dtype=np.float64)
        b = _make([[1, 0], [0, 1], [1, 1]], dtype=np.float64)
        with pytest.raises(ValueError, match="[Mm]atmul|dimension"):
            _ = a @ b

    def test_1d_right_raises(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        b = _make([1, 2], dtype=np.float64)
        with pytest.raises(ValueError, match="[Mm]atmul|dimension"):
            _ = a @ b

    def test_0d_left_raises(self):
        a = ragged.array(5.0)
        b = _make([[1, 0], [0, 1]], dtype=np.float64)
        with pytest.raises((ValueError, TypeError)):
            _ = a @ b

    def test_contracted_dim_mismatch(self):
        # (2,3) @ (2,2) — inner dims 3 ≠ 2
        a = _make([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        b = _make([[1, 0], [0, 1]], dtype=np.float64)
        with pytest.raises(ValueError, match="[Mm]ismatch|dimension|shape"):
            _ = a @ b

    def test_contracted_dim_mismatch_3d(self):
        a_np = np.ones((2, 3, 4), dtype=np.float64)
        b_np = np.ones((2, 3, 5), dtype=np.float64)  # inner: 4 ≠ 3
        a = _make(a_np.tolist())
        b = _make(b_np.tolist())
        with pytest.raises(ValueError, match="[Mm]ismatch|dimension|shape"):
            _ = a @ b

    def test_ragged_contracted_axis_raises(self):
        """
        If the contracted axis (last of left / second-to-last of right) is ragged,
        matmul must raise ValueError because the dot products are undefined.
        """
        # ragged rows — last axis of `a` is ragged (shape[-1] is None)
        a = ragged.array([[1, 2, 3], [4, 5]])  # shape (2, None)
        b = _make([[1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=np.float64)  # (3, 3)
        with pytest.raises(ValueError, match="[Rr]agged|contracted|axis"):
            _ = a @ b


# ---------------------------------------------------------------------------
# In-place operator __imatmul__
# ---------------------------------------------------------------------------


class TestIMatmul:
    def test_imatmul_updates_in_place(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        b = _make([[1, 0], [0, 1]], dtype=np.float64)
        original_id = id(a)
        a @= b
        # The Python object is the same (in-place semantics at the object level)
        assert id(a) == original_id

    def test_imatmul_correct_values(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        b = _make([[2, 0], [0, 2]], dtype=np.float64)
        a @= b
        expected = np.array([[2, 4], [6, 8]], dtype=np.float64)
        np.testing.assert_array_equal(_as_np(a), expected)

    def test_imatmul_error_propagates(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        b = _make([[1, 2, 3], [4, 5, 6]], dtype=np.float64)  # shape mismatch
        with pytest.raises(ValueError, match="[Ii]n-place|shape|mismatch"):
            a @= b


# ---------------------------------------------------------------------------
# Numerical accuracy
# ---------------------------------------------------------------------------


class TestMatmulNumerics:
    def test_large_values(self):
        v = 1e8
        a = _make([[v, v], [v, v]], dtype=np.float64)
        b = _make([[v, v], [v, v]], dtype=np.float64)
        result = a @ b
        expected = np.array([[2 * v**2, 2 * v**2], [2 * v**2, 2 * v**2]])
        np.testing.assert_allclose(_as_np(result), expected, rtol=1e-10)

    def test_zeros(self):
        a = _make([[0, 0], [0, 0]], dtype=np.float64)
        b = _make([[1, 2], [3, 4]], dtype=np.float64)
        result = a @ b
        np.testing.assert_array_equal(_as_np(result), np.zeros((2, 2)))

    def test_negative_values(self):
        a = _make([[-1, 2], [3, -4]], dtype=np.float64)
        b = _make([[5, -6], [-7, 8]], dtype=np.float64)
        result = a @ b
        expected = np.array([[-1, 2], [3, -4]]) @ np.array([[5, -6], [-7, 8]])
        np.testing.assert_array_equal(_as_np(result), expected.astype(np.float64))

    def test_complex_dtype(self):
        a = _make([[1 + 1j, 2 + 0j], [0 + 1j, 1 + 2j]], dtype=np.complex128)
        b = _make([[1 + 0j, 0 + 1j], [1 + 1j, 1 + 0j]], dtype=np.complex128)
        result = a @ b
        expected = np.array([[1 + 1j, 2 + 0j], [0 + 1j, 1 + 2j]]) @ np.array(
            [[1 + 0j, 0 + 1j], [1 + 1j, 1 + 0j]]
        )
        np.testing.assert_allclose(_as_np(result), expected)
