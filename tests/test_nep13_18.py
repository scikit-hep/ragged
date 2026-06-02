# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
Tests for NEP-13 (__array_ufunc__) and NEP-18 (__array_function__) support.

Coverage
--------
NEP-13 (__array_ufunc__)
  - unary ufunc: np.sqrt, np.abs, np.negative
  - binary ufunc: np.add, np.multiply, np.subtract
  - binary with a plain numpy array on one side
  - result is ragged.array, dtype and values correct
  - multi-output ufunc: np.modf
  - 2-D ragged inputs

NEP-18 (__array_function__)
  - np.concatenate -> ragged.concat
  - np.stack       -> ragged.stack
  - np.reshape     -> ragged.reshape
  - np.squeeze / np.expand_dims
  - np.flip
  - np.roll
  - np.broadcast_to
  - fallback: np.sum, np.unique (delegated through awkward)

result type
  - __array_ufunc__ always returns ragged.array (or tuple thereof)
  - __array_function__ mapped functions return ragged.array
"""

from __future__ import annotations

import numpy as np
import pytest

import ragged

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make(data, dtype=None) -> ragged.array:
    return ragged.array(data, dtype=dtype)


# ---------------------------------------------------------------------------
# NEP-13: __array_ufunc__
# ---------------------------------------------------------------------------


class TestArrayUfunc:
    # --- unary ufuncs ---

    def test_sqrt_1d(self):
        a = _make([1.0, 4.0, 9.0])
        result = np.sqrt(a)
        assert isinstance(result, ragged.array)
        np.testing.assert_allclose(result.tolist(), [1.0, 2.0, 3.0])

    def test_abs_negative(self):
        a = _make([-1.0, 2.0, -3.0])
        result = np.abs(a)
        assert isinstance(result, ragged.array)
        np.testing.assert_array_equal(result.tolist(), [1.0, 2.0, 3.0])

    def test_negative_ufunc(self):
        a = _make([1.0, 2.0, 3.0])
        result = np.negative(a)
        assert isinstance(result, ragged.array)
        np.testing.assert_array_equal(result.tolist(), [-1.0, -2.0, -3.0])

    def test_sqrt_2d_uniform(self):
        a = _make([[1.0, 4.0], [9.0, 16.0]])
        result = np.sqrt(a)
        assert isinstance(result, ragged.array)
        assert result.tolist() == [[1.0, 2.0], [3.0, 4.0]]

    def test_sqrt_2d_ragged(self):
        a = _make([[1.0, 4.0, 9.0], [16.0]], dtype=np.float64)
        result = np.sqrt(a)
        assert isinstance(result, ragged.array)
        assert result.tolist() == [[1.0, 2.0, 3.0], [4.0]]

    # --- binary ufuncs ---

    def test_add_two_ragged(self):
        a = _make([1.0, 2.0, 3.0])
        b = _make([10.0, 20.0, 30.0])
        result = np.add(a, b)
        assert isinstance(result, ragged.array)
        np.testing.assert_array_equal(result.tolist(), [11.0, 22.0, 33.0])

    def test_multiply_ragged_scalar(self):
        a = _make([1.0, 2.0, 3.0])
        result = np.multiply(a, 2.0)
        assert isinstance(result, ragged.array)
        np.testing.assert_array_equal(result.tolist(), [2.0, 4.0, 6.0])

    def test_add_ragged_numpy(self):
        a = _make([1.0, 2.0, 3.0])
        b = np.array([10.0, 20.0, 30.0])
        result = np.add(a, b)
        assert isinstance(result, ragged.array)
        np.testing.assert_array_equal(result.tolist(), [11.0, 22.0, 33.0])

    def test_subtract(self):
        a = _make([5.0, 6.0, 7.0])
        b = _make([1.0, 2.0, 3.0])
        result = np.subtract(a, b)
        assert isinstance(result, ragged.array)
        np.testing.assert_array_equal(result.tolist(), [4.0, 4.0, 4.0])

    # --- dtype preservation ---

    def test_dtype_float32_preserved(self):
        a = _make([1.0, 4.0, 9.0], dtype=np.float32)
        result = np.sqrt(a)
        assert result.dtype == np.float32

    # --- multi-output ufunc ---

    def test_modf_returns_tuple(self):
        a = _make([1.5, 2.7, 3.9])
        frac, integ = np.modf(a)
        assert isinstance(frac, ragged.array)
        assert isinstance(integ, ragged.array)
        np.testing.assert_allclose(frac.tolist(), [0.5, 0.7, 0.9], atol=1e-6)
        np.testing.assert_allclose(integ.tolist(), [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# NEP-18: __array_function__
# ---------------------------------------------------------------------------


class TestArrayFunction:
    def test_concatenate(self):
        a = _make([1.0, 2.0])
        b = _make([3.0, 4.0])
        result = np.concatenate([a, b])
        assert isinstance(result, ragged.array)
        assert result.tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_stack_axis0(self):
        a = _make([1.0, 2.0])
        b = _make([3.0, 4.0])
        result = np.stack([a, b], axis=0)
        assert isinstance(result, ragged.array)
        assert result.tolist() == [[1.0, 2.0], [3.0, 4.0]]

    def test_reshape(self):
        a = _make(np.arange(6, dtype=np.float64).tolist())
        result = np.reshape(a, (2, 3))
        assert isinstance(result, ragged.array)
        assert result.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]

    def test_expand_dims(self):
        a = _make([1.0, 2.0, 3.0])
        result = np.expand_dims(a, axis=0)
        assert isinstance(result, ragged.array)
        assert result.shape == (1, 3)

    def test_squeeze(self):
        a = _make([[1.0, 2.0, 3.0]])
        expanded = np.expand_dims(a, axis=0)
        result = np.squeeze(expanded, axis=0)
        assert isinstance(result, ragged.array)
        assert result.tolist() == [[1.0, 2.0, 3.0]]

    def test_flip(self):
        a = _make([1.0, 2.0, 3.0])
        result = np.flip(a)
        assert isinstance(result, ragged.array)
        assert result.tolist() == [3.0, 2.0, 1.0]

    def test_roll(self):
        a = _make([1.0, 2.0, 3.0, 4.0])
        result = np.roll(a, 1)
        assert isinstance(result, ragged.array)
        assert result.tolist() == [4.0, 1.0, 2.0, 3.0]

    def test_broadcast_to_1d(self):
        a = _make([1.0, 2.0, 3.0])
        result = np.broadcast_to(a, (3,))
        assert isinstance(result, ragged.array)
        assert result.tolist() == [1.0, 2.0, 3.0]

    def test_concatenate_ragged(self):
        a = _make([[1.0, 2.0], [3.0]])
        b = _make([[4.0], [5.0, 6.0]])
        result = np.concatenate([a, b], axis=0)
        assert isinstance(result, ragged.array)
        assert result.tolist() == [[1.0, 2.0], [3.0], [4.0], [5.0, 6.0]]

    # --- fallback path (awkward delegation) ---

    def test_sum_fallback(self):
        """np.sum is not in the ragged map; falls back through awkward."""
        a = _make([1.0, 2.0, 3.0])
        result = np.sum(a)
        # awkward returns a scalar or 0-d array — just check the value
        assert float(result) == pytest.approx(6.0)
