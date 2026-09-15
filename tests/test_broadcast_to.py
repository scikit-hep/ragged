# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
Tests for ragged.broadcast_to.

Coverage
--------
1-D arrays
  - identity: (n,) -> (n,)
  - scalar-like broadcast: (1,) -> (n,)
  - outer replication: (n,) -> (m, n)

2-D uniform arrays (fast path via numpy)
  - identity: (m, n) -> (m, n)
  - outer replication: (m, n) -> (k, m, n)
  - outer dim 1 -> k: (1, n) -> (k, n)

2-D ragged arrays
  - identity: (m, None) -> (m, None)
  - outer replication: (m, None) -> (k, m, None)
  - outer dim 1 -> k: (1, None) -> (k, None)

3-D uniform arrays
  - outer replication: (m, n, p) -> (k, m, n, p)

result type
  - always returns ragged.array
  - dtype preserved

error paths
  - non-tuple shape raises TypeError
  - non-int dim raises TypeError
  - negative dim raises ValueError
  - fewer dims than x raises ValueError
  - incompatible non-broadcastable dim raises ValueError
  - non-array input raises TypeError
"""

from __future__ import annotations

from typing import Any

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


class TestBroadcastTo1D:
    def test_identity(self):
        a = _make([1.0, 2.0, 3.0])
        result = ragged.broadcast_to(a, (3,))
        np.testing.assert_array_equal(_np(result), [1.0, 2.0, 3.0])

    def test_single_elem_to_n(self):
        a = _make([7.0])
        result = ragged.broadcast_to(a, (4,))
        np.testing.assert_array_equal(_np(result), [7.0, 7.0, 7.0, 7.0])

    def test_outer_replication(self):
        a = _make([1.0, 2.0, 3.0])
        result = ragged.broadcast_to(a, (2, 3))
        np.testing.assert_array_equal(_np(result), [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])

    def test_outer_replication_3d(self):
        a = _make([1.0, 2.0])
        result = ragged.broadcast_to(a, (3, 4, 2))
        assert result.shape == (3, 4, 2)
        result_list: list[Any] = result.tolist()  # type: ignore[assignment]
        for i in range(3):
            for j in range(4):
                assert result_list[i][j] == [1.0, 2.0]

    def test_result_is_ragged_array(self):
        a = _make([1.0, 2.0])
        assert isinstance(ragged.broadcast_to(a, (2, 2)), ragged.array)

    def test_dtype_preserved_float32(self):
        a = _make([1.0, 2.0], dtype=np.float32)
        assert ragged.broadcast_to(a, (2, 2)).dtype == np.float32

    def test_dtype_preserved_int64(self):
        a = _make([1, 2, 3], dtype=np.int64)
        assert ragged.broadcast_to(a, (3, 3)).dtype == np.int64


# ---------------------------------------------------------------------------
# 2-D uniform arrays (fast path)
# ---------------------------------------------------------------------------


class TestBroadcastTo2DUniform:
    def test_identity(self):
        a_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        a = _make(a_np.tolist(), dtype=np.float64)
        result = ragged.broadcast_to(a, (2, 3))
        np.testing.assert_array_equal(_np(result), a_np)

    def test_outer_replication(self):
        a_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        a = _make(a_np.tolist(), dtype=np.float64)
        result = ragged.broadcast_to(a, (4, 2, 3))
        # The 2-D source's own inner dimension is variable-length, so it stays
        # ``None``; the newly prepended outer dimension is regular. This matches
        # the convention produced for a genuinely ragged input.
        assert result.shape == (4, 2, None)
        result_list: list[Any] = result.tolist()  # type: ignore[assignment]
        for i in range(4):
            np.testing.assert_array_equal(np.array(result_list[i]), a_np)

    def test_outer_dim_1_to_k(self):
        a_np = np.array([[1, 2, 3]], dtype=np.float64)
        a = _make(a_np.tolist(), dtype=np.float64)
        result = ragged.broadcast_to(a, (5, 3))
        expected = np.broadcast_to(a_np, (5, 3))
        np.testing.assert_array_equal(_np(result), expected)

    def test_dtype_preserved(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float32)
        assert ragged.broadcast_to(a, (3, 2, 2)).dtype == np.float32


# ---------------------------------------------------------------------------
# 2-D ragged arrays
# ---------------------------------------------------------------------------


class TestBroadcastTo2DRagged:
    def test_identity(self):
        a = _make([[1, 2, 3], [4, 5]], dtype=np.float64)
        result = ragged.broadcast_to(a, (2, None))
        assert result.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0]]

    def test_outer_replication(self):
        a = _make([[1, 2, 3], [4, 5]], dtype=np.float64)
        result = ragged.broadcast_to(a, (3, 2, None))
        assert result.shape == (3, 2, None)
        inner = [[1.0, 2.0, 3.0], [4.0, 5.0]]
        assert result.tolist() == [inner, inner, inner]

    def test_outer_dim_1_to_k(self):
        a = _make([[1, 2, 3]], dtype=np.float64)
        result = ragged.broadcast_to(a, (4, None))
        assert result.shape == (4, None)
        assert result.tolist() == [[1.0, 2.0, 3.0]] * 4

    def test_dtype_preserved(self):
        a = _make([[1.5, 2.5], [3.5]], dtype=np.float64)
        assert ragged.broadcast_to(a, (2, 2, None)).dtype == np.float64

    def test_single_row(self):
        a = _make([[10, 20]], dtype=np.int64)
        result = ragged.broadcast_to(a, (3, None))
        assert result.tolist() == [[10, 20], [10, 20], [10, 20]]

    def test_empty_rows_preserved(self):
        a = _make([[], [1, 2]], dtype=np.float64)
        result = ragged.broadcast_to(a, (2, None))
        assert result.tolist() == [[], [1.0, 2.0]]  # type: ignore[comparison-overlap]


# ---------------------------------------------------------------------------
# 3-D uniform arrays
# ---------------------------------------------------------------------------


class TestBroadcastTo3DUniform:
    def test_outer_replication(self):
        a_np = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        a = _make(a_np.tolist(), dtype=np.float64)
        result = ragged.broadcast_to(a, (5, 2, 3, 4))
        # The 3-D source has shape (2, None, None); its variable-length inner
        # dimensions are preserved and the new outer dimension is regular.
        assert result.shape == (5, 2, None, None)
        result_list: list[Any] = result.tolist()  # type: ignore[assignment]
        for i in range(5):
            np.testing.assert_array_equal(np.array(result_list[i]), a_np)

    def test_identity(self):
        a_np = np.arange(6, dtype=np.float64).reshape(2, 3)
        a = _make(a_np.tolist(), dtype=np.float64)
        result = ragged.broadcast_to(a, (2, 3))
        np.testing.assert_array_equal(_np(result), a_np)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestBroadcastToErrors:
    def test_non_tuple_shape_raises(self):
        a = _make([1.0, 2.0])
        with pytest.raises(TypeError, match="tuple"):
            ragged.broadcast_to(a, [2, 2])  # type: ignore[arg-type]

    def test_non_int_dim_raises(self):
        a = _make([1.0, 2.0])
        with pytest.raises(TypeError, match="int"):
            ragged.broadcast_to(a, (2, "3"))  # type: ignore[arg-type]

    def test_negative_dim_raises(self):
        a = _make([1.0, 2.0])
        with pytest.raises(ValueError, match="non-negative"):
            ragged.broadcast_to(a, (-1, 2))

    def test_fewer_dims_raises(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        with pytest.raises(ValueError, match="fewer dimensions"):
            ragged.broadcast_to(a, (2,))

    def test_incompatible_dim_raises(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        with pytest.raises(ValueError, match="broadcast"):
            ragged.broadcast_to(a, (3, 2))

    def test_non_array_input_raises(self):
        with pytest.raises(TypeError):
            ragged.broadcast_to([[1, 2], [3, 4]], (2, 2))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Shape convention: a uniform (numpy-convertible) source must broadcast to the
# same shape signature as a genuinely ragged source. The source array's own
# inner dimensions stay variable-length (``None``); newly prepended outer
# dimensions are regular.
# ---------------------------------------------------------------------------


class TestBroadcastToShapeConvention:
    def test_uniform_2d_inner_dim_none(self):
        a = _make([[1.0, 2.0], [3.0, 4.0]])  # uniform data, ragged layout
        assert a.shape == (2, None)
        assert ragged.broadcast_to(a, (2, 2)).shape == (2, None)

    def test_uniform_outer_replication_inner_none(self):
        a = _make([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        assert ragged.broadcast_to(a, (4, 2, 3)).shape == (4, 2, None)

    def test_uniform_matches_ragged_signature(self):
        # Broadcasting a single ragged-typed row outward yields the same shape
        # whether the row data is uniform or genuinely ragged.
        uniform = _make([[1.0, 2.0]])  # shape (1, None)
        assert ragged.broadcast_to(uniform, (3, None)).shape == (3, None)
        assert ragged.broadcast_to(uniform, (3, 2)).shape == (3, None)
