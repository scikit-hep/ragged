# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
Tests for ragged.meshgrid.

Coverage
--------
Cartesian "xy" indexing (default)
  - 2-D case: shape (N2, N1) for each output
  - 3-D case: shape (N2, N1, N3) for each output
  - values match numpy reference

Matrix "ij" indexing
  - 2-D case: shape (N1, N2) for each output
  - 3-D case: shape (N1, N2, N3)

Edge cases
  - zero inputs -> empty list
  - one input -> single 1-D array returned unchanged

dtype
  - output dtype follows np.result_type promotion
  - int input gives int output

result type
  - each returned item is ragged.array

error paths
  - non-1-D input raises ValueError
  - invalid indexing string raises ValueError
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
# Cartesian "xy" indexing
# ---------------------------------------------------------------------------


class TestMeshgridXY:
    def test_2d_shapes(self):
        x = _make([1, 2, 3], dtype=np.float64)  # len 3
        y = _make([4, 5], dtype=np.float64)  # len 2
        xi, yi = ragged.meshgrid(x, y)
        # xy: shape is (N2, N1) = (2, 3)
        assert xi.shape == (2, 3)
        assert yi.shape == (2, 3)

    def test_2d_values(self):
        x = _make([1, 2, 3], dtype=np.float64)
        y = _make([4, 5], dtype=np.float64)
        xi, yi = ragged.meshgrid(x, y)
        xi_np, yi_np = np.meshgrid([1, 2, 3], [4, 5], indexing="xy")
        np.testing.assert_array_equal(_np(xi), xi_np)
        np.testing.assert_array_equal(_np(yi), yi_np)

    def test_3d_shapes(self):
        x = _make([1, 2], dtype=np.float64)  # N1=2
        y = _make([3, 4, 5], dtype=np.float64)  # N2=3
        z = _make([6, 7], dtype=np.float64)  # N3=2
        xi, yi, zi = ragged.meshgrid(x, y, z)
        # xy: shape is (N2, N1, N3) = (3, 2, 2)
        assert xi.shape == (3, 2, 2)
        assert yi.shape == (3, 2, 2)
        assert zi.shape == (3, 2, 2)

    def test_2d_is_default_indexing(self):
        x = _make([1.0, 2.0, 3.0])
        y = _make([4.0, 5.0])
        xi_default, yi_default = ragged.meshgrid(x, y)
        xi_xy, yi_xy = ragged.meshgrid(x, y, indexing="xy")
        np.testing.assert_array_equal(_np(xi_default), _np(xi_xy))
        np.testing.assert_array_equal(_np(yi_default), _np(yi_xy))

    def test_returns_list_of_ragged_arrays(self):
        x = _make([1.0, 2.0])
        y = _make([3.0, 4.0, 5.0])
        result = ragged.meshgrid(x, y)
        assert isinstance(result, list)
        assert all(isinstance(g, ragged.array) for g in result)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Matrix "ij" indexing
# ---------------------------------------------------------------------------


class TestMeshgridIJ:
    def test_2d_shapes(self):
        x = _make([1, 2, 3], dtype=np.float64)  # N1=3
        y = _make([4, 5], dtype=np.float64)  # N2=2
        xi, yi = ragged.meshgrid(x, y, indexing="ij")
        # ij: shape is (N1, N2) = (3, 2)
        assert xi.shape == (3, 2)
        assert yi.shape == (3, 2)

    def test_2d_values(self):
        x = _make([1, 2, 3], dtype=np.float64)
        y = _make([4, 5], dtype=np.float64)
        xi, yi = ragged.meshgrid(x, y, indexing="ij")
        xi_np, yi_np = np.meshgrid([1, 2, 3], [4, 5], indexing="ij")
        np.testing.assert_array_equal(_np(xi), xi_np)
        np.testing.assert_array_equal(_np(yi), yi_np)

    def test_3d_shapes(self):
        x = _make([1, 2], dtype=np.float64)  # N1=2
        y = _make([3, 4, 5], dtype=np.float64)  # N2=3
        z = _make([6, 7], dtype=np.float64)  # N3=2
        xi, yi, zi = ragged.meshgrid(x, y, z, indexing="ij")
        # ij: shape is (N1, N2, N3) = (2, 3, 2)
        assert xi.shape == (2, 3, 2)
        assert yi.shape == (2, 3, 2)
        assert zi.shape == (2, 3, 2)

    def test_3d_values(self):
        x = _make([1, 2], dtype=np.float64)
        y = _make([3, 4, 5], dtype=np.float64)
        z = _make([6, 7], dtype=np.float64)
        xi, yi, zi = ragged.meshgrid(x, y, z, indexing="ij")
        xi_np, yi_np, zi_np = np.meshgrid([1, 2], [3, 4, 5], [6, 7], indexing="ij")
        np.testing.assert_array_equal(_np(xi), xi_np.astype(np.float64))
        np.testing.assert_array_equal(_np(yi), yi_np.astype(np.float64))
        np.testing.assert_array_equal(_np(zi), zi_np.astype(np.float64))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestMeshgridEdge:
    def test_zero_inputs(self):
        result = ragged.meshgrid()
        assert result == []

    def test_one_input(self):
        x = _make([10, 20, 30], dtype=np.float64)
        (r,) = ragged.meshgrid(x)
        np.testing.assert_array_equal(_np(r), [10.0, 20.0, 30.0])
        assert r.shape == (3,)


# ---------------------------------------------------------------------------
# dtype
# ---------------------------------------------------------------------------


class TestMeshgridDtype:
    def test_float64_output(self):
        x = _make([1.0, 2.0], dtype=np.float64)
        y = _make([3.0, 4.0], dtype=np.float64)
        xi, yi = ragged.meshgrid(x, y)
        assert xi.dtype == np.float64
        assert yi.dtype == np.float64

    def test_int_input_gives_int_output(self):
        x = _make([1, 2, 3], dtype=np.int64)
        y = _make([4, 5], dtype=np.int64)
        xi, yi = ragged.meshgrid(x, y)
        assert np.issubdtype(xi.dtype, np.integer)

    def test_promotion_int_float(self):
        x = _make([1, 2], dtype=np.int32)
        y = _make([3.0, 4.0], dtype=np.float64)
        xi, yi = ragged.meshgrid(x, y)
        assert np.issubdtype(xi.dtype, np.floating)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestMeshgridErrors:
    def test_non_1d_input_raises(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float64)
        b = _make([1.0, 2.0])
        with pytest.raises(ValueError, match="1-dimensional|ndim"):
            ragged.meshgrid(a, b)

    def test_invalid_indexing_raises(self):
        x = _make([1.0, 2.0])
        with pytest.raises(ValueError, match="indexing"):
            ragged.meshgrid(x, indexing="bad")

    def test_non_array_input_raises(self):
        with pytest.raises(TypeError, match="ragged.array"):
            ragged.meshgrid([1, 2, 3])  # type: ignore[arg-type]
